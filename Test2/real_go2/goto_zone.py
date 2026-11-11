#!/usr/bin/env python3
"""Autonomous "go to zone" on the real Go2, driven by the trained zone classifier (real-world Q1, step 5).

    python real_go2/goto_zone.py --source sdk --iface eth1                    # then press 1-4 to pick a zone
    python real_go2/goto_zone.py --source sdk --iface eth1 --target shelves   # start straight away

How it finds a zone, using only the camera classifier (labels = where the dog is, as when recording):
  LOOK      turn once on the spot, classifying every frame. Two answers come out of one turn:
            * where am I: the zone most of the views belong to (averaged over all directions);
            * which way: the heading in which the classifier is most confident in the TARGET zone.
            If the target zone holds >= navigation.arrive_share of the views -> ARRIVED (waves hello).
  TURN      turn to the best heading.
  WALK      walk forward for one leg (navigation.leg_s), holding the heading. The leg ends early when
            the LiDAR sees something within stop_obstacle_m ahead, when the target is predicted steadily
            (early_stop_*), or when it fades (lost_*). Then LOOK again. Gives up after max_legs.
Safety: the dog moves only through Test 1's teleop autonomy hook (go2_stream/teleop.py set_autonomy):
requests lapse unless refreshed, are clamped to the teleop limits, and ANY manual key (SPACE, W/A/S/D/Q/E,
ESC) cancels the mission at once. Walking needs LiDAR (Ethernet / --source sdk), as Test 1's missions do;
over Wi-Fi it only runs with --allow-no-lidar, slower and with shorter legs.
Keys (window): 1..N go to that zone, C cancel, V start/stop the MP4, ESC quit.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.live_zone import COLORS, PANEL_W, VIEW_H, VIEW_W, Classifier, put_text  # noqa: E402
from real_go2.sources import add_source_args, open_link, start_teleop  # noqa: E402

log = logging.getLogger("goto_zone")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ZoneNavigator:
    """The decision logic, independent of the robot: feed it classifier outputs, obstacle distances and the
    heading; it returns velocity requests (vx, wz). States: idle, look, turn, walk, arrived, failed."""

    def __init__(self, classes, nav, have_lidar):
        self.classes, self.nav, self.have_lidar = list(classes), dict(nav), have_lidar
        if not have_lidar:
            self.nav["approach_speed"] = min(self.nav["approach_speed"], 0.15)
            self.nav["leg_s"] = min(self.nav["leg_s"], 2.0)
        self.state, self.target, self.msg = "idle", None, "press 1-%d to choose a zone" % len(classes)
        self.legs = 0
        self.scan = []                      # (heading, prob vector) during LOOK
        self.best_heading, self.where, self.share, self.bins = None, None, 0.0, None
        self._recent = deque()              # (t, prob of target) during WALK
        self._t0 = 0.0
        self._start_heading = 0.0

    # ------------------------------------------------------------------ commands
    def start(self, k, now, heading):
        self.target, self.legs = int(k), 0
        self.msg = f"going to {self.classes[k]}"
        self._look(now, heading)

    def cancel(self, why="cancelled"):
        if self.state not in ("idle", "arrived", "failed"):
            self.state, self.msg = "idle", why

    @property
    def active(self):
        return self.state in ("look", "turn", "walk")

    # ------------------------------------------------------------------ states
    def _look(self, now, heading):
        self.state, self.scan, self._t0, self._start_heading = "look", [], now, heading

    def _finish_look(self, now, heading):
        nav, k = self.nav, self.target
        probs = np.array([p for _, p in self.scan])
        heads = np.array([(h - self._start_heading) % (2 * math.pi) for h, _ in self.scan])
        votes = np.bincount(probs.argmax(1), minlength=len(self.classes)) / len(probs)
        self.where, self.share = int(votes.argmax()), float(votes[k])
        nb = nav["heading_bins"]
        b = (heads / (2 * math.pi) * nb).astype(int) % nb
        sums, cnt = np.bincount(b, probs[:, k], nb), np.bincount(b, minlength=nb)
        mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        filled = np.where(np.isnan(mean), np.nanmean(mean), mean)
        smooth = (np.roll(filled, 1) + filled + np.roll(filled, -1)) / 3        # circular, +-1 bin
        self.bins = smooth
        best = int(np.argmax(smooth))
        self.best_heading = self._start_heading + (best + 0.5) / nb * 2 * math.pi
        if self.share >= nav["arrive_share"]:
            self.state = "arrived"
            self.msg = f"ARRIVED at {self.classes[k]} ({self.share:.0%} of the views)"
        elif smooth[best] < nav["seen_conf"]:
            self.state = "failed"
            self.msg = f"{self.classes[k]} not seen in any direction (best {smooth[best]:.2f})"
        elif self.legs >= nav["max_legs"]:
            self.state = "failed"
            self.msg = f"gave up after {self.legs} legs (here looks like {self.classes[self.where]})"
        else:
            self.state = "turn"
            self.msg = (f"here looks like {self.classes[self.where]}; {self.classes[k]} is best at "
                        f"{math.degrees(wrap(self.best_heading - heading)):+.0f} deg")

    def update(self, now, heading, prob=None, obstacle_m=None):
        """One control step. prob: latest classifier output (None if no new frame); obstacle_m: LiDAR distance
        to the nearest obstacle ahead (None = clear, 0.0 = no LiDAR data = blocked). Returns (vx, wz)."""
        nav, k = self.nav, self.target
        if self.state == "look":
            if prob is not None:
                self.scan.append((heading, np.asarray(prob, float)))
            turned = sum(abs(wrap(b[0] - a[0])) for a, b in zip(self.scan, self.scan[1:]))
            if turned >= 2 * math.pi - 0.05 and len(self.scan) >= 8:
                self._finish_look(now, heading)
                return 0.0, 0.0
            return 0.0, nav["scan_rate"]
        if self.state == "turn":
            err = wrap(self.best_heading - heading)
            if abs(err) < 0.08:
                self.state, self.legs, self._t0, self._recent = "walk", self.legs + 1, now, deque()
                self.msg = f"walking towards {self.classes[k]} (leg {self.legs})"
                return 0.0, 0.0
            return 0.0, float(np.clip(1.2 * err, -nav["scan_rate"], nav["scan_rate"]))
        if self.state == "walk":
            if prob is not None:
                self._recent.append((now, float(prob[k]), int(np.argmax(prob)) == k))
            while self._recent and now - self._recent[0][0] > max(nav["early_stop_s"], nav["lost_s"]):
                self._recent.popleft()
            end = None
            if self.have_lidar and obstacle_m is not None:
                end = "no LiDAR data" if obstacle_m <= 0.0 else (
                    f"obstacle {obstacle_m:.2f} m ahead" if obstacle_m <= nav["stop_obstacle_m"] else None)
            span = now - self._recent[0][0] if self._recent else 0.0
            if end is None and span >= nav["early_stop_s"] * 0.9:
                w = [r for r in self._recent if now - r[0] <= nav["early_stop_s"]]
                if all(r[2] for r in w) and np.mean([r[1] for r in w]) >= nav["early_stop_conf"]:
                    end = f"{self.classes[k]} recognised steadily"
            if end is None and span >= nav["lost_s"] * 0.9:
                w = [r for r in self._recent if now - r[0] <= nav["lost_s"]]
                if max(r[1] for r in w) < nav["lost_conf"]:
                    end = f"lost sight of {self.classes[k]}"
            if end is None and now - self._t0 >= nav["leg_s"]:
                end = "leg done"
            if end is not None:
                self.msg = f"{end}; looking around"
                self._look(now, heading)
                return 0.0, 0.0
            return nav["approach_speed"], float(np.clip(0.8 * wrap(self.best_heading - heading), -0.3, 0.3))
        return 0.0, 0.0


class Heading:
    """Heading for the navigator: the dog's IMU yaw when available (Ethernet), else the integral of the
    commanded turn rate (Wi-Fi / fake: approximate)."""

    def __init__(self, pose_fn=None):
        self.pose_fn, self.h, self.t = pose_fn, 0.0, None

    def __call__(self, now, wz_cmd):
        if self.pose_fn is not None:
            p = self.pose_fn()
            if p is not None:
                return float(p[2])
        if self.t is not None:
            self.h = wrap(self.h + wz_cmd * (now - self.t))
        self.t = now
        return self.h


def nav_panel(nav, classes, prob, fps, recording):
    p = np.full((VIEW_H, PANEL_W, 3), 24, np.uint8)
    put_text(p, "Go to zone - real Go2", (12, 26), 0.6, (255, 255, 255))
    put_text(p, "ResNet18 zone classifier + LiDAR safety", (12, 46), 0.42, (170, 170, 170))
    tgt = classes[nav.target] if nav.target is not None else "-"
    put_text(p, f"target: {tgt}", (12, 80), 0.75, COLORS[nav.target % len(COLORS)] if nav.target is not None
             else (200, 200, 200), 2)
    put_text(p, f"state: {nav.state.upper()}   legs {nav.legs}", (12, 108), 0.5)
    words, line, y = nav.msg.split(), "", 130
    for w in words:                                          # wrap the status message
        if len(line + " " + w) > 40:
            put_text(p, line, (12, y), 0.42, (200, 200, 200))
            line, y = w, y + 18
        else:
            line = (line + " " + w).strip()
    put_text(p, line, (12, y), 0.42, (200, 200, 200))
    k = int(np.argmax(prob))
    put_text(p, f"camera sees: {classes[k]} ({prob[k]:.2f})", (12, 186), 0.5)
    if nav.bins is not None and nav.target is not None:      # last look-around: target confidence by heading
        cx, cy, r = PANEL_W // 2, 300, 70
        cv2.circle(p, (cx, cy), r, (70, 70, 70), 1)
        nb = len(nav.bins)
        for i, v in enumerate(nav.bins):
            a = -((i + 0.5) / nb * 2 * math.pi)              # drawn relative to the scan start, up = start
            rr = int(r * float(np.clip(v, 0, 1)))
            cv2.line(p, (cx, cy), (int(cx + rr * math.sin(-a)), int(cy - rr * math.cos(-a))),
                     COLORS[nav.target % len(COLORS)], 3)
        put_text(p, "last look-around: where the target looked likely", (12, 220), 0.4, (170, 170, 170))
        if nav.where is not None:
            put_text(p, f"here looked like {classes[nav.where]}; target share {nav.share:.0%}", (12, 385), 0.42)
    put_text(p, f"{fps:4.1f} fps   1-{len(classes)} go  C cancel  V video", (12, VIEW_H - 30), 0.42, (170, 170, 170))
    put_text(p, "REC" if recording else "not recording (V)", (12, VIEW_H - 10), 0.45,
             (60, 60, 235) if recording else (150, 150, 150))
    return p


def make_safety(link, go2cfg, allow_no_lidar):
    """(clearance_fn or None, pose_fn or None). Ethernet: Test 1's LiDAR corridor check and odometry heading."""
    if link.name == "sdk":
        import lidar as lidar_mod
        from behavior.clearance import make_real_clearance, make_real_pose_fn
        bcfg = yaml.safe_load((Path(lidar_mod.__file__).parent / "config/behaviors.yaml").read_text())
        src = lidar_mod.LidarSource(go2cfg["lidar"])
        return (make_real_clearance(src, link.state, lidar_mod, "auto", bcfg["safety"]["corridor"]),
                make_real_pose_fn(link.state))
    if not allow_no_lidar:
        raise SystemExit("Autonomous walking needs the LiDAR obstacle check (Ethernet, --source sdk). Over Wi-Fi, "
                         "run with --allow-no-lidar only in a clear area with someone ready to press SPACE.")
    log.warning("No LiDAR: obstacle check OFF, speed capped at 0.15 m/s, 2 s legs. Supervise closely.")
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--model", help="checkpoint (default: <results_dir>/resnet18/best.pt)")
    ap.add_argument("--target", help="zone to go to at start (name or 1-based number)")
    ap.add_argument("--allow-no-lidar", action="store_true", help="Wi-Fi / fake: walk without the obstacle check")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds")
    ap.add_argument("--video-fps", type=float, default=10.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = TEST2 / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    clf = Classifier(a.model or out / "resnet18/best.pt", cfg["crop"])
    classes = clf.classes
    link, go2cfg = open_link(a)
    a.no_control = False
    tele = start_teleop(link, go2cfg, a, keyboard=not a.headless)
    clearance, pose_fn = make_safety(link, go2cfg, a.allow_no_lidar)
    nav = ZoneNavigator(classes, cfg["navigation"], have_lidar=clearance is not None)
    heading = Heading(pose_fn)
    tele.on_manual_input = lambda key: nav.cancel(f"cancelled by key '{key}'")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out / f"goto_{stamp}.mp4"
    events = []
    writer, recording = None, a.headless
    prob, wz, last_seq, last_state = np.ones(len(classes)) / len(classes), 0.0, -1, "idle"
    t0 = time.monotonic()
    next_video = t0
    if a.target:
        k = classes.index(a.target) if a.target in classes else int(a.target) - 1
        nav.start(k, t0, heading(t0, 0.0))
    try:
        while True:
            now = time.monotonic()
            h = heading(now, wz)
            img, seq = link.frame()
            new = None
            if img is not None and seq != last_seq:
                last_seq = seq
                prob = new = clf(img)
            obstacle = clearance(None) if (clearance is not None and nav.state == "walk") else None
            vx, wz = nav.update(now, h, new, obstacle)
            if nav.active:
                tele.set_autonomy(vx, 0.0, wz)
            else:
                tele.clear_autonomy()
                wz = 0.0
            if nav.state != last_state:
                events.append({"t": round(now - t0, 2), "state": nav.state, "msg": nav.msg})
                log.info("%s: %s", nav.state, nav.msg)
                if nav.state == "arrived" and cfg["navigation"].get("hello_on_arrival"):
                    tele.request_action("hello")
                last_state = nav.state
            view = np.zeros((VIEW_H, VIEW_W, 3), np.uint8) if img is None else cv2.resize(img, (VIEW_W, VIEW_H))
            comp = np.hstack([view, nav_panel(nav, classes, prob, link.fps(), recording)])
            if recording:
                if writer is None:
                    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.video_fps,
                                             (comp.shape[1], comp.shape[0]))
                while now >= next_video:
                    writer.write(comp)
                    next_video += 1.0 / a.video_fps
            else:
                next_video = now
            if a.seconds and now - t0 >= a.seconds:
                break
            if a.headless:
                time.sleep(0.02)
                continue
            cv2.imshow("Go2 go to zone", comp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or tele.quit_event.is_set():
                break
            if ord("1") <= key < ord("1") + len(classes):
                nav.start(key - ord("1"), now, h)
            elif key in (ord("c"), ord("C")):
                nav.cancel()
            elif key in (ord("v"), ord("V")):
                recording = not recording
    except KeyboardInterrupt:
        pass
    finally:
        tele.clear_autonomy()
        tele.shutdown()
        link.close()
        if writer is not None:
            writer.release()
        if not a.headless:
            cv2.destroyAllWindows()
        (out / f"goto_{stamp}.json").write_text(json.dumps(
            {"classes": classes, "target": classes[nav.target] if nav.target is not None else None,
             "final_state": nav.state, "legs": nav.legs, "source": link.name, "lidar": clearance is not None,
             "video": mp4.name if writer is not None else None, "events": events}, indent=1))
        print(f"{nav.state}: {nav.msg}")


if __name__ == "__main__":
    main()
