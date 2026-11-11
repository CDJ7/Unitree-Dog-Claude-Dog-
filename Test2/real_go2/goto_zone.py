#!/usr/bin/env python3
"""Autonomous "go to zone" on the real Go2, driven by the trained zone classifier (real-world Q1, step 5).

    python real_go2/goto_zone.py --source sdk --iface eth1                    # then press 1-4 to pick a zone
    python real_go2/goto_zone.py --source sdk --iface eth1 --target shelves   # start straight away
    python real_go2/goto_zone.py --source sdk --iface eth1 --lidar-check      # no autonomy: check the LiDAR first

How it finds a zone (camera classifier) and avoids obstacles on the way (LiDAR):
  LOOK      turn once on the spot, classifying every frame and storing the LiDAR free distance per heading
            (Test 1's body corridor, floor ignored). Out of one turn come:
            * where am I: the zone most of the views belong to;
            * which way: heading score = P(target) x free-space factor x blocked-memory penalty, so a
              direction with a desk in the way loses even if the target is behind it;
            * if the target's way is blocked: DETOUR to the open heading (>= want_free_m) closest to the
              target, staying on the same side; if the target is not seen at all: EXPLORE open space
              (momentum + obstacle-edge bonus), at most explore_legs; nothing open: fail "boxed in".
            If the target zone holds >= navigation.arrive_share of the views -> ARRIVED (waves hello).
  TURN      turn to the chosen heading.
  WALK      walk forward for one leg (navigation.leg_s). Every LiDAR scan: within slow_obstacle_m ahead ->
            slow down and steer to the freer side; within stop_obstacle_m -> stop, remember that heading
            as blocked, LOOK again. The leg also ends when the target is predicted steadily (early_stop_*)
            or fades (lost_*). Gives up after max_legs. Closed-loop check: real_go2/nav_sim.py.
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


def free_profile(points, nb, corridor, lookahead):
    """Free walking distance (m) in nb directions around the dog, from LiDAR points in the body frame
    (x forward, y left, z up). Direction i is i * 360/nb degrees counter-clockwise from straight ahead.
    For each direction the points are rotated into that direction's frame and Test 1's walking corridor is
    applied (half_width_m, z_min..z_max, from x_min); the free distance is the min_points-th nearest
    obstacle in it, or `lookahead` when the corridor is clear that far."""
    out = np.full(nb, float(lookahead))
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1) & (p[:, 2] >= corridor["z_min"]) & (p[:, 2] <= corridor["z_max"])]
    if not len(p):
        return out
    th = np.arange(nb) * 2 * math.pi / nb
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    xr = c * p[None, :, 0] + s * p[None, :, 1]                # coordinates in each direction's frame
    yr = -s * p[None, :, 0] + c * p[None, :, 1]
    k = max(int(corridor["min_points"]), 1)
    for i in range(nb):
        m = (xr[i] >= corridor["x_min"]) & (xr[i] <= lookahead) & (np.abs(yr[i]) <= corridor["half_width_m"])
        if m.sum() >= k:
            out[i] = float(np.sort(xr[i][m])[k - 1])
    return out


class ZoneNavigator:
    """The decision logic, independent of the robot: feed it classifier outputs, the heading and (with LiDAR)
    the free-space profile; it returns velocity requests (vx, wz). States: idle, look, turn, walk, arrived,
    failed. Directions are absolute heading bins: bin b points at b * 360/heading_bins degrees.

    Obstacle avoidance is part of the path choice:
      * while looking around, every LiDAR profile is stored per absolute direction (the forward +-30 deg,
        where the sensor sees best), so a full turn maps the free distance in all directions;
      * the walking direction maximises P(target in that direction) x free-space factor, minus directions
        that blocked a recent leg; if the target's direction is blocked it takes a detour through the free
        direction closest to it;
      * while walking, an obstacle within slow_obstacle_m slows the dog and steers it to the freer side;
        within stop_obstacle_m the leg ends, that direction is remembered as blocked, and it plans again."""

    def __init__(self, classes, nav, have_lidar):
        self.classes, self.nav, self.have_lidar = list(classes), dict(nav), have_lidar
        if not have_lidar:
            self.nav["approach_speed"] = min(self.nav["approach_speed"], 0.15)
            self.nav["leg_s"] = min(self.nav["leg_s"], 2.0)
        self.nb = int(self.nav["heading_bins"])
        self.state, self.target, self.msg = "idle", None, "press 1-%d to choose a zone" % len(classes)
        self.legs = 0
        self.scan = []                      # (heading, prob vector) during LOOK
        self.best_heading, self.where, self.share = None, None, 0.0
        self.bins = self.free = self.score = None
        self.blocked = {}                   # absolute bin -> leg number when a leg ended blocked there
        self.detour = False
        self.exploring, self.explored = False, 0   # walking into open space because the target is not in view
        self._last_leg_heading = None
        self.detour_side = 0                # +1 going round obstacles on the left, -1 on the right, 0 none
        self._explore_heading = None
        self._world_free = np.full(self.nb, np.nan)
        self._recent = deque()              # (t, prob of target, target is argmax) during WALK
        self._t0 = 0.0
        self.avoiding = 0                   # -1 right, 0 none, +1 left (for the panel)
        self._last_free = (None, 0.0)       # latest LiDAR profile while walking (the LiDAR is slower than the loop)

    # ------------------------------------------------------------------ helpers
    def bin_of(self, heading):
        return int(round((heading % (2 * math.pi)) / (2 * math.pi) * self.nb)) % self.nb

    def heading_of(self, b):
        return b * 2 * math.pi / self.nb

    def _free_factor(self, free):
        nav = self.nav
        f = (free - nav["stop_obstacle_m"]) / max(nav["want_free_m"] - nav["stop_obstacle_m"], 1e-6)
        return np.where(np.isnan(free), 0.0, np.clip(f, 0.0, 1.0))

    # ------------------------------------------------------------------ commands
    def start(self, k, now, heading):
        self.target, self.legs, self.blocked = int(k), 0, {}
        self.explored, self.exploring, self._last_leg_heading = 0, False, None
        self.detour_side, self._explore_heading = 0, None
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
        self.state, self.scan, self._t0 = "look", [], now
        self._world_free = np.full(self.nb, np.nan)
        self.avoiding = 0
        self._last_free = (None, 0.0)

    def _store_free(self, heading, free):
        """Body-frame profile -> absolute directions, keeping the forward +-2 bins (the LiDAR's best view)."""
        cb = self.bin_of(heading)
        for j in (-2, -1, 0, 1, 2):
            self._world_free[(cb + j) % self.nb] = free[j % self.nb]

    def _finish_look(self, now, heading):
        nav, k, nb = self.nav, self.target, self.nb
        probs = np.array([p for _, p in self.scan])
        b = np.array([self.bin_of(h) for h, _ in self.scan])
        votes = np.bincount(probs.argmax(1), minlength=len(self.classes)) / len(probs)
        self.where, self.share = int(votes.argmax()), float(votes[k])
        sums, cnt = np.bincount(b, probs[:, k], nb), np.bincount(b, minlength=nb)
        mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        filled = np.where(np.isnan(mean), np.nanmean(mean), mean)
        p = (np.roll(filled, 1) + filled + np.roll(filled, -1)) / 3            # circular, +-1 bin
        self.bins = p
        tb = int(np.argmax(p))                                                  # where the target looks likeliest
        if self.share >= nav["arrive_share"]:
            self.state = "arrived"
            self.msg = f"ARRIVED at {self.classes[k]} ({self.share:.0%} of the views)"
            return
        if self.legs >= nav["max_legs"]:
            self.state = "failed"
            self.msg = f"gave up after {self.legs} legs (here looks like {self.classes[self.where]})"
            return
        self.exploring = False
        if p[tb] < nav["seen_conf"]:
            if not (self.have_lidar and self.explored < nav["explore_legs"]):
                self.state = "failed"
                self.msg = f"{self.classes[k]} not seen in any direction (best {p[tb]:.2f})"
                return
            # explore: walk into the most open direction, not back the way it came, then look again
            self.free = self._world_free.copy()
            fac = self._free_factor(self.free)
            back = np.ones(nb)
            if self._last_leg_heading is not None:
                bb = self.bin_of(self._last_leg_heading + math.pi)
                for j in range(-3, 4):
                    back[(bb + j) % nb] = 0.3
            keep = np.ones(nb)                                                 # momentum: keep exploring the same way
            if self._explore_heading is not None:
                eb = self.bin_of(self._explore_heading)
                for j in range(-3, 4):
                    keep[(eb + j) % nb] = 1.6
            fz = np.nan_to_num(self.free, nan=0.0)
            edge = np.ones(nb)                                                 # obstacle edges lead round obstacles
            jump = np.maximum(np.abs(fz - np.roll(fz, 1)), np.abs(fz - np.roll(fz, -1)))
            edge[(jump >= nav["edge_jump_m"]) & (fz >= nav["want_free_m"])] = 1.8
            score = fz * (fac > 0) * back * keep * edge
            best = int(np.argmax(score))
            if score[best] <= 0:
                self.state = "failed"
                self.msg = f"{self.classes[k]} not seen and no free direction to explore"
                return
            self.exploring, self.explored, self.detour, self.score = True, self.explored + 1, False, score
            self.best_heading = self._explore_heading = self.heading_of(best)
            self.state = "turn"
            self.msg = (f"{self.classes[k]} not in view (best {p[tb]:.2f}); exploring the most open way "
                        f"({self.free[best]:.1f} m free, try {self.explored}/{nav['explore_legs']})")
            return
        if self.have_lidar:
            self.free = self._world_free.copy()
            fac = self._free_factor(self.free)
            pen = np.ones(nb)
            for bb, leg in self.blocked.items():                               # avoid recently blocked ways
                if self.legs - leg < nav["blocked_memory_legs"]:
                    for j in (-1, 0, 1):
                        pen[(bb + j) % nb] *= 0.2
            self.score = p * fac * pen
            best = int(np.argmax(self.score))
            if self.score[best] < 0.25 * nav["seen_conf"]:                     # target's way is blocked: detour
                ok = np.flatnonzero((fac >= 0.6) & (pen >= 1.0))
                off = (ok - tb + nb / 2) % nb - nb / 2                          # signed bins from the target's way
                if self.detour_side:                                            # keep going round the same side
                    same = off * self.detour_side > 0
                    if same.any():
                        ok, off = ok[same], off[same]
                if not len(ok):
                    self.state = "failed"
                    self.msg = "boxed in: no free direction (LiDAR)"
                    return
                i = int(np.argmin(np.abs(off)))
                best = int(ok[i])
                self.detour_side = 1 if off[i] > 0 else -1
            else:
                self.detour_side = 0
        else:
            self.free, self.score, best = None, p, tb
        self.detour = abs((best - tb + nb / 2) % nb - nb / 2) > 1
        self.best_heading = self.heading_of(best)
        way = f"detour {math.degrees(wrap(self.best_heading - self.heading_of(tb))):+.0f} deg off the target's way" \
            if self.detour else "clear"
        free_txt = f", free {self.free[best]:.1f} m" if self.free is not None else ""
        self.state = "turn"
        self.msg = (f"here looks like {self.classes[self.where]}; heading {math.degrees(wrap(self.best_heading)):+.0f} "
                    f"deg towards {self.classes[k]} ({way}{free_txt})")

    def update(self, now, heading, prob=None, free=None):
        """One control step. prob: latest classifier output (None if no new frame). free: latest LiDAR free-space
        profile in the body frame (free_profile; None if no new scan; all zeros = no LiDAR data). Returns (vx, wz)."""
        nav, k = self.nav, self.target
        if self.state == "look":
            if prob is not None:
                self.scan.append((heading, np.asarray(prob, float)))
            if free is not None and self.have_lidar:
                self._store_free(heading, np.asarray(free, float))
            turned = sum(abs(wrap(b[0] - a[0])) for a, b in zip(self.scan, self.scan[1:]))
            if turned >= 2 * math.pi - 0.05 and len(self.scan) >= 8:
                self._finish_look(now, heading)
                return 0.0, 0.0
            return 0.0, nav["scan_rate"]
        if self.state == "turn":
            err = wrap(self.best_heading - heading)
            if abs(err) < 0.08:
                self.state, self.legs, self._t0, self._recent = "walk", self.legs + 1, now, deque()
                self._last_leg_heading = self.best_heading
                kind = ", exploring" if self.exploring else ", detour" if self.detour else ""
                self.msg = f"walking towards {self.classes[k]} (leg {self.legs}{kind})"
                return 0.0, 0.0
            return 0.0, float(np.clip(1.2 * err, -nav["scan_rate"], nav["scan_rate"]))
        if self.state == "walk":
            if prob is not None:
                self._recent.append((now, float(prob[k]), int(np.argmax(prob)) == k))
            while self._recent and now - self._recent[0][0] > max(nav["early_stop_s"], nav["lost_s"]):
                self._recent.popleft()
            end, vx = None, nav["approach_speed"]
            wz = float(np.clip(0.8 * wrap(self.best_heading - heading), -0.3, 0.3))
            self.avoiding = 0
            if free is not None:
                self._last_free = (free, now)
            elif self._last_free[0] is not None and now - self._last_free[1] <= 0.5:
                free = self._last_free[0]                                     # hold the last scan between scans
            if self.have_lidar and free is not None:
                free = np.asarray(free, float)
                if not np.any(free > 0):
                    end = "no LiDAR data"
                else:
                    front = float(free[0])
                    if front <= nav["stop_obstacle_m"]:
                        self.blocked[self.bin_of(heading)] = self.legs
                        end = f"blocked {front:.2f} m ahead"
                    elif front <= nav["slow_obstacle_m"]:                       # steer round it, slowly
                        left, right = min(free[1], free[2]), min(free[-1], free[-2])
                        self.avoiding = 1 if left >= right else -1
                        wz = self.avoiding * nav["avoid_turn_rate"]
                        vx = nav["approach_speed"] * 0.4
            span = now - self._recent[0][0] if self._recent else 0.0
            if end is None and not self.detour and span >= nav["early_stop_s"] * 0.9:
                w = [r for r in self._recent if now - r[0] <= nav["early_stop_s"]]
                if all(r[2] for r in w) and np.mean([r[1] for r in w]) >= nav["early_stop_conf"]:
                    end = f"{self.classes[k]} recognised steadily"
            if end is None and not (self.detour or self.exploring) and span >= nav["lost_s"] * 0.9:
                w = [r for r in self._recent if now - r[0] <= nav["lost_s"]]
                if max(r[1] for r in w) < nav["lost_conf"]:
                    end = f"lost sight of {self.classes[k]}"
            if end is None and now - self._t0 >= nav["leg_s"]:
                end = "leg done"
            if end is not None:
                self.msg = f"{end}; looking around"
                self._look(now, heading)
                return 0.0, 0.0
            return vx, wz
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
    if nav.avoiding:
        put_text(p, f"avoiding obstacle: steering {'left' if nav.avoiding > 0 else 'right'}", (12, 206), 0.45,
                 (0, 200, 255))
    if nav.bins is not None and nav.target is not None:      # last look-around, per absolute direction
        cx, cy, r = PANEL_W // 2, 305, 70
        cv2.circle(p, (cx, cy), r, (70, 70, 70), 1)
        nb = len(nav.bins)
        look = nav.nav.get("lookahead_m", 2.5)
        for i in range(nb):
            a = i / nb * 2 * math.pi                         # up = heading 0, counter-clockwise
            ux, uy = -math.sin(a), -math.cos(a)
            if nav.free is not None and not np.isnan(nav.free[i]):   # grey: free distance
                rf = int(r * min(nav.free[i] / look, 1.0))
                cv2.line(p, (cx, cy), (int(cx + rf * ux), int(cy + rf * uy)), (95, 95, 95), 7)
            rr = int(r * float(np.clip(nav.bins[i], 0, 1)))  # colour: target likelihood
            cv2.line(p, (cx, cy), (int(cx + rr * ux), int(cy + rr * uy)), COLORS[nav.target % len(COLORS)], 3)
        if nav.best_heading is not None:                     # white: chosen direction
            a = nav.best_heading
            cv2.line(p, (cx, cy), (int(cx - (r + 8) * math.sin(a)), int(cy - (r + 8) * math.cos(a))),
                     (255, 255, 255), 1)
        put_text(p, "look-around: target (colour), free space (grey), chosen (white)", (12, 226), 0.36,
                 (170, 170, 170))
        if nav.where is not None:
            put_text(p, f"here looked like {classes[nav.where]}; target share {nav.share:.0%}", (12, 392), 0.42)
    put_text(p, f"{fps:4.1f} fps   1-{len(classes)} go  C cancel  V video", (12, VIEW_H - 30), 0.42, (170, 170, 170))
    put_text(p, "REC" if recording else "not recording (V)", (12, VIEW_H - 10), 0.45,
             (60, 60, 235) if recording else (150, 150, 150))
    return p


def lidar_panel(free, nav_cfg, fps):
    """--lidar-check: the live LiDAR free distance per direction around the dog (up = straight ahead) and what
    the navigator would do with it. No mission and no autonomous motion."""
    p = np.full((VIEW_H, PANEL_W, 3), 24, np.uint8)
    put_text(p, "LiDAR check - no autonomy", (12, 26), 0.6, (255, 255, 255))
    put_text(p, "free distance in Test 1's body corridor", (12, 46), 0.42, (170, 170, 170))
    if free is None:
        put_text(p, "waiting for the LiDAR ...", (12, 90), 0.55)
        return p
    look = nav_cfg["lookahead_m"]
    ahead = float(free[0])
    stop, slow = nav_cfg["stop_obstacle_m"], nav_cfg["slow_obstacle_m"]
    if not free.any():
        verdict, col = "NO LiDAR DATA (would not walk)", (60, 60, 235)
    elif ahead <= stop:
        verdict, col = f"would STOP (< {stop:.2f} m)", (60, 60, 235)
    elif ahead <= slow:
        verdict, col = f"would SLOW + steer (< {slow:.2f} m)", (0, 200, 255)
    else:
        verdict, col = "clear ahead", (90, 230, 90)
    put_text(p, f"ahead: {ahead:.2f} m", (12, 82), 0.7, (255, 255, 255), 2)
    put_text(p, verdict, (12, 108), 0.5, col)
    put_text(p, f"left {free[len(free) // 4]:.2f} m   right {free[-len(free) // 4]:.2f} m", (12, 132), 0.48)
    cx, cy, r = PANEL_W // 2, 270, 110
    for rad, c in ((stop, (60, 60, 235)), (slow, (0, 200, 255)), (look, (90, 90, 90))):
        cv2.circle(p, (cx, cy), int(r * min(rad / look, 1.0)), c, 1)
    nb = len(free)
    for i in range(nb):
        a = i / nb * 2 * math.pi
        rf = int(r * min(free[i] / look, 1.0))
        cv2.line(p, (cx, cy), (int(cx - rf * math.sin(a)), int(cy - rf * math.cos(a))),
                 (170, 170, 170) if i else (255, 255, 255), 3 if i else 5)
    put_text(p, "rings: stop (red), slow (orange), lookahead", (12, VIEW_H - 48), 0.4, (170, 170, 170))
    put_text(p, f"{fps:4.1f} fps   drive with teleop keys, ESC quit", (12, VIEW_H - 28), 0.42, (170, 170, 170))
    return p


class LidarProfile:
    """Free-space profile from the real LiDAR (Ethernet): Test 1's LidarSource + cloud_to_body conversion,
    then free_profile with Test 1's walking corridor. Returns a new profile per new cloud (None otherwise);
    all zeros when the cloud is missing or older than max_age_s (= blocked: no data, no walking)."""

    def __init__(self, link, go2cfg, nav, max_age_s=1.0):
        import lidar as lidar_mod
        bcfg = yaml.safe_load((Path(lidar_mod.__file__).parent / "config/behaviors.yaml").read_text())
        self.corridor, self.nav, self.max_age_s = bcfg["safety"]["corridor"], nav, max_age_s
        self.lidar_mod, self.state = lidar_mod, link.state
        self.src = lidar_mod.LidarSource(go2cfg["lidar"])
        self._seq = -1

    def __call__(self):
        from perception.pipeline import cloud_to_body
        msg, seq, stamp = self.src.latest.get()
        nb = int(self.nav["heading_bins"])
        if msg is None or time.monotonic() - stamp > self.max_age_s:
            return np.zeros(nb)
        if seq == self._seq:
            return None
        self._seq = seq
        sport = self.state.sport.get()[0] if self.state is not None else None
        pts, _ = cloud_to_body(msg, sport, "auto", self.lidar_mod)
        if pts is None:
            return np.zeros(nb)
        return free_profile(pts, nb, self.corridor, self.nav["lookahead_m"])


def make_safety(link, go2cfg, nav, allow_no_lidar):
    """(free-space profile fn or None, pose fn or None). Ethernet: real LiDAR profile + odometry heading."""
    if link.name == "sdk":
        from behavior.clearance import make_real_pose_fn
        return LidarProfile(link, go2cfg, nav), make_real_pose_fn(link.state)
    if not allow_no_lidar:
        raise SystemExit("Autonomous walking needs the LiDAR (Ethernet, --source sdk) for obstacle avoidance. Over "
                         "Wi-Fi, run with --allow-no-lidar only in a clear area with someone ready to press SPACE.")
    log.warning("No LiDAR: obstacle avoidance OFF, speed capped at 0.15 m/s, 2 s legs. Supervise closely.")
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--model", help="checkpoint (default: <results_dir>/resnet18/best.pt)")
    ap.add_argument("--target", help="zone to go to at start (name or 1-based number)")
    ap.add_argument("--allow-no-lidar", action="store_true", help="Wi-Fi / fake: walk without the obstacle check")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--lidar-check", action="store_true",
                    help="no autonomy: show the live LiDAR free space and whether the navigator would stop/slow "
                         "(tune Test 1's safety.corridor with this before the first mission)")
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
    profile, pose_fn = make_safety(link, go2cfg, cfg["navigation"], a.allow_no_lidar)
    nav = ZoneNavigator(classes, cfg["navigation"], have_lidar=profile is not None)
    heading = Heading(pose_fn)
    tele.on_manual_input = lambda key: nav.cancel(f"cancelled by key '{key}'")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out / f"goto_{stamp}.mp4"
    events, missions = [], []          # missions: one entry per zone request (target, result, reason, legs, time)
    writer, recording = None, a.headless
    prob, wz, last_seq, last_state = np.ones(len(classes)) / len(classes), 0.0, -1, "idle"
    t0 = time.monotonic()
    next_video, last_free, last_print = t0, None, 0.0
    if a.lidar_check and profile is None:
        tele.shutdown()
        link.close()
        raise SystemExit("--lidar-check needs the LiDAR: use --source sdk --iface <name> (Ethernet).")

    def begin(k, now, h):
        if missions and missions[-1]["result"] is None:          # a new request replaces an unfinished one
            missions[-1].update(result="replaced", reason="new zone requested", legs=nav.legs,
                                seconds=round(now - t0 - missions[-1]["t"], 1))
        nav.start(k, now, h)
        missions.append({"target": classes[k], "t": round(now - t0, 2), "result": None})

    if a.target and not a.lidar_check:
        k = classes.index(a.target) if a.target in classes else int(a.target) - 1
        begin(k, t0, heading(t0, 0.0))
    try:
        while True:
            now = time.monotonic()
            h = heading(now, wz)
            img, seq = link.frame()
            new = None
            if img is not None and seq != last_seq:
                last_seq = seq
                prob = new = clf(img)
            if a.lidar_check:
                f = profile() if profile is not None else None
                if f is not None:
                    last_free = f
                    if now - last_print >= 1.0:
                        log.info("LiDAR free ahead %.2f m, left %.2f m, right %.2f m", f[0], f[len(f) // 4],
                                 f[-len(f) // 4])
                        last_print = now
                view = np.zeros((VIEW_H, VIEW_W, 3), np.uint8) if img is None else cv2.resize(img, (VIEW_W, VIEW_H))
                comp = np.hstack([view, lidar_panel(last_free, cfg["navigation"], link.fps())])
                if a.seconds and now - t0 >= a.seconds:
                    break
                if a.headless:
                    time.sleep(0.02)
                    continue
                cv2.imshow("Go2 LiDAR check", comp)
                if (cv2.waitKey(1) & 0xFF) == 27 or tele.quit_event.is_set():
                    break
                continue
            free = profile() if (profile is not None and nav.active) else None
            vx, wz = nav.update(now, h, new, free)
            if nav.active:
                tele.set_autonomy(vx, 0.0, wz)
            else:
                tele.clear_autonomy()
                wz = 0.0
            if nav.state != last_state:
                events.append({"t": round(now - t0, 2), "state": nav.state, "msg": nav.msg})
                log.info("%s: %s", nav.state, nav.msg)
                if nav.state in ("arrived", "failed", "idle") and missions and missions[-1]["result"] is None:
                    missions[-1].update(result="cancelled" if nav.state == "idle" else nav.state, reason=nav.msg, legs=nav.legs,
                                        explored=getattr(nav, "explored", 0),
                                        seconds=round(now - t0 - missions[-1]["t"], 1))
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
                begin(key - ord("1"), now, h)
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
        if missions and missions[-1]["result"] is None:
            missions[-1].update(result="unfinished", reason="program ended", legs=nav.legs)
        (out / f"goto_{stamp}.json").write_text(json.dumps(
            {"classes": classes, "target": classes[nav.target] if nav.target is not None else None,
             "final_state": nav.state, "legs": nav.legs, "source": link.name, "lidar": profile is not None,
             "video": mp4.name if writer is not None else None, "missions": missions, "events": events},
            indent=1))
        print(f"{nav.state}: {nav.msg}")


if __name__ == "__main__":
    main()
