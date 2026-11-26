#!/usr/bin/env python3
"""Crime-scene search (C2-C4): explore, find stains with the camera, map them, never walk over them, inspect.

    python scripts/investigate.py --scenario 1                  # simulation, scored against ground truth
    python scripts/investigate.py --scenario 1 --no-keepout     # the same without keeping off floor evidence
    python scripts/investigate.py --scenario 3 --video

What the dog does:
1. explores (LiDAR frontiers) and covers the floor with look-arounds about 1 m apart (mapping/mapper.py);
   in house 185 it also tags the zones (Test 2's simulation classifier);
2. every second, the front camera (1280 x 720, 100 deg, like the real Go2 stream) goes through the stain
   detector (perception/stain_detector.py); detections become evidence clusters on the map
   (mapping/evidence_map.py) with surface kind, zone, views and probability;
3. confirmed floor evidence becomes a keep-out layer in the planner, so later paths go around it;
4. the confirmed clusters (most certain first) get a close-up tour (navigation/closeups.py): aim points
   across each cluster, the camera within ~0.5-1 m, 1920 x 1080 close-ups (detected positions, not ground truth);
5. every stain in the close-ups is measured (analysis/closeup.py: ellipse, impact angle, direction) and the
   scene is analysed (analysis/scene_report.py): pattern groups, mechanism, weapon categories it is consistent
   with, area of origin. Writes scene_report.md and analysis.json (scored against ground truth in simulation).
   --no-analyze uses the old 2-view inspection instead.
Outputs in outputs/investigate/<scenario>[_nokeepout]/: report.json (ground-truth score, stains stepped on,
collisions), evidence.json, evidence_map.png, close-ups, and with --video investigation.mp4.
Simulation only for now: on the real Go2 the detector needs region depth from the floor plane / LiDAR.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402

W, H, HFOV = 1280, 720, 100.0
FOVY = float(np.degrees(2 * np.arctan(np.tan(np.radians(HFOV / 2)) * H / W)))


class EvidenceHook:
    """Every `every` s of robot time: render the front camera at 720p, detect stains, update the evidence map."""

    def __init__(self, sim, detector, evidence, zone_map=None, keepout=True, every=1.0, video=None, camera_height=None,
                 mast_pitch_deg=25.0):
        self.sim, self.det, self.ev, self.zm = sim, detector, evidence, zone_map
        self.camera_height, self.mast_pitch = camera_height, np.radians(mast_pitch_deg)
        self.keepout, self.every, self.next = keepout, every, -1.0
        self.video, self.last = video, None
        self.n_det = 0

    def zone_of(self, xy):
        if self.zm is None:
            return None
        k, _ = self.zm.zone_at(xy)
        return self.zm.classes[k] if k >= 0 else None

    def __call__(self, nav):
        t = self.sim.time
        if t < self.next:
            return
        self.next = t + self.every
        T = self.sim.camera_pose()
        pos, fwd = T[:3, 3].copy(), T[:3, 2].copy()
        if self.camera_height is not None:                 # emulated mast camera: higher, tilted down
            pos[2] = self.camera_height
            yaw = np.arctan2(fwd[1], fwd[0])
            fwd = np.array([np.cos(self.mast_pitch) * np.cos(yaw), np.cos(self.mast_pitch) * np.sin(yaw),
                            -np.sin(self.mast_pitch)])
        frame = self.sim.render_camera(pos, fwd, fovy=FOVY, width=W, height=H, depth=True, frame_id="front_720p")
        regs = self.det.regions(frame)
        hits = [r for r in regs if r.prob >= self.det.threshold]
        self.n_det += len(hits)
        self.ev.add_frame(hits, t, self.zone_of if self.zm is not None else None)
        if self.keepout:
            nav.grid.set_keepout(self.ev.floor_points())
        self.last = (frame, regs)
        if self.video is not None:
            self.video(nav, frame, regs)


def evidence_image(nav, zm, ev, gt=None, scale=4):
    """Zone map (or occupancy) with the path, ground-truth clusters (hollow) and evidence (filled)."""
    g = nav.grid
    path = [(x, y) for _, x, y, _ in nav.log.path]
    st = nav.sim.state()
    img = zm.image(path, robot=(st.position[0], st.position[1], st.yaw), scale=scale)
    ko = getattr(g, "keepout", None)
    if ko is not None and ko.any():
        m = cv2.resize(np.ascontiguousarray(ko[::-1]).astype(np.uint8), (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = (0.5 * img[m] + 0.5 * np.array([255, 170, 170])).astype(np.uint8)

    def px(xy):
        return (int((xy[0] - g.origin[0]) / g.res * scale), int((g.shape[0] - (xy[1] - g.origin[1]) / g.res) * scale))
    if gt is not None:
        for c in gt.get("clusters", []):
            if c["n_stains"] >= 3:
                cv2.circle(img, px(c["center"]), max(6, int(c["radius"] / g.res * scale)), (90, 90, 90), 1, cv2.LINE_AA)
    for c in ev.summary():
        col = (200, 0, 0) if c["confirmed"] else (240, 150, 150)
        p = px(c["xyz"])
        cv2.circle(img, p, 6 if c["confirmed"] else 4, col, -1, cv2.LINE_AA)
        if c["confirmed"]:
            cv2.putText(img, f"{c['id']} {c['surface'][0]}", (p[0] + 7, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (120, 0, 0), 1, cv2.LINE_AA)
    return img


class Video:
    def __init__(self, path, ev_ref, every=1.0, fps=8):
        self.path, self.every, self.fps, self.w, self.next = str(path), every, fps, None, -1.0
        self.ev_ref = ev_ref

    def __call__(self, nav, frame, regs):
        from perception.stain_detector import draw
        cam = cv2.resize(draw(frame.rgb, regs), (640, 360))
        m = evidence_image(nav, *self.ev_ref, scale=3)
        m = cv2.resize(m, (int(m.shape[1] * 360 / m.shape[0]), 360), interpolation=cv2.INTER_NEAREST)
        out = np.hstack([cam, m])
        out = cv2.cvtColor(np.ascontiguousarray(out[:, : out.shape[1] // 2 * 2]), cv2.COLOR_RGB2BGR)
        ev = self.ev_ref[1]
        txt = f"t={nav.sim.time:4.0f}s  evidence clusters {len(ev.clusters)}, confirmed {len(ev.confirmed())}"
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.w is None:
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (out.shape[1], out.shape[0]))
        self.w.write(out)

    def close(self):
        if self.w is not None:
            self.w.release()


def main():
    from sim_env.launch import add_sim_args, session_from_args
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-keepout", action="store_true", help="do not keep off floor evidence (for comparison)")
    ap.add_argument("--no-zones", action="store_true", help="skip zone tagging")
    ap.add_argument("--detector", help="stain detector model (default perception/models/stain_rf_depth.pkl)")
    ap.add_argument("--max-time", type=float, default=900.0, help="exploration + coverage budget, s of robot time")
    ap.add_argument("--inspect", type=int, default=6, help="close-ups of at most this many confirmed clusters")
    ap.add_argument("--no-analyze", action="store_true",
                    help="old inspection (2 standoff views per cluster) instead of the close-up tour + stain analysis")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--camera-height", type=float,
                    help="simulation only: emulate a camera on a mast at this height (m) for detection and close-ups "
                         "(the Go2's own camera is at ~0.41 m and cannot see onto surfaces above it, e.g. a bed top)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not (a.scenario or a.random is not None):
        a.scenario = "1"
    from mapping.evidence_map import EvidenceMap, floor_stains_stepped_on, score, stepped_after_found
    from mapping.mapper import ZoneMapper
    from navigation.explorer import Navigator
    from analysis.closeup import merge_observations, score_against_truth, stains_in_frame
    from analysis.scene_report import analyse
    from navigation.closeups import closeup_tour
    from navigation.inspect import inspect_point
    from perception.stain_detector import StainDetector

    sim, gt = session_from_args(a)
    name = getattr(sim, "scenario_id", sim.environment_id) + ("_nokeepout" if a.no_keepout else "") + \
        (f"_mast{a.camera_height:.1f}" if a.camera_height else "")
    out = Path(a.out or ROOT / "outputs" / "investigate" / name)
    out.mkdir(parents=True, exist_ok=True)
    clf = None
    if not a.no_zones and sim.environment_id == "scenesmith_house_185":
        from perception.zone_classifier import SIM_MODEL, ZoneClassifier
        if SIM_MODEL.exists():
            clf = ZoneClassifier(SIM_MODEL)
    nav = Navigator(sim, verbose=True)
    mapper = ZoneMapper(nav, clf)
    det = StainDetector(a.detector, use_depth=True)
    ev = EvidenceMap()
    video = Video(out / "investigation.mp4", (mapper.zone_map, ev, gt)) if a.video else None
    hook = EvidenceHook(sim, det, ev, mapper.zone_map if clf is not None else None, keepout=not a.no_keepout,
                        camera_height=a.camera_height,
                        video=video)
    nav.on_tick.append(hook)
    t0, wall0 = sim.time, time.time()
    observations = []
    try:
        rep = mapper.run(max_time=a.max_time)
        targets = sorted(ev.confirmed(), key=lambda c: -c["views"] * np.mean(c["probs"]))[: a.inspect]
        inspected = []
        if a.no_analyze:
            for c in targets:
                radius = float(np.clip(np.sqrt(max(c["area_m2"], 1e-4) / np.pi) * 2, 0.15, 0.6))
                res = inspect_point(nav, c["xyz"], c["normal"], name=c["id"], out_dir=out / "closeups", n_views=2,
                                    radius=radius)
                inspected.append({"id": c["id"], "views_captured": len(res["views"])})
        else:
            # C5: close-ups across every detected cluster, then measure each stain (no ground truth used)
            for c in targets:
                frames = closeup_tour(nav, c, out_dir=str(out / "closeups"), camera_height=a.camera_height)
                for _, fr in frames:
                    observations.extend(stains_in_frame(fr, frame_id=fr.frame_id))
                inspected.append({"id": c["id"], "closeups": len(frames)})
        rep["inspected"] = inspected
    finally:
        nav.vel.stop()
        if video:
            video.close()
    rep.update({"scenario": name, "keepout": not a.no_keepout, "detector": Path(a.detector or "stain_rf_depth.pkl").name, "detector_threshold": det.threshold, "detections": hook.n_det,
                "evidence_frames": ev.frames, "sim_time_total_s": round(sim.time - t0, 1),
                "wall_time_total_s": round(time.time() - wall0, 1)})
    rep["score"] = score(ev, gt)
    stepped = floor_stains_stepped_on(gt, nav.log.path, times=True)
    n_floor = sum(1 for s in gt["stains"] if s["center_world"][2] < 0.03)
    rep["floor_stains_stepped_on"] = len(stepped)
    rep["floor_stains_stepped_on_after_found"] = len(stepped_after_found(ev, stepped))
    rep["floor_stains_total"] = n_floor
    rep.pop("regions", None)
    if observations:
        stains = merge_observations(observations)
        import pickle
        (out / "stains.pkl").write_bytes(pickle.dumps(stains))      # scripts/reanalyse.py re-runs the analysis
        zone_of = (lambda xy: hook.zone_of(np.asarray(xy)[:2])) if clf is not None else None
        ana = analyse(stains, zone_of=zone_of, gt=gt)
        ana["measurement"] = score_against_truth(stains, gt)
        (out / "scene_report.md").write_text(ana.pop("text"))
        (out / "analysis.json").write_text(json.dumps(ana, indent=1, default=float))
        rep["analysis"] = {k: ana.get(k) for k in ("stains_measured", "stains_usable", "mechanism_correct",
                                                   "origin_error_median_m")}
        rep["analysis"]["groups"] = [{k: g.get(k) for k in ("id", "n_stains", "surface", "zone", "mechanism",
                                                             "mechanism_confidence", "origin", "truth")}
                                     for g in ana["groups"]]
        rep["analysis"]["measurement"] = ana["measurement"]
    ev.save(out / "evidence.json")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    cv2.imwrite(str(out / "evidence_map.png"), cv2.cvtColor(evidence_image(nav, mapper.zone_map, ev, gt), cv2.COLOR_RGB2BGR))
    s = rep["score"]
    print(f"[investigate] {name}: {s['truth_found']}/{s['truth_clusters']} stain clusters found (precision "
          f"{s['precision']}, median error {s['median_error_to_centre_m']} m), distractors flagged "
          f"{s['distractors_flagged']}/{s['distractors']}; floor stains stepped on {len(stepped)}/{n_floor} "
          f"({rep['floor_stains_stepped_on_after_found']} after they were found); "
          f"collisions {rep['n_collisions']} -> {out}")
    sim.close()


if __name__ == "__main__":
    main()
