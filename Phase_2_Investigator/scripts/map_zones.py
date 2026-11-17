#!/usr/bin/env python3
"""Build a zone map: the dog explores on its own and tags its LiDAR map with zones (lab_1, lab_2, ...).

    # simulation (house 185, the Test 2 Q1 classifier), scored against the true zones:
    python scripts/map_zones.py --random 1854 --video
    # the real Go2 over Ethernet (the lab classifier), terminal only:
    python scripts/map_zones.py --backend real --max-time 600

Output (outputs/zone_map/<name>/): map.npz (occupancy + zone evidence), zones.json (each zone's area,
centre, goal point, confidence), zone_map.png, report.json, and with --video a map-building MP4.
`scripts/goto_zone.py --map outputs/zone_map/<name>` then drives to a zone on this map.

Real robot: the dog moves on its own. Stand by with the remote; SPACE / any driving key in the terminal
stops it and ends the run (see real_robot/README.md).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (offscreen GL backend before mujoco loads)


class VideoHook:
    """Camera | live zone map, one frame every `every` seconds of robot time."""

    def __init__(self, path, mapper, every=1.0, fps=10):
        self.mapper, self.every, self.next = mapper, every, -1.0
        self.path, self.fps, self.w = str(path), fps, None

    def __call__(self, nav):
        import cv2
        import numpy as np
        t = nav.sim.time
        if t < self.next:
            return
        self.next = t + self.every
        st = nav.sim.state()
        path = [(x, y) for _, x, y, _ in nav.log.path[::5]]
        m = self.mapper.zone_map.image(path, robot=(st.position[0], st.position[1], st.yaw), scale=3)
        h = 360
        m = cv2.resize(m, (int(m.shape[1] * h / m.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        cam = cv2.resize(nav.sim.camera().rgb, (480, 360))
        p = self.mapper.last_prob
        if p is not None:
            k = int(np.argmax(p))
            cv2.rectangle(cam, (0, 0), (480, 24), (0, 0, 0), -1)
            cv2.putText(cam, f"camera sees: {self.mapper.zone_map.classes[k]} ({p[k]:.2f})", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.hstack([cam, m])
        cv2.rectangle(frame, (0, h - 22), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(frame, f"t={t:5.0f}s  {self.mapper.status()}"[:150], (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.cvtColor(np.ascontiguousarray(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]),
                             cv2.COLOR_RGB2BGR)
        if self.w is None:
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                     (frame.shape[1], frame.shape[0]))
        self.w.write(frame)

    def close(self):
        if self.w is not None:
            self.w.release()


def main():
    from sim_env.launch import add_sim_args
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--backend", choices=["sim", "real"], default="sim")
    ap.add_argument("--model", help="zone classifier checkpoint (default: sim or real Test 2 model)")
    ap.add_argument("--max-time", type=float, default=900.0, help="seconds of robot time")
    ap.add_argument("--no-fill-in", action="store_true", help="skip the viewpoint coverage phase")
    ap.add_argument("--spacing", type=float, default=1.0, help="metres between look-around viewpoints")
    ap.add_argument("--out", help="output directory (default outputs/zone_map/<name>)")
    ap.add_argument("--video", action="store_true", help="save map_building.mp4")
    ap.add_argument("--house", default="scenesmith_house_185", help="sim: house for --random scenes")
    ap.add_argument("--zones-gt", default=str(ROOT / "mapping/config/house185_zones.yaml"),
                    help="sim only: true zone polygons used to score the map")
    from real_robot.session import add_real_args
    add_real_args(ap)
    a = ap.parse_args()

    from mapping.mapper import ZoneMapper
    from navigation.explorer import Navigator
    from perception.zone_classifier import REAL_MODEL, SIM_MODEL, ZoneClassifier

    if a.backend == "sim":
        from sim_env.launch import session_from_args
        if not (a.scenario or a.random is not None):
            a.random = 1854                     # the Test 2 Q1 test scene (never seen in training)
        if a.random is not None and not a.env:
            # random scenes in house 185, as Test 2 made them (the zone classifier and zones are house 185's)
            from sim_env.scenarios.random_sim import write_random
            a.scenario, a.random = str(write_random(a.random, env=a.house)), None
        sim, _ = session_from_args(a)
        name = getattr(sim, "scenario_id", sim.environment_id)
        model = a.model or SIM_MODEL
    else:
        from real_robot.session import RealGo2Session
        sim = RealGo2Session.from_args(a)
        name = "real_" + time.strftime("%Y%m%d_%H%M%S")
        model = a.model or REAL_MODEL
    out = Path(a.out or ROOT / "outputs" / "zone_map" / name)
    out.mkdir(parents=True, exist_ok=True)
    clf = ZoneClassifier(model)
    print(f"zone classifier {Path(model).name}: {clf.classes}")
    nav = Navigator(sim, verbose=True)
    mapper = ZoneMapper(nav, clf)
    video = None
    if a.video:
        video = VideoHook(out / "map_building.mp4", mapper)
        nav.on_tick.append(video)
    try:
        rep = mapper.run(max_time=a.max_time, fill_in=not a.no_fill_in, spacing=a.spacing)
    except KeyboardInterrupt:
        print("stopped by the operator: saving the map so far")
        rep = nav.report()
        rep["stopped_by_operator"] = True
    finally:
        nav.vel.stop()
        if video:
            video.close()
    path = [(x, y) for _, x, y, _ in nav.log.path]
    rep["backend"], rep["model"] = a.backend, str(model)
    if a.backend == "sim" and getattr(sim, "environment_id", "") and Path(a.zones_gt).exists():
        from mapping.evaluate import load_polygons, score
        if load_polygons(a.zones_gt)[2] == sim.environment_id:
            rep["score"] = score(mapper.zone_map, a.zones_gt)
    mapper.zone_map.save(out, path, extra={"backend": a.backend, "model": str(model),
                                           "scenario": getattr(a, "scenario", None), "link": getattr(a, "link", None)})
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("path_length_m", "n_collisions", "frames_tagged", "look_arounds", "regions")
                      if k in rep}, indent=1))
    if "score" in rep:
        s = rep["score"]
        print(f"zone map vs truth: cell accuracy {s['cell_accuracy']}, goals in the right zone {s['goals_correct']}")
    print(f"-> {out}")
    if hasattr(sim, "close"):
        sim.close()


if __name__ == "__main__":
    main()
