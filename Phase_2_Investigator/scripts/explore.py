#!/usr/bin/env python3
"""Autonomous exploration (A4): frontier exploration with the LiDAR, then a camera-coverage patrol.

    python scripts/explore.py --scenario 1                 # headless; report + map in outputs/explore/<id>/
    python scripts/explore.py --scenario 1 --viewer        # watch it live in the MuJoCo viewer (real time)
    python scripts/explore.py --scenario 3 --inspect       # then inspect the ground-truth stain clusters (A4)
    python scripts/explore.py --scenario 2 --record        # also record the run (A6): runs/<id>_<time>/

The robot drives only through cmd_vel (navigation/velocity.py) with the safety
layer active. Collisions are logged; the run fails (exit 1) if there are any.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (first: picks the offscreen GL backend before mujoco loads)

from navigation.explorer import Navigator  # noqa: E402
from navigation.plots import plot_map  # noqa: E402
from sim_env.launch import add_sim_args, session_from_args  # noqa: E402


class LiveViewer:
    """Passive MuJoCo viewer synced on every navigation tick, paced to real time."""

    def __init__(self, sim, speed=1.0):
        import mujoco
        import mujoco.viewer
        self.sim, self.speed = sim, speed
        self.v = mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False)
        with self.v.lock():
            self.v.opt.geomgroup[3] = 0
            self.v.opt.geomgroup[4] = 0
            self.v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.v.cam.trackbodyid = sim.base_body
            self.v.cam.distance, self.v.cam.elevation = 3.0, -45
        self.t0, self.s0 = time.monotonic(), sim.time

    def __call__(self, nav):
        if not self.v.is_running():
            raise KeyboardInterrupt
        self.v.sync()
        lag = (self.sim.time - self.s0) / self.speed - (time.monotonic() - self.t0)
        if lag > 0:
            time.sleep(lag)


class VideoHook:
    """Demo video: third-person view | front camera over the live map (path, coverage, stains)."""

    def __init__(self, sim, gt, path, every=0.5, fps=10):
        import imageio.v2 as imageio
        self.sim, self.gt, self.every = sim, gt, every
        self.next = 0.0
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.path = path

    def __call__(self, nav):
        if self.sim.time < self.next:
            return
        self.next = self.sim.time + self.every
        import cv2
        import numpy as np
        from navigation.plots import map_image
        chase = self.sim.render_view(640, 480, distance=2.4, elevation=-35)
        front = cv2.resize(self.sim.camera().rgb, (320, 240))
        g = nav.grid
        m = (map_image(g, nav.coverage) * 255).astype(np.uint8)[::-1]
        def px(x, y):
            return int((x - g.origin[0]) / g.res), int(g.shape[0] - 1 - (y - g.origin[1]) / g.res)
        if self.gt:
            for s in self.gt["stains"][::3]:
                cv2.circle(m, px(*s["center_world"][:2]), 1, (220, 0, 0), -1)
        pts = [px(x, y) for _, x, y, _ in nav.log.path]
        if len(pts) > 1:
            cv2.polylines(m, [np.array(pts, np.int32)], False, (30, 90, 230), 1)
        st = self.sim.state()
        p0 = px(*st.position[:2])
        cv2.circle(m, p0, 4, (30, 90, 230), -1)
        cv2.line(m, p0, (int(p0[0] + 10 * np.cos(st.yaw)), int(p0[1] - 10 * np.sin(st.yaw))), (30, 90, 230), 2)
        ys, xs = np.nonzero(g.seen)
        if len(xs):
            y0, y1 = g.shape[0] - 1 - ys.max(), g.shape[0] - 1 - ys.min()
            x0, x1 = xs.min(), xs.max()
            m = m[max(y0 - 10, 0):y1 + 10, max(x0 - 10, 0):x1 + 10]
        m = cv2.resize(m, (320, 240), interpolation=cv2.INTER_NEAREST)
        frame = np.hstack([chase, np.vstack([front, m])])
        txt = (f"t={self.sim.time:5.0f}s  path {sum(np.hypot(*np.diff(np.array([(x, y) for _, x, y, _ in nav.log.path]), axis=0).T)) if len(nav.log.path) > 1 else 0:.1f} m  "
               f"camera coverage {nav.coverage.coverage()}  collisions {len(nav.log.collisions)}")
        cv2.rectangle(frame, (0, 0), (960, 22), (0, 0, 0), -1)
        cv2.putText(frame, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.w.append_data(frame)

    def close(self):
        self.w.close()
        return self.path


def main():
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--max-time", type=float, default=300.0, help="sim seconds for each phase")
    ap.add_argument("--no-patrol", action="store_true", help="frontier exploration only")
    ap.add_argument("--inspect", action="store_true", help="afterwards, inspect the ground-truth stain clusters")
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer (real time)")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--out", help="output directory (default outputs/explore/<scenario>)")
    ap.add_argument("--video", action="store_true", help="save a demo video (5x speed) to the output directory")
    ap.add_argument("--record", nargs="?", const="auto", metavar="DIR",
                    help="record camera, depth, LiDAR, poses (docs/recording_format.md); default runs/<id>_<time>")
    a = ap.parse_args()
    if not (a.scenario or a.random is not None):
        a.scenario = "1"
    sim, gt = session_from_args(a)
    name = getattr(sim, "scenario_id", sim.environment_id)
    out = Path(a.out or ROOT / "outputs" / "explore" / name)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sim.settle(1.0)
        hooks = [LiveViewer(sim, a.speed)] if a.viewer else []
        vid = None
        if a.video:
            vid = VideoHook(sim, gt, out / "explore.mp4")
            hooks.append(vid)
        rec = None
        if a.record:
            from sim_env.recorder import RunRecorder
            rdir = ROOT / "runs" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}" if a.record == "auto" else Path(a.record)
            rec = RunRecorder(rdir, sim, gt)
            hooks.append(rec)
        nav = Navigator(sim, on_tick=hooks)
        t0, wall0 = sim.time, time.time()
        try:
            nav.explore(max_time=a.max_time)
            if not a.no_patrol:
                nav.patrol(max_time=a.max_time)
            if a.inspect and gt:
                from navigation.inspect import inspect_clusters
                inspect_clusters(nav, gt, out_dir=out / "evidence")
        except KeyboardInterrupt:
            print("\nstopped by user")
        rep = nav.report(t0, wall0)
        rep["scenario"] = name
        if getattr(nav, "inspected", None):
            rep["inspected"] = nav.inspected
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
        if rec is not None:
            print(f"recorded {rec.n_frames} frames, {rec.n_scans} scans -> {rec.close(rep)}")
        if vid is not None:
            print(f"video -> {vid.close()}")
        plot_map(nav, gt, title=f"{name}: {rep['sim_time_s']} s sim, {rep['path_length_m']} m, "
                                f"camera coverage {rep['camera_coverage']}, collisions {rep['n_collisions']}",
                 path=out / "map.png")
        print(json.dumps({k: rep[k] for k in ("sim_time_s", "wall_time_s", "path_length_m", "camera_coverage",
                                              "n_collisions", "safety_stops")}, default=float))
        print(f"report and map: {out}")
    finally:
        sim.close()
    sys.exit(1 if rep["n_collisions"] else 0)


if __name__ == "__main__":
    main()
