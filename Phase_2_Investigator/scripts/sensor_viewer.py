#!/usr/bin/env python3
"""Live view of every Go2 sensor, to prove each one works.

    python scripts/sensor_viewer.py                      # live window, drive with the keys
    python scripts/sensor_viewer.py --scenario 1
    python scripts/sensor_viewer.py --snapshot docs/media/a1_sensors.png --demo-drive
    python scripts/sensor_viewer.py --gif docs/media/a1_sensors.gif --demo-drive

Panels: front RGB camera; depth (metres); third-person view; 2D LiDAR scan in
the robot frame; 3D LiDAR cloud from above (world frame, coloured by height)
with the odometry path; the cloud from the side; IMU, odometry, command and
collision readout. Keys in the live window: W/S forward/back, A/D turn,
Q/E strafe, Space stop, R reset, Esc quit.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from navigation.teleop import HELP, KeyTeleop  # noqa: E402
from sim_env.launch import add_sim_args, session_from_args  # noqa: E402

# A short scripted drive (seconds, vx, vy, wz) so snapshots show motion.
DEMO_DRIVE = [(1.0, 0, 0, 0), (2.5, 0.3, 0, 0), (3.0, 0, 0, 0.5), (2.0, 0.2, 0, -0.2), (1.0, 0, 0.15, 0)]


class SensorFigure:
    def __init__(self, plt):
        self.plt = plt
        self.fig = plt.figure(figsize=(15, 8.6))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.25, wspace=0.32,
                                   left=0.04, right=0.98, top=0.93, bottom=0.15)
        ax = self.ax = {
            "rgb": self.fig.add_subplot(gs[0, 0]), "depth": self.fig.add_subplot(gs[0, 1]),
            "chase": self.fig.add_subplot(gs[0, 2]), "scan": self.fig.add_subplot(gs[1, 0]),
            "top": self.fig.add_subplot(gs[1, 1]), "side": self.fig.add_subplot(gs[1, 2]),
        }
        blank = np.zeros((240, 320, 3), np.uint8)
        self.im_rgb = ax["rgb"].imshow(blank)
        self.im_depth = ax["depth"].imshow(np.zeros((240, 320)), cmap="turbo", vmin=0, vmax=6)
        self.fig.colorbar(self.im_depth, ax=ax["depth"], fraction=0.035, label="depth (m)")
        self.im_chase = ax["chase"].imshow(np.zeros((300, 400, 3), np.uint8))
        for k in ("rgb", "depth", "chase"):
            ax[k].set_xticks([]), ax[k].set_yticks([])
        ax["scan"].set_aspect("equal")
        ax["scan"].set_xlim(-5, 5), ax["scan"].set_ylim(-5, 5)
        ax["scan"].set_xlabel("right (m)"), ax["scan"].set_ylabel("forward (m)")
        ax["scan"].grid(alpha=0.3)
        self.scan_pts = ax["scan"].scatter([], [], s=3, c="tab:red")
        ax["scan"].add_patch(plt.Polygon([[-0.15, -0.35], [0.15, -0.35], [0, 0.35]], color="k"))
        ax["top"].set_aspect("equal"), ax["top"].grid(alpha=0.3)
        ax["top"].set_xlabel("world x (m)"), ax["top"].set_ylabel("world y (m)")
        self.top_pts = ax["top"].scatter([], [], s=2, c=[], cmap="viridis", vmin=-0.3, vmax=2.5)
        self.fig.colorbar(self.top_pts, ax=ax["top"], fraction=0.035, label="height (m)")
        (self.path_line,) = ax["top"].plot([], [], "k-", lw=1.5, label="odometry path")
        (self.robot_dot,) = ax["top"].plot([], [], "ko", ms=6)
        ax["top"].legend(loc="upper right", fontsize=8)
        ax["side"].set_xlabel("forward (m, LiDAR frame)"), ax["side"].set_ylabel("up (m)")
        ax["side"].set_xlim(-6, 6), ax["side"].set_ylim(-1, 3), ax["side"].grid(alpha=0.3)
        self.side_pts = ax["side"].scatter([], [], s=2, c="tab:blue")
        self.text = self.fig.text(0.04, 0.015, "", family="monospace", fontsize=9, va="bottom")
        self.title = self.fig.suptitle("")
        self.path = []

    def track(self, sim):
        """Record the odometry path (call every control step)."""
        p = sim.state().position[:2].copy()
        if not self.path or np.linalg.norm(p - self.path[-1]) > 0.02:
            self.path.append(p)

    def update(self, sim, frame, scan, collisions, teleop_status=""):
        st = sim.state()
        self.track(sim)
        ax = self.ax
        self.im_rgb.set_data(frame.rgb)
        d = np.where(np.isfinite(frame.depth_m), frame.depth_m, np.nan)
        self.im_depth.set_data(d)
        self.im_chase.set_data(sim.render_view(400, 300))
        ax["rgb"].set_title(f"front RGB {frame.rgb.shape[1]}x{frame.rgb.shape[0]}, "
                            f"fx={frame.K[0, 0]:.1f}px", fontsize=9)
        valid = np.isfinite(frame.depth_m)
        ax["depth"].set_title(f"depth: {valid.mean() * 100:.0f}% valid, "
                              f"centre {frame.depth_m[frame.rgb.shape[0] // 2, frame.rgb.shape[1] // 2]:.2f} m",
                              fontsize=9)
        ax["chase"].set_title("third-person view", fontsize=9)
        xy = scan.scan_points_lidar()
        self.scan_pts.set_offsets(np.column_stack([-xy[:, 1], xy[:, 0]]) if len(xy) else np.zeros((0, 2)))
        n_ok = np.isfinite(scan.ranges).sum()
        ax["scan"].set_title(f"2D LiDAR scan: {n_ok}/{scan.ranges.size} returns, "
                             f"nearest {np.min(scan.ranges):.2f} m", fontsize=9)
        pw = scan.points_world()
        if len(pw):
            self.top_pts.set_offsets(pw[:, :2])
            self.top_pts.set_array(pw[:, 2])
        path = np.array(self.path)
        self.path_line.set_data(path[:, 0], path[:, 1])
        self.robot_dot.set_data([st.position[0]], [st.position[1]])
        cx, cy = st.position[:2]
        ax["top"].set_xlim(cx - 6, cx + 6), ax["top"].set_ylim(cy - 6, cy + 6)
        ax["top"].set_title(f"3D LiDAR cloud from above: {len(pw)} points", fontsize=9)
        pl = scan.points_lidar
        self.side_pts.set_offsets(pl[:, [0, 2]] if len(pl) else np.zeros((0, 2)))
        ax["side"].set_title("3D LiDAR cloud from the side (MID-360, -7 to 52 deg)", fontsize=9)
        g, a, q = st.imu.gyro, st.imu.accel, st.imu.quat_wxyz
        x, y, yaw = st.pose2d
        self.text.set_text(
            f"t={st.t:7.2f}s  ODOM x={x:+.2f} y={y:+.2f} z={st.position[2]:.3f} yaw={np.degrees(yaw):+7.1f} deg  "
            f"v=({st.lin_vel_world[0]:+.2f},{st.lin_vel_world[1]:+.2f}) m/s  upright={st.upright:.3f}\n"
            f"IMU gyro=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}) rad/s  accel=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) m/s2  "
            f"quat=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f})\n"
            f"CMD_VEL vx={st.cmd_vel[0]:+.2f} vy={st.cmd_vel[1]:+.2f} wz={st.cmd_vel[2]:+.2f}   "
            f"collisions so far: {collisions}   {teleop_status}")
        self.title.set_text(f"Go2 sensors: {sim.environment_id}")


def collect(sim, seen, counter):
    now = {(c.robot_geom, c.other_body) for c in sim.contacts()}
    counter[0] += len(now - seen)
    return now


def run_scripted(sim, fig_obj, out_png=None, out_gif=None, drive=True, fps=4):
    from PIL import Image

    frames, seen, count = [], set(), [0]
    plan = DEMO_DRIVE if drive else [(1.0, 0, 0, 0)]
    for dur, vx, vy, wz in plan:
        t_end = sim.time + dur
        while sim.time < t_end - 1e-9:
            sim.cmd_vel(vx, vy, wz)
            for _ in range(5):
                sim.step(0.2 / fps)
                fig_obj.track(sim)
            seen = collect(sim, seen, count)
            if out_gif:
                fig_obj.update(sim, sim.camera(), sim.lidar(), count[0])
                frames.append(_fig_image(fig_obj.fig))
    fig_obj.update(sim, sim.camera(), sim.lidar(), count[0])
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig_obj.fig.savefig(out_png, dpi=90)
        print(f"wrote {out_png}")
    if out_gif:
        frames.append(_fig_image(fig_obj.fig))
        small = [f.resize((f.width * 2 // 3, f.height * 2 // 3)).quantize(colors=96) for f in frames]
        small[0].save(out_gif, save_all=True, append_images=small[1:], duration=1000 // fps, loop=0)
        print(f"wrote {out_gif} ({len(frames)} frames)")
    print(f"collisions: {count[0]}")
    return count[0]


def _fig_image(fig):
    from PIL import Image
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def run_live(sim, fig_obj):
    """Interactive mode: the figure is drawn offscreen (matplotlib Agg) and shown in a
    GLFW window, so neither a matplotlib GUI toolkit nor OpenCV's Qt is needed."""
    from sim_env.display import ImageWindow

    tele = KeyTeleop()
    win = ImageWindow("Go2 sensors (click here, then drive with the keys)", 1350, 774)
    sim0, wall0 = sim.time, time.monotonic()
    seen, count = set(), [0]
    next_frame = 0.0
    print(HELP)
    try:
        while not tele.quit_requested and not win.closed:
            for k in win.keys():
                tele.press(k)
            if tele.reset_requested:
                sim.reset()
                tele.reset_requested = False
            sim.cmd_vel(*tele.target)
            target = sim0 + time.monotonic() - wall0
            n = 0
            while sim.time < target and n < 125:
                sim.step(steps=1)
                n += 1
            if sim.time < target - 0.25:
                sim0, wall0 = sim.time, time.monotonic()
            seen = collect(sim, seen, count)
            fig_obj.track(sim)
            if sim.time >= next_frame:
                fig_obj.update(sim, sim.camera(), sim.lidar(), count[0], tele.status())
                next_frame = sim.time + 0.2       # camera 5 Hz like the simulator's
                win.show(np.asarray(_fig_image(fig_obj.fig)))
            time.sleep(0.001)
    finally:
        win.close()
    print(f"collisions: {count[0]}")


def main():
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__,
                                              formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--snapshot", help="save one PNG and exit (headless)")
    ap.add_argument("--gif", help="save an animated GIF of the drive and exit (headless)")
    ap.add_argument("--demo-drive", action="store_true", help="drive a short scripted path first")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")      # live mode shows the rendered figure in a GLFW window
    import matplotlib.pyplot as plt
    headless = bool(args.snapshot or args.gif)

    sim, _ = session_from_args(args)
    try:
        sim.settle()
        fig_obj = SensorFigure(plt)
        if headless:
            run_scripted(sim, fig_obj, args.snapshot, args.gif, drive=args.demo_drive)
        else:
            run_live(sim, fig_obj)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
