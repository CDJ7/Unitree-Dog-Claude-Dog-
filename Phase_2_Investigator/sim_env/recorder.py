"""Run recording (A6) and reading, in the format documented in docs/recording_format.md.

    rec = RunRecorder(out_dir, sim, gt)       # then nav = Navigator(sim, on_tick=[rec]) ...
    rec.close(nav_report)

    run = RunReader(out_dir)
    for frame in run.frames():                # interfaces.CameraFrame (rgb, depth_m, K, T_world_cam)
    for scan in run.scans():                  # interfaces.LidarScan
    run.states                                # list of dicts (robot pose, velocity, IMU, cmd_vel) at 10 Hz
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sim_env.interfaces import CameraFrame, LidarScan

FORMAT_VERSION = "1.0"


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).parent).strip()
    except Exception:
        return None


class RunRecorder:
    def __init__(self, out_dir, sim, gt=None, camera_hz=5.0, lidar_hz=10.0, scenario=None):
        self.out = Path(out_dir)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        (self.out / "lidar").mkdir(parents=True, exist_ok=True)
        self.sim, self.gt = sim, gt
        self.camera_dt, self.lidar_dt = 1.0 / camera_hz, 1.0 / lidar_hz
        self.next_cam = self.next_lidar = 0.0
        self.n_frames = self.n_scans = 0
        self.frames_f = open(self.out / "frames.jsonl", "w")
        self.states_f = open(self.out / "states.jsonl", "w")
        self.scans_f = open(self.out / "scans.jsonl", "w")
        f0 = sim.camera()
        self.meta = {
            "format_version": FORMAT_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_rev": _git_rev(), "scenario": scenario or getattr(sim, "scenario_id", None),
            "environment": sim.environment_id, "camera_hz": camera_hz, "lidar_hz": lidar_hz,
            "camera": {"frame_id": f0.frame_id, "width": int(f0.rgb.shape[1]), "height": int(f0.rgb.shape[0]),
                       "K": f0.K.tolist(), "distortion": [0, 0, 0, 0, 0],
                       "convention": "OpenCV: x right, y down, z forward; T_world_cam maps camera to world"},
            "depth": "frames/NNNNNN_depth.png: uint16 millimetres along the optical axis, 0 = no return",
            "lidar": "lidar/NNNNNN.npz: ranges (planar, inf = none), angle_min, angle_increment, "
                     "points_lidar (N,3), T_world_lidar (4,4)",
            "sensors_sim_vs_real": "sim LiDAR is a MID-360 model (-7..52 deg); the real Go2 has an L1. "
                                   "Poses are simulator ground truth (no odometry drift).",
        }
        if gt is not None:
            (self.out / "ground_truth.json").write_text(json.dumps(gt))

    def __call__(self, nav):
        """Navigator on_tick hook (10 Hz)."""
        self.record(nav.scan)

    def record(self, scan=None):
        sim = self.sim
        t = sim.time
        st = sim.state()
        self.states_f.write(json.dumps({
            "t": round(t, 4), "position": st.position.round(5).tolist(), "quat_wxyz": st.quat_wxyz.round(6).tolist(),
            "yaw": round(st.yaw, 5), "lin_vel_world": st.lin_vel_world.round(4).tolist(),
            "ang_vel_body": st.ang_vel_body.round(4).tolist(), "cmd_vel": st.cmd_vel.round(4).tolist(),
            "imu": {"gyro": st.imu.gyro.round(5).tolist(), "accel": st.imu.accel.round(4).tolist(),
                    "quat_wxyz": st.imu.quat_wxyz.round(6).tolist()},
            "upright": round(st.upright, 4)}) + "\n")
        if t >= self.next_lidar:
            scan = scan or sim.lidar()
            k = self.n_scans
            np.savez_compressed(self.out / "lidar" / f"{k:06d}.npz", ranges=scan.ranges,
                                angle_min=scan.angle_min, angle_increment=scan.angle_increment,
                                range_min=scan.range_min, range_max=scan.range_max,
                                points_lidar=scan.points_lidar, T_world_lidar=scan.T_world_lidar)
            self.scans_f.write(json.dumps({"index": k, "t": round(scan.t, 4), "file": f"lidar/{k:06d}.npz"}) + "\n")
            self.n_scans += 1
            self.next_lidar = t + self.lidar_dt - 1e-6
        if t >= self.next_cam:
            f = sim.camera()
            k = self.n_frames
            Image.fromarray(f.rgb).save(self.out / "frames" / f"{k:06d}_rgb.png")
            mm = np.where(np.isfinite(f.depth_m), np.clip(f.depth_m * 1000, 0, 65535), 0).astype(np.uint16)
            Image.fromarray(mm).save(self.out / "frames" / f"{k:06d}_depth.png")
            self.frames_f.write(json.dumps({
                "index": k, "t": round(f.t, 4), "rgb": f"frames/{k:06d}_rgb.png",
                "depth": f"frames/{k:06d}_depth.png", "T_world_cam": np.round(f.T_world_cam, 6).tolist(),
                "robot_position": st.position.round(5).tolist(), "robot_yaw": round(st.yaw, 5)}) + "\n")
            self.n_frames += 1
            self.next_cam = t + self.camera_dt - 1e-6

    def close(self, nav_report=None, extra=None):
        for f in (self.frames_f, self.states_f, self.scans_f):
            f.close()
        self.meta.update(n_frames=self.n_frames, n_scans=self.n_scans, duration_s=round(self.sim.time, 2))
        if extra:
            self.meta.update(extra)
        (self.out / "meta.json").write_text(json.dumps(self.meta, indent=1))
        if nav_report is not None:
            (self.out / "nav.json").write_text(json.dumps(nav_report, indent=1, default=float))
        return self.out


class RunReader:
    """Read a recorded run back as interfaces.* objects (what Stage B consumes offline)."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.K = np.array(self.meta["camera"]["K"])
        self.frame_index = [json.loads(line) for line in open(self.dir / "frames.jsonl")]
        self.scan_index = [json.loads(line) for line in open(self.dir / "scans.jsonl")]
        self.states = [json.loads(line) for line in open(self.dir / "states.jsonl")]
        gt = self.dir / "ground_truth.json"
        self.ground_truth = json.loads(gt.read_text()) if gt.exists() else None

    def __len__(self):
        return len(self.frame_index)

    def frame(self, k):
        e = self.frame_index[k]
        rgb = np.asarray(Image.open(self.dir / e["rgb"]).convert("RGB"))
        mm = np.asarray(Image.open(self.dir / e["depth"])).astype(np.float32)
        depth = np.where(mm > 0, mm / 1000.0, np.inf).astype(np.float32)
        return CameraFrame(t=e["t"], rgb=rgb, depth_m=depth, K=self.K, T_world_cam=np.array(e["T_world_cam"]),
                           frame_id=self.meta["camera"]["frame_id"])

    def frames(self):
        for k in range(len(self.frame_index)):
            yield self.frame(k)

    def scan(self, k):
        e = self.scan_index[k]
        z = np.load(self.dir / e["file"])
        return LidarScan(t=e["t"], ranges=z["ranges"], angle_min=float(z["angle_min"]),
                         angle_increment=float(z["angle_increment"]), range_min=float(z["range_min"]),
                         range_max=float(z["range_max"]), points_lidar=z["points_lidar"],
                         T_world_lidar=z["T_world_lidar"])

    def scans(self):
        for k in range(len(self.scan_index)):
            yield self.scan(k)
