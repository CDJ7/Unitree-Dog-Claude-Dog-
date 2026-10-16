"""Detection thread: latest camera frame -> YOLO -> LiDAR fusion -> latest result buffer.

Runs on its own thread so slow inference never blocks camera capture, display or teleop.
"""
import csv
import dataclasses
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import depth_fusion as df

log = logging.getLogger("perception")


@dataclass
class DetectionResult:
    frame: np.ndarray
    detections: List
    lidar_uvr: Optional[np.ndarray]      # (N,3): u, v, range for the overlay
    infer_fps: float
    seq: int
    wall_time: float
    status: str = ""


def fuse_detections(detections, frame_shape, cloud_msg, sport_state, cfg, lidar_mod):
    """Attach distance / xyz to detections using one LiDAR cloud. Returns (detections, uvr, status)."""
    if cloud_msg is None:
        return detections, None, "no LiDAR data yet"
    h, w = frame_shape[:2]
    cam = df.CameraModel.from_config(cfg["camera_model"], w, h)
    ext = cfg["extrinsics"]
    pts = lidar_mod.pointcloud2_to_xyz(cloud_msg).astype(np.float64)
    if not len(pts):
        return detections, None, "empty point cloud"

    frame_id = str(getattr(cloud_msg.header, "frame_id", ""))
    mode = cfg["fusion"].get("lidar_frame", "auto")
    is_odom = mode == "odom" or (mode == "auto" and "odom" in frame_id.lower())
    if is_odom:
        if sport_state is None:
            return detections, None, "cloud is in odom frame but no sportmodestate pose yet"
        pts = df.odom_to_body(pts, sport_state.position, sport_state.imu_state.quaternion)

    pts_cam = df.body_to_camera(pts, ext["translation"], ext["pitch_deg"], ext["yaw_deg"], ext["roll_deg"])
    u, v, idx = df.project_points(pts_cam, cam)
    ranges = np.linalg.norm(pts_cam[idx], axis=1)
    fz = cfg["fusion"]
    out = []
    for d in detections:
        sel = df.select_box_points(u, v, ranges, d.bbox, fz["central_frac"], fz["min_points"],
                                   fz["foreground_frac"], fz["min_range_m"], fz["max_range_m"])
        if sel is None:
            out.append(d)                                  # distance_m stays None -> "depth: n/a"
            continue
        near = idx[sel]
        xyz_cam, dist = df.summarize_points(pts_cam[near])
        xyz_body, _ = df.summarize_points(pts[near])
        out.append(dataclasses.replace(d, distance_m=dist, xyz_cam=tuple(float(v) for v in xyz_cam),
                                       xyz_body=tuple(float(v) for v in xyz_body), n_depth_points=len(near)))
    return out, np.column_stack([u, v, ranges]), f"frame_id={frame_id or '?'}"


class DetectionLogger:
    """Writes detections to .csv (one row per detection) or .jsonl (one line per frame)."""

    COLUMNS = ["wall_time", "frame_seq", "class", "conf", "x1", "y1", "x2", "y2", "distance_m",
               "x_cam", "y_cam", "z_cam", "x_body", "y_body", "z_body", "n_depth_points"]

    def __init__(self, path):
        self.jsonl = path.lower().endswith((".jsonl", ".json"))
        self._f = open(path, "w", newline="")
        self._csv = None if self.jsonl else csv.writer(self._f)
        if self._csv:
            self._csv.writerow(self.COLUMNS)

    def write(self, res):
        if self.jsonl:
            self._f.write(json.dumps({"t": res.wall_time, "seq": res.seq,
                                      "detections": [dataclasses.asdict(d) for d in res.detections]}) + "\n")
        else:
            for d in res.detections:
                xc = d.xyz_cam or (None,) * 3
                xb = d.xyz_body or (None,) * 3
                self._csv.writerow([res.wall_time, res.seq, d.name, f"{d.conf:.3f}", *[f"{v:.1f}" for v in d.bbox],
                                    d.distance_m, *xc, *xb, d.n_depth_points])
        self._f.flush()

    def close(self):
        self._f.close()


class DetectionThread(threading.Thread):
    def __init__(self, detector, camera_buffer, cfg, lidar_source=None, state_source=None,
                 lidar_mod=None, logger=None, fuse_fn=None, on_result=None):
        super().__init__(name="detection", daemon=True)
        self.detector, self.camera_buffer, self.cfg = detector, camera_buffer, cfg
        self.lidar, self.state, self.lidar_mod, self.logger = lidar_source, state_source, lidar_mod, logger
        self.fuse_fn = fuse_fn          # fn(frame, detections) -> (detections, uvr, status); replaces LiDAR fusion (sim)
        self.on_result = on_result      # fn(DetectionResult), called on this thread for every result (mapping)
        self._lock = threading.Lock()
        self._result = None
        self._stop_evt = threading.Event()
        self._frame_id_logged = False

    def stop(self):
        self._stop_evt.set()

    def latest(self):
        with self._lock:
            return self._result

    def run(self):
        last_seq, times, fps = 0, [], 0.0
        while not self._stop_evt.is_set():
            frame, seq, _ = self.camera_buffer.get()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                dets = self.detector.detect(frame)
                if self.fuse_fn is not None:
                    dets, uvr, status = self.fuse_fn(frame, dets)
                    cloud = None
                    sport = None
                else:
                    cloud = self.lidar.latest.get()[0] if self.lidar else None
                    sport = self.state.sport.get()[0] if self.state else None
                if cloud is not None and not self._frame_id_logged:
                    self._frame_id_logged = True
                    log.info("LiDAR cloud header.frame_id = %r (lidar_frame=%s)",
                             getattr(cloud.header, "frame_id", None), self.cfg["fusion"].get("lidar_frame", "auto"))
                if self.fuse_fn is None:
                    dets, uvr, status = fuse_detections(dets, frame.shape, cloud, sport, self.cfg, self.lidar_mod) \
                        if self.lidar_mod else (dets, None, "LiDAR disabled")
            except Exception:
                log.exception("detection/fusion failed on a frame")
                time.sleep(0.2)
                continue
            now = time.monotonic()
            times.append(now)
            while times and now - times[0] > 2.0:
                times.pop(0)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            res = DetectionResult(frame, dets, uvr, fps, seq, time.time(), status)
            with self._lock:
                self._result = res
            if self.logger:
                self.logger.write(res)
            if self.on_result:
                try:
                    self.on_result(res)
                except Exception:
                    log.exception("on_result callback failed")
