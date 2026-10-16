"""Where do the points in front of the dog come from: the sim's depth image, or the real LiDAR cloud."""
import math
import time

import numpy as np

from perception.depth_fusion import optical_to_world
from .control import corridor_obstacle_distance


def sim_points_body(world, stride=None, max_range=6.0):
    """Sim depth image -> (N,3) points in the robot body frame (x fwd, y left, z up relative to the body origin).
    stride=None samples about 80 columns whatever the resolution, so a surface 5 m away still gives several points."""
    b = world.latest_bundle()
    if b is None:
        return np.empty((0, 3))
    stride = stride or max(1, b.depth.shape[1] // 80)
    cm = world.camera_model
    d = b.depth[::stride, ::stride]
    v, u = np.mgrid[0:b.depth.shape[0]:stride, 0:b.depth.shape[1]:stride]
    ok = np.isfinite(d) & (d > 0.1) & (d < max_range)
    z = d[ok]
    p_opt = np.stack([(u[ok] + 0.5 - cm.cx) / cm.fx * z, (v[ok] + 0.5 - cm.cy) / cm.fy * z, z], axis=1)
    pw = optical_to_world(p_opt, b.cam_pos, b.cam_mat)
    x, y, yaw = b.robot_xyyaw
    dx, dy = pw[:, 0] - x, pw[:, 1] - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    base_z = float(world.data.qpos[2])
    return np.stack([c * dx - s * dy, s * dx + c * dy, pw[:, 2] - base_z], axis=1)


def make_sim_clearance(world, corridor):
    def clearance(exclude_xy):
        return corridor_obstacle_distance(sim_points_body(world), corridor, exclude_xy)
    return clearance


def make_real_clearance(lidar_src, state_src, lidar_mod, lidar_frame_mode, corridor, max_age_s=1.0):
    """Corridor check from the real LiDAR. If the cloud is missing or old it reports 0.0 (= blocked): no data, no motion."""
    from perception.pipeline import cloud_to_body
    cache = {"seq": -1, "pts": None}

    def clearance(exclude_xy):
        msg, seq, stamp = lidar_src.latest.get()
        if msg is None or time.monotonic() - stamp > max_age_s:
            return 0.0
        if seq != cache["seq"]:
            sport = state_src.sport.get()[0] if state_src is not None else None
            pts, _ = cloud_to_body(msg, sport, lidar_frame_mode, lidar_mod)
            cache["seq"], cache["pts"] = seq, pts
        if cache["pts"] is None:
            return 0.0
        return corridor_obstacle_distance(cache["pts"], corridor, exclude_xy)
    return clearance
