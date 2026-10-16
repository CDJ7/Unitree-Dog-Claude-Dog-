"""Fuse detections with the SIM's depth image -> distance, camera-frame, body-frame and world-frame positions."""
import dataclasses
import math

import numpy as np

from perception import depth_fusion as df


def make_sim_fuse_fn(world, fusion_cfg):
    """Returns fn(frame, detections) -> (detections, None, status) for perception.pipeline.DetectionThread."""
    cm = world.camera_model

    def fuse(frame, dets):
        b = world.bundle_for(frame)
        if b is None:
            return dets, None, "no sim pose for this frame"
        x, y, yaw = b.robot_xyyaw
        c, s = math.cos(-yaw), math.sin(-yaw)
        out = []
        for d in dets:
            est = df.estimate_from_depth(b.depth, d.bbox, cm, fusion_cfg["central_frac"], fusion_cfg["foreground_frac"],
                                         fusion_cfg.get("min_depth_pixels", 20), fusion_cfg["min_range_m"],
                                         fusion_cfg["max_range_m"])
            if est is None:
                out.append(d)                      # "depth: n/a"
                continue
            xyz_opt, rng, n = est
            xw = df.optical_to_world(xyz_opt, b.cam_pos, b.cam_mat)
            dx, dy = xw[0] - x, xw[1] - y
            xb = (c * dx - s * dy, s * dx + c * dy, xw[2])          # robot frame: x fwd, y left, z up (above floor)
            out.append(dataclasses.replace(d, distance_m=rng, xyz_cam=tuple(float(v) for v in xyz_opt),
                                           xyz_body=tuple(float(v) for v in xb), xyz_world=tuple(float(v) for v in xw),
                                           n_depth_points=n))
        return out, None, "depth: sim depth image"

    return fuse
