"""Real-robot side of mapping: LiDAR-fused detections (xyz in the robot body frame) + the robot pose from
rt/sportmodestate -> world (odom) frame observations for ObjectMap. NOT yet tested on the real robot."""
import numpy as np

from perception.depth_fusion import quat_to_matrix


def body_to_world(xyz_body, position, quat_wxyz):
    """Robot body frame (x fwd, y left, z up) -> odom/world frame using the sportmodestate pose."""
    r = quat_to_matrix(quat_wxyz)
    return r @ np.asarray(xyz_body, dtype=float) + np.asarray(position, dtype=float)


def make_on_result(state_source, objmap):
    """DetectionThread on_result hook for the real robot: feeds LiDAR-based detections into the map."""
    def on_result(res):
        s = state_source.sport.get()[0] if state_source else None
        if s is None:
            return
        pos, q = list(s.position), list(s.imu_state.quaternion)
        for d in res.detections:
            if d.xyz_body is None or d.distance_m is None:
                continue
            xw = body_to_world(d.xyz_body, pos, q)
            objmap.add(d.name, xw, d.distance_m, ray_origin=pos, robot_xy=pos[:2])
    return on_result
