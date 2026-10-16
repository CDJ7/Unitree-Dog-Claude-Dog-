"""Distance and 3D-position math for fusing LiDAR points with camera detections.

Pure numpy; no SDK/camera imports so it can be unit tested with synthetic arrays.

Frames
  body:    x forward, y left, z up  (Go2 convention; LiDAR cloud is expected here)
  optical: x right, y down, z forward (pinhole camera)
"""
from dataclasses import dataclass

import numpy as np

# body -> optical axes (camera looks along body +x)
_R_BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width, height, hfov_deg):
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(width, height, fx, fx, width / 2.0, height / 2.0)

    @classmethod
    def from_config(cls, cfg, width, height):
        """cfg: {hfov_deg} or {fx, fy, cx, cy, ref_width, ref_height} (scaled to the actual frame size)."""
        if "fx" in cfg:
            sx, sy = width / cfg["ref_width"], height / cfg["ref_height"]
            return cls(width, height, cfg["fx"] * sx, cfg["fy"] * sy, cfg["cx"] * sx, cfg["cy"] * sy)
        return cls.from_hfov(width, height, cfg["hfov_deg"])


def deproject(u, v, depth, cam):
    """Pixel (u, v) and depth along the optical axis (m) -> optical-frame (x, y, z)."""
    return np.array([(u - cam.cx) / cam.fx * depth, (v - cam.cy) / cam.fy * depth, depth])


def _rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_to_camera(points_body, translation=(0.0, 0.0, 0.0), pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
    """Body-frame points (N,3) -> optical frame.

    translation: camera position in the body frame (x fwd, y left, z up, metres).
    pitch_deg > 0 means the camera is tilted down, yaw_deg > 0 turned left, roll_deg about its axis.
    """
    p = (np.asarray(points_body, dtype=np.float64) - np.asarray(translation, dtype=np.float64)) @ _R_BODY_TO_OPTICAL.T
    r = _rot_z(roll_deg) @ _rot_x(pitch_deg) @ _rot_y(yaw_deg)
    return p @ r.T


def quat_to_matrix(q_wxyz):
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64) / np.linalg.norm(q_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def odom_to_body(points_odom, position, q_wxyz):
    """World/odom-frame points -> body frame, given the body's position and orientation (w,x,y,z) in odom."""
    r = quat_to_matrix(q_wxyz)
    return (np.asarray(points_odom, dtype=np.float64) - np.asarray(position, dtype=np.float64)) @ r


def project_points(points_cam, cam, min_z=0.1):
    """Optical-frame points (N,3) -> (u, v, idx) for those in front of the camera and inside the image.

    idx are indices into points_cam, so callers can look up the matching 3D points.
    """
    p = np.asarray(points_cam, dtype=np.float64)
    if p.size == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
    z = p[:, 2]
    ok = np.isfinite(p).all(axis=1) & (z > min_z)
    idx = np.nonzero(ok)[0]
    u = cam.fx * p[idx, 0] / z[idx] + cam.cx
    v = cam.fy * p[idx, 1] / z[idx] + cam.cy
    inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return u[inside], v[inside], idx[inside]


def select_box_points(u, v, ranges, box_xyxy, central_frac=0.4, min_points=3,
                      foreground_frac=0.5, min_range=0.2, max_range=20.0):
    """Choose which projected points describe the object in a bounding box.

    Takes points inside the central `central_frac` of the box (falling back to the whole box if too
    few), drops invalid / out-of-range ones, then keeps the nearest `foreground_frac` so background
    seen through or around the object doesn't drag the distance up.
    Returns indices into u/v/ranges, or None when there aren't `min_points` valid points.
    """
    x1, y1, x2, y2 = box_xyxy
    ranges = np.asarray(ranges, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges >= min_range) & (ranges <= max_range)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for frac in (central_frac, 1.0):
        hw, hh = (x2 - x1) * frac / 2.0, (y2 - y1) * frac / 2.0
        inside = valid & (u >= mx - hw) & (u <= mx + hw) & (v >= my - hh) & (v <= my + hh)
        idx = np.nonzero(inside)[0]
        if len(idx) >= min_points:
            order = idx[np.argsort(ranges[idx])]
            k = max(min_points, int(np.ceil(len(order) * foreground_frac)))
            return order[:k]
    return None


def summarize_points(points):
    """Per-axis median of points -> (xyz, euclidean distance)."""
    xyz = np.median(np.asarray(points, dtype=np.float64), axis=0)
    return xyz, float(np.linalg.norm(xyz))


# ----------------------------------------------------------------------------------------------
# Depth-image fusion (simulation, or any camera with a registered depth image)
def estimate_from_depth(depth, box_xyxy, cam, central_frac=0.4, foreground_frac=0.5, min_valid=20,
                        min_range=0.2, max_range=20.0):
    """3D position of the object in a box from a depth image (metres along the optical axis).

    Median depth of the nearest `foreground_frac` of valid pixels in the central `central_frac` of the box
    (falls back to the whole box when there are too few), deprojected at the box centre with the intrinsics.
    Returns (xyz_optical, range_m, n_pixels) or None. The result is a point on the object's visible surface.
    """
    h, w = depth.shape
    x1, y1, x2, y2 = box_xyxy
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for frac in (central_frac, 1.0):
        hw, hh = (x2 - x1) * frac / 2.0, (y2 - y1) * frac / 2.0
        xa, xb = int(max(mx - hw, 0)), int(min(mx + hw, w))
        ya, yb = int(max(my - hh, 0)), int(min(my + hh, h))
        patch = depth[ya:yb, xa:xb].ravel()
        valid = patch[np.isfinite(patch) & (patch >= min_range) & (patch <= max_range)]
        if len(valid) >= min_valid:
            near = np.sort(valid)[:max(min_valid, int(np.ceil(len(valid) * foreground_frac)))]
            z = float(np.median(near))
            xyz = deproject(min(max(mx, 0), w - 1), min(max(my, 0), h - 1), z, cam)
            return xyz, float(np.linalg.norm(xyz)), len(near)
    return None


def optical_to_world(p_opt, cam_pos, cam_mat):
    """Optical frame (x right, y down, z fwd) -> world, given a MuJoCo camera pose (cam_mat columns: right, up, back)."""
    p_opt = np.asarray(p_opt, dtype=np.float64)
    p_mj = np.stack([p_opt[..., 0], -p_opt[..., 1], -p_opt[..., 2]], axis=-1)
    return np.asarray(cam_pos) + p_mj @ np.asarray(cam_mat).T


def world_to_optical(p_world, cam_pos, cam_mat):
    p_mj = (np.asarray(p_world, dtype=np.float64) - np.asarray(cam_pos)) @ np.asarray(cam_mat)
    return np.stack([p_mj[..., 0], -p_mj[..., 1], -p_mj[..., 2]], axis=-1)
