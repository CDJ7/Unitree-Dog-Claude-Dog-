"""Camera geometry: pixels + depth -> world points -> surface plane coordinates.

Conventions
  * Camera frame is the OpenCV optical frame: x right, y down, z forward.
    (The simulator's `front_camera_optical_frame` uses the same convention.)
  * A pose is a 4x4 matrix T_world_cam mapping camera-frame points to world.
  * Depth is metric distance along the optical z axis (not along the ray),
    as published on the sim's /front/depth topic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fovy(cls, width, height, fovy_deg):
        """Pinhole intrinsics from a vertical field of view (MuJoCo cameras use fovy)."""
        fy = height / (2 * np.tan(np.radians(fovy_deg) / 2))
        return cls(fy, fy, (width - 1) / 2, (height - 1) / 2, width, height)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def look_at(position, target, up=(0, 0, 1)):
    """T_world_cam for an optical-frame camera at `position` looking at `target`."""
    p = np.asarray(position, float)
    z = np.asarray(target, float) - p
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:                      # looking straight up/down
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, p
    return T


def deproject(px, py, depth, intr):
    """Pixel coordinates (arrays) and z-depth -> (N, 3) points in the camera frame."""
    px, py, depth = (np.asarray(a, float) for a in (px, py, depth))
    x = (px - intr.cx) / intr.fx * depth
    y = (py - intr.cy) / intr.fy * depth
    return np.stack([x, y, depth], axis=-1)


def project(points_world, T_world_cam, intr):
    """World points -> (pixel x, pixel y, depth). Points behind the camera get depth <= 0."""
    T_cam_world = np.linalg.inv(T_world_cam)
    pc = (np.asarray(points_world, float) @ T_cam_world[:3, :3].T) + T_cam_world[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * pc[:, 0] / z + intr.cx
        v = intr.fy * pc[:, 1] / z + intr.cy
    return u, v, z


def to_world(points_cam, T_world_cam):
    return np.asarray(points_cam, float) @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def mask_to_world(mask, depth, intr, T_world_cam):
    """World points of every valid-depth pixel in a boolean mask."""
    py, px = np.nonzero(mask)
    d = depth[py, px]
    ok = np.isfinite(d) & (d > 0)
    return to_world(deproject(px[ok], py[ok], d[ok], intr), T_world_cam)


@dataclass
class Plane:
    origin: np.ndarray        # a point on the plane
    normal: np.ndarray        # unit, pointing toward the camera / free space
    u: np.ndarray
    v: np.ndarray
    rms: float = 0.0          # fit residual, m

    def to_local(self, p):
        d = np.asarray(p, float) - self.origin
        return np.stack([d @ self.u, d @ self.v], axis=-1)

    def to_world(self, uv):
        uv = np.asarray(uv, float)
        return self.origin + uv[..., :1] * self.u + uv[..., 1:2] * self.v

    def dir_to_world(self, d2):
        d2 = np.asarray(d2, float)
        return d2[..., :1] * self.u + d2[..., 1:2] * self.v


def plane_frame(normal):
    """Same (u, v) convention as sim_env.spatter.geometry.frame_from_normal."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    hint = np.array([0.0, 1.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    v = hint - (hint @ n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    return u, v


def fit_plane(points, toward=None):
    """Least-squares plane through 3D points; normal oriented toward `toward` if given."""
    p = np.asarray(points, float)
    c = p.mean(axis=0)
    _, s, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    if toward is not None and (np.asarray(toward, float) - c) @ n < 0:
        n = -n
    u, v = plane_frame(n)
    rms = float(s[2] / np.sqrt(len(p)))
    return Plane(c, n, u, v, rms)


def pixel_footprint(distance, intr, incidence_deg=0.0):
    """Approximate size (m) of one pixel on a surface at `distance`, viewed at an incidence angle."""
    return distance / intr.fx / max(np.cos(np.radians(incidence_deg)), 0.2)
