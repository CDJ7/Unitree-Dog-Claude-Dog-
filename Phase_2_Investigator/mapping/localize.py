"""Relocalisation: where is the robot on a saved map? (Correlative scan matching.)

The real Go2's odometry starts from zero every time it boots, so a map saved in an earlier session is in
a different frame. After a look-around, the robot has LiDAR points around it in its *current* odometry
frame. `align(grid, points_xy, pose_xy, pose_yaw)` finds the rigid transform (dx, dy, dyaw) that best lays
those points onto the saved map's walls:

- a likelihood field of the map: exp(-d^2 / 2 sigma^2), d = distance to the nearest occupied cell;
- for every candidate rotation (2 deg steps), the rotated points are rasterised and cross-correlated with
  the field by FFT, which scores every translation at once; the best (rotation, translation) wins;
- a fine search (0.5 deg) around the winner refines it.
`score` is the mean field value at the transformed points (1 = every point on a wall). A score below
`min_score` means the match is not trustworthy (a different room, or too few points).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve


def likelihood_field(grid, sigma=0.10):
    occ = grid.occupied()
    d = distance_transform_edt(~occ) * grid.res if occ.any() else np.full(grid.shape, 9.0)
    return np.exp(-0.5 * (d / sigma) ** 2)


def _rot(pts, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return pts @ np.array([[c, s], [-s, c]])


def _best_shift(field, pts, res, origin):
    """Best translation (world xy) for points (already rotated), by FFT cross-correlation, and its score."""
    ij = np.floor(pts / res).astype(int)
    ij -= ij.min(axis=0)
    h, w = ij[:, 1].max() + 1, ij[:, 0].max() + 1
    kern = np.zeros((h, w), np.float32)
    np.add.at(kern, (ij[:, 1], ij[:, 0]), 1.0)
    corr = fftconvolve(field, kern[::-1, ::-1], mode="full")
    r, c = np.unravel_index(int(np.argmax(corr)), corr.shape)
    # full correlation: kernel cell (0, 0) lands on field cell (r - h + 1, c - w + 1)
    r0, c0 = r - h + 1, c - w + 1
    base = np.floor(pts / res).astype(int).min(axis=0)
    shift = origin + (np.array([c0, r0]) + 0.5) * res - (base + 0.5) * res
    return shift, float(corr[r, c]) / len(pts)


def align(grid, points_xy, pose_xy=(0.0, 0.0), pose_yaw=0.0, coarse_deg=2.0, max_points=1500, min_score=0.35,
          sigma=0.10, rng=None):
    """Transform (dx, dy, dyaw) from the current odometry frame to the map frame, plus the robot's pose on
    the map and the match score. points_xy: (N, 2) obstacle points in the current odometry frame."""
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(points_xy, float)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    field = likelihood_field(grid, sigma)
    best = (-1.0, 0.0, None)
    for yaw in np.radians(np.arange(0, 360, coarse_deg)):
        shift, sc = _best_shift(field, _rot(pts, yaw), grid.res, grid.origin)
        if shift is not None and sc > best[0]:
            best = (sc, yaw, shift)
    sc, yaw0, _ = best
    for yaw in yaw0 + np.radians(np.arange(-coarse_deg, coarse_deg + 1e-9, 0.5)):
        shift, s2 = _best_shift(field, _rot(pts, yaw), grid.res, grid.origin)
        if shift is not None and s2 >= best[0]:
            best = (s2, yaw, shift)
    # score = mean field value at the transformed points (bilinear-free: nearest cell)
    sc, yaw, shift = best
    q = _rot(pts, yaw) + shift
    r, c = grid.to_cell(q)
    ok = grid.inside(r, c)
    score = float(np.mean(np.where(ok, field[np.clip(r, 0, grid.shape[0] - 1), np.clip(c, 0, grid.shape[1] - 1)], 0)))
    robot = _rot(np.asarray(pose_xy, float)[None], yaw)[0] + shift
    return {"dx": float(shift[0]), "dy": float(shift[1]), "dyaw": float((yaw + np.pi) % (2 * np.pi) - np.pi),
            "x": float(robot[0]), "y": float(robot[1]),
            "yaw": float((pose_yaw + yaw + np.pi) % (2 * np.pi) - np.pi),
            "score": round(score, 3), "ok": bool(score >= min_score), "points": int(len(pts))}
