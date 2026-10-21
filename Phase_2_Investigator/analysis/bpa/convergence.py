"""Least-squares intersection of lines, in 2D (area of convergence) or 3D (area of origin).

The point x minimising sum_i w_i * dist(x, line_i)^2 for lines p_i + t d_i solves
    sum_i w_i (I - d_i d_i^T) x = sum_i w_i (I - d_i d_i^T) p_i
Robust version: iteratively reweighted with Huber weights so a few stains with
a wrong direction do not drag the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LineFit:
    point: np.ndarray
    distances: np.ndarray     # distance from the point to every line
    weights: np.ndarray       # final robust weights (1 = full)
    behind: np.ndarray        # True where the point lies behind the stain (as it should)
    rms: float
    n_used: int

    @property
    def inliers(self):
        return self.weights > 0.5


def line_distances(x, points, dirs):
    r = x[None, :] - points
    t = np.sum(r * dirs, axis=1)
    return np.linalg.norm(r - t[:, None] * dirs, axis=1), t


def lstsq_lines(points, dirs, weights=None):
    P = np.asarray(points, float)
    D = np.asarray(dirs, float)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    dim = P.shape[1]
    A = np.zeros((dim, dim))
    b = np.zeros(dim)
    I = np.eye(dim)
    for p, d, wi in zip(P, D, w):
        M = I - np.outer(d, d)
        A += wi * M
        b += wi * (M @ p)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def intersect_lines(points, dirs, robust=True, huber=None, iters=10, backward_only=True):
    """Best common point of a bundle of lines.

    dirs are the directions the droplets TRAVELLED. With backward_only the
    source must lie behind each stain (t < 0 along its travel direction);
    lines whose best point lies ahead of the stain are down-weighted.
    huber: distance scale (m) above which a line's weight falls off; default
    from the median residual.
    """
    P = np.asarray(points, float)
    D = np.asarray(dirs, float)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    if len(P) < 2:
        raise ValueError("need at least two lines")
    w = np.ones(len(P))
    x = lstsq_lines(P, D, w)
    for _ in range(iters if robust else 0):
        dist, t = line_distances(x, P, D)
        scale = huber if huber is not None else max(1.5 * np.median(dist), 1e-3)
        w = np.where(dist <= scale, 1.0, scale / np.maximum(dist, 1e-12))
        if backward_only:
            w = np.where(t < 0, w, w * 0.1)
        x_new = lstsq_lines(P, D, w)
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    dist, t = line_distances(x, P, D)
    return LineFit(x, dist, w, t < 0, float(np.sqrt(np.mean(dist ** 2))), len(P))
