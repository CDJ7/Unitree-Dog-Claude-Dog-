"""Measure one stain: ellipse width, length, impact angle and travel direction.

Input is a set of 2D points covering the stain in metric surface coordinates
(pixel centres of a mask after it has been projected onto the surface plane),
plus the sampling spacing. Works for regular grids and for scattered points.

Method (follows how analysts measure by hand: fit the ellipse to the body of
the stain and ignore the tail):
  1. principal axes of the points: the long axis is the travel line;
  2. slice the stain along that axis and take the width of every slice;
     the widest slice gives the width W and the centre of the ellipse;
  3. the stain extends further on the tail side, so the shorter half
     (centre -> head end) is the clean half of the ellipse: L = 2 * that;
  4. impact angle alpha = arcsin(W / L);
  5. the travel direction (gamma) points from the centre toward the tail side.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class EllipseFit:
    center: np.ndarray        # (2,) ellipse centre (tail excluded)
    width: float
    length: float
    axis: np.ndarray          # (2,) unit long axis, pointing in the travel direction
    impact_angle_deg: float
    gamma_deg: float          # angle of `axis`, deg CCW from +u
    asymmetry: float          # (tail-side extent - head-side extent) / length; ~0 means direction unclear
    n_points: int
    area: float

    @property
    def ratio(self):
        return self.width / self.length if self.length > 0 else 1.0

    def to_dict(self):
        d = asdict(self)
        d["center"] = self.center.tolist()
        d["axis"] = self.axis.tolist()
        return d


def _smooth(x, k=3):
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(np.pad(x, k // 2, mode="edge"), w, mode="valid")


def fit_stain(points, spacing):
    """Fit an ellipse (with direction) to the 2D points of one stain."""
    pts = np.asarray(points, float)
    n = len(pts)
    if n < 6:
        raise ValueError("too few points to fit an ellipse")
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    e1 = evecs[:, 1]                      # long axis
    e2 = np.array([-e1[1], e1[0]])
    s = (pts - mean) @ e1
    q = (pts - mean) @ e2

    h = spacing
    s_min, s_max = s.min(), s.max()
    nb = max(int(np.ceil((s_max - s_min) / h)) + 1, 1)
    k = np.clip(((s - s_min) / h).astype(int), 0, nb - 1)
    widths = np.zeros(nb)
    qmid = np.zeros(nb)
    for b in range(nb):
        sel = k == b
        if sel.any():
            widths[b] = q[sel].max() - q[sel].min() + h
            qmid[b] = 0.5 * (q[sel].max() + q[sel].min())
    sb = s_min + (np.arange(nb) + 0.5) * h
    ws = _smooth(widths, 3)
    # Body of the ellipse: an ellipse's width profile obeys w^2 = W^2 (1 - (s-c)^2 / a^2),
    # a parabola in s. Fit it to the wide slices (the tail is at most ~0.7 W wide).
    body = widths >= 0.75 * ws.max()
    if body.sum() >= 3:
        C2, B1, A0 = np.polyfit(sb[body], widths[body] ** 2, 2)
    else:
        C2 = 0.0
    if C2 < 0:
        c = -B1 / (2 * C2)
        W = float(np.sqrt(max(A0 - B1 ** 2 / (4 * C2), 0.0)))
    else:                                   # degenerate (tiny or blocky stain): widest slice
        c = sb[int(np.argmax(ws))]
        W = float(ws.max())
    c = float(np.clip(c, s_min, s_max))
    head_neg = c - s_min + h / 2          # extent on the -e1 side
    ext_pos = s_max + h / 2 - c           # extent on the +e1 side
    tail_pos = ext_pos >= head_neg
    L = 2 * min(head_neg, ext_pos)
    W = min(W, widths.max())
    L = max(L, W)
    axis = e1 if tail_pos else -e1
    asym = abs(ext_pos - head_neg) / L
    bi = int(np.clip(round((c - s_min) / h - 0.5), 0, nb - 1))
    center = mean + c * e1 + qmid[bi] * e2
    alpha = float(np.degrees(np.arcsin(np.clip(W / L, 0, 1))))
    gamma = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return EllipseFit(center, float(W), float(L), axis, alpha, gamma, float(asym), n, n * h * h)


def usable(fit, min_length=1.5e-3, max_ratio=0.9, min_asymmetry=0.04):
    """Can this stain be used for direction and origin work?

    Drops stains that are too small to measure, too round to have a direction
    (W/L above max_ratio, i.e. alpha above ~64 deg), or with no visible tail.
    Returns (ok, reason).
    """
    if fit.length < min_length:
        return False, "too small"
    if fit.ratio > max_ratio:
        return False, "too round"
    if fit.asymmetry < min_asymmetry:
        return False, "direction unclear"
    return True, "ok"
