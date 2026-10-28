"""Interpretable features of a stain group, for mechanism classification.

A group is a list of `Observation`s (analysis/bpa/measure.py) that lie close
together. The features describe what an analyst looks at: how many stains,
how big and how uniform, at what angles, whether their directions agree or
radiate, whether they lie along a line or a curve, and on which surfaces.
"""
from __future__ import annotations

import numpy as np

from analysis.bpa.convergence import intersect_lines

FEATURE_NAMES = [
    "log_n", "width_med_mm", "width_iqr_mm", "width_max_mm", "width_cv",
    "alpha_med", "alpha_iqr", "frac_round", "frac_shallow",
    "dir_resultant", "along_axis", "convergence_frac",
    "linearity", "extent_m", "spread_minor_m", "curvature", "density_log",
    "frac_floor", "frac_wall", "frac_ceiling", "frac_furniture",
    "height_med", "max_area_cm2", "frac_usable",
]


def _iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def group_features(group):
    """Feature dict for one group of observations (needs at least one)."""
    n = len(group)
    W = np.array([o.fit.width for o in group]) * 1000
    A = np.array([o.alpha_deg for o in group])
    P = np.array([o.position for o in group])
    use = np.array([o.usable for o in group])
    kinds = [o.surface_kind for o in group]
    f = {"log_n": float(np.log(n)),
         "width_med_mm": float(np.median(W)), "width_iqr_mm": _iqr(W), "width_max_mm": float(W.max()),
         "width_cv": float(W.std() / max(W.mean(), 1e-9)),
         "alpha_med": float(np.median(A)), "alpha_iqr": _iqr(A),
         "frac_round": float(np.mean(A > 64)), "frac_shallow": float(np.mean(A < 30)),
         "frac_usable": float(use.mean()),
         "max_area_cm2": float(max(o.fit.area for o in group) * 1e4)}
    for k in ("floor", "wall", "ceiling", "furniture"):
        f[f"frac_{k}"] = float(np.mean([kk == k for kk in kinds]))
    f["height_med"] = float(np.median(P[:, 2]))

    # spatial shape
    if n >= 3:
        c = P - P.mean(0)
        evals, evecs = np.linalg.eigh(np.cov(c.T))
        evals = np.clip(evals[::-1], 0, None)
        axis = evecs[:, ::-1][:, 0]
        f["linearity"] = float(1 - np.sqrt(evals[1] / max(evals[0], 1e-12)))
        f["extent_m"] = float(4 * np.sqrt(evals[0]))
        f["spread_minor_m"] = float(4 * np.sqrt(evals[1]))
        # curvature: how much a parabola beats a line in the main plane of the points
        s = c @ axis
        t = c @ evecs[:, ::-1][:, 1]
        if np.ptp(s) > 1e-6 and n >= 5:
            r1 = np.polyfit(s, t, 1, full=True)[1]
            r2 = np.polyfit(s, t, 2, full=True)[1]
            r1 = float(r1[0]) if len(r1) else 0.0
            r2 = float(r2[0]) if len(r2) else 0.0
            f["curvature"] = float((r1 - r2) / max(r1, 1e-12))
        else:
            f["curvature"] = 0.0
        area = max(f["extent_m"] * max(f["spread_minor_m"], 0.01), 1e-4)
        f["density_log"] = float(np.log(n / area))
    else:
        axis = np.array([1.0, 0, 0])
        f.update(linearity=0.0, extent_m=0.0, spread_minor_m=0.0, curvature=0.0, density_log=0.0)

    # directionality (usable stains only)
    D = np.array([o.direction for o in group if o.usable])
    if len(D) >= 2:
        f["dir_resultant"] = float(np.linalg.norm(D.mean(0)))
        f["along_axis"] = float(np.mean(np.abs(D @ axis)))
    else:
        f["dir_resultant"] = 0.0
        f["along_axis"] = 0.0
    if len(D) >= 4:
        Pu = np.array([o.position for o in group if o.usable])
        T = np.array([o.measured().travel for o in group if o.usable])
        try:
            fit = intersect_lines(Pu, T)
            f["convergence_frac"] = float(np.mean((fit.distances < 0.25) & fit.behind))
        except (ValueError, np.linalg.LinAlgError):
            f["convergence_frac"] = 0.0
    else:
        f["convergence_frac"] = 0.0
    return {k: f[k] for k in FEATURE_NAMES}


def feature_vector(group):
    f = group_features(group)
    return np.array([f[k] for k in FEATURE_NAMES])
