"""Area of convergence and area of origin from measured stains.

A measured stain gives: its 3D position on a surface, the surface normal
(pointing to the side the blood came from), the travel direction along the
surface (gamma, as a 3D unit vector) and the impact angle alpha.

The droplet arrived travelling along
    travel = cos(alpha) * dir_on_surface - sin(alpha) * normal
so going back in time from the stain the straight-line path is
    stain_pos - t * travel,  t > 0.

Methods
  * straight_line_origin: 3D least-squares point of all back-traced lines.
    Works across several surfaces at once. Ignores gravity, so on a floor or
    wall it places the origin too far from the surface (it over-estimates
    height), the textbook bias of string/tangent reconstructions.
  * tangent_method: the classic two-step method on one surface: 2D area of
    convergence, then height = distance * tan(alpha) for each stain.
  * gravity_corrected_origin: replaces the straight lines by ballistic curves
    (gravity + drag) traced backwards. Impact speed is unknown, so each curve
    is chosen from a speed range (a prior for the mechanism); the result
    depends on that prior and says so.
Uncertainty: bootstrap over stains -> covariance -> 95% ellipsoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .convergence import intersect_lines

CHI2_3D_95 = 7.815
# 1-sigma model error (x, y, z in m) of the gravity-corrected origin that the resampling
# does not capture (unknown speeds and drop sizes; for projected patterns also a source
# that moves during the spurts). Calibrated so ~90-95% of synthetic events fall inside the
# 95% ellipsoid; re-check with evaluation/run_bpa_eval.py after changing the generator.
MODEL_SD = {"impact": np.array([0.04, 0.04, 0.12]),
            "projected": np.array([0.17, 0.17, 0.25]),
            "default": np.array([0.08, 0.08, 0.18])}
CHI2_2D_95 = 5.991
G_VEC = np.array([0.0, 0.0, -9.81])


@dataclass
class MeasuredStain:
    position: np.ndarray      # (3,) world
    normal: np.ndarray        # (3,) unit, toward the side the blood came from
    direction: np.ndarray     # (3,) unit travel direction along the surface
    alpha_deg: float
    width: float = 3e-3       # m, used to guess the droplet size for drag
    surface_id: str = ""
    id: str = ""

    @property
    def travel(self):
        a = np.radians(self.alpha_deg)
        return np.cos(a) * self.direction - np.sin(a) * self.normal


@dataclass
class OriginEstimate:
    point: np.ndarray
    cov: np.ndarray
    semi_axes: np.ndarray     # 95% ellipsoid semi-axis lengths, m (sorted large -> small)
    axes: np.ndarray          # columns = ellipsoid axis directions
    n_used: int
    method: str
    inliers: np.ndarray = None
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {"point": self.point.tolist(), "cov": self.cov.tolist(), "semi_axes_95": self.semi_axes.tolist(),
                "axes": self.axes.tolist(), "n_used": self.n_used, "method": self.method,
                "notes": self.notes, **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in self.extra.items()}}


def _ellipsoid(cov, chi2):
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0, None), V[:, order]
    return np.sqrt(w * chi2), V


def _bootstrap(fn, n, B, rng, dim):
    pts = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        try:
            pts.append(fn(idx))
        except (ValueError, np.linalg.LinAlgError):
            continue
    pts = np.array(pts)
    if len(pts) < 5:
        return np.eye(dim) * 1e6
    return np.cov(pts.T)


# ------------------------------------------------------------------ 2D

@dataclass
class Convergence2D:
    point_uv: np.ndarray
    point_world: np.ndarray
    cov_uv: np.ndarray
    semi_axes: np.ndarray
    axes: np.ndarray
    inliers: np.ndarray
    rms: float


def area_of_convergence(stains, plane, bootstrap=200, rng=None):
    """2D least-squares convergence of the stain direction lines on one plane."""
    rng = np.random.default_rng(0) if rng is None else rng
    P = plane.to_local(np.array([s.position for s in stains]))
    D = np.stack([[s.direction @ plane.u, s.direction @ plane.v] for s in stains])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    fit = intersect_lines(P, D)
    cov = _bootstrap(lambda idx: intersect_lines(P[idx], D[idx]).point, len(P), bootstrap, rng, 2)
    semi, axes = _ellipsoid(cov, CHI2_2D_95)
    return Convergence2D(fit.point, plane.to_world(fit.point), cov, semi, axes, fit.inliers, fit.rms)


# ------------------------------------------------------------------ 3D straight lines

def straight_line_origin(stains, bootstrap=200, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    if len(stains) < 3:
        raise ValueError("need at least three usable stains")
    P = np.array([s.position for s in stains])
    T = np.array([s.travel for s in stains])
    fit = intersect_lines(P, T)
    cov = _bootstrap(lambda idx: intersect_lines(P[idx], T[idx]).point, len(P), bootstrap, rng, 3)
    semi, axes = _ellipsoid(cov, CHI2_3D_95)
    notes = ["Straight-line trajectories ignore gravity; the distance from the surface is usually over-estimated."]
    return OriginEstimate(fit.point, cov, semi, axes, int(fit.inliers.sum()), "straight_line", fit.inliers, notes,
                          {"rms_line_distance": fit.rms})


def tangent_method(stains, plane):
    """Classic: convergence on the plane, then height_i = dist_i * tan(alpha_i). Returns (point, heights)."""
    conv = area_of_convergence(stains, plane, bootstrap=0)
    P = plane.to_local(np.array([s.position for s in stains]))
    dist = np.linalg.norm(P - conv.point_uv[None], axis=1)
    h = dist * np.tan(np.radians([s.alpha_deg for s in stains]))
    h_in = h[conv.inliers] if conv.inliers.any() else h
    point = conv.point_world + np.median(h_in) * plane.normal
    return point, h, conv


# ------------------------------------------------------------------ gravity-corrected

def _back_curves(stains, speeds, dt, t_max, drag=True, max_len=3.0):
    """Reverse-time ballistic curves for every stain and speed: (N, S, K, 3) array (NaN padded)."""
    from sim_env.spatter.constants import drag_k   # same constants as the generator

    N, S = len(stains), len(speeds)
    K = int(t_max / dt) + 1
    out = np.full((N, S, K, 3), np.nan)
    for i, s in enumerate(stains):
        d_guess = np.clip(s.width / 2.0, 3e-4, 5e-3)        # stain ~2x the drop diameter
        k = float(drag_k(d_guess)) if drag else 0.0
        x = np.repeat(s.position[None], S, axis=0).astype(float)
        v = np.asarray(speeds, float)[:, None] * s.travel[None]
        travelled = np.zeros(S)
        alive = np.ones(S, bool)
        out[i, :, 0] = x
        for j in range(1, K):
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            a = G_VEC[None] - k * sp * v
            v = np.where(alive[:, None], v - a * dt, v)   # backwards in time
            step = np.where(alive[:, None], v * dt, 0.0)
            x = x - step
            travelled += np.linalg.norm(step, axis=1)
            # stop where the path would pass through the surface it landed on, or gets too long
            # reverse-time drag speeds droplets up; past ~25 m/s the curve is meaningless
            fast = np.linalg.norm(v, axis=1) > 25.0
            alive &= ~fast & np.isfinite(x).all(axis=1)
            behind = np.where(alive, (np.nan_to_num(x) - s.position) @ s.normal, 0.0) < -1e-3
            alive &= ~behind & (travelled < max_len) & (x[:, 2] > -0.01)
            out[i, alive, j] = x[alive]
            if not alive.any():
                break
    return out


def gravity_corrected_origin(stains, start=None, speed_prior=(1.5, 4.0, 7.5), speed_step=0.1,
                             dt=4e-3, t_max=1.0, iters=100, prior_weight=0.05, bootstrap=30, rng=None,
                             model_sd=None):
    """Origin from back-traced ballistic curves.

    speed_prior = (low, typical, high) impact speed in m/s. Each stain picks the
    speed and point on its curve that is closest to the current estimate, with
    a small penalty for speeds far from `typical`; the estimate is the robust
    mean of those points (alternating refinement). Because the misfit is nearly
    flat along one line when all stains are on one surface, the refinement
    starts from the best of several candidates between the straight-line
    estimate and the stains.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if len(stains) < 3:
        raise ValueError("need at least three usable stains")
    lo, typ, hi = speed_prior
    # fine speed grid: a 0.2 m/s step already moves a 0.6 s back-trace by ~10 cm
    speeds = np.linspace(lo, hi, max(int(round((hi - lo) / speed_step)) + 1, 2))
    curves = _back_curves(stains, speeds, dt, t_max)
    N, S, K, _ = curves.shape
    def penalty(centre):
        return prior_weight * ((speeds - centre) / max(hi - lo, 1e-6)) ** 2   # (S,) in m^2

    pen = penalty(typ)

    def solve(idx, x0, n_iter, pen=pen):
        x = np.array(x0, float)
        C = curves[idx]
        for _ in range(n_iter):
            d2 = np.sum((C - x[None, None, None]) ** 2, axis=-1) + pen[None, :, None]
            d2 = np.where(np.isnan(d2), np.inf, d2)
            flat = d2.reshape(len(idx), -1).argmin(axis=1)
            si, ki = np.unravel_index(flat, (S, K))
            q = C[np.arange(len(idx)), si, ki]
            dist = np.linalg.norm(q - x[None], axis=1)
            scale = max(1.5 * np.median(dist), 0.02)
            w = np.where(dist <= scale, 1.0, scale / np.maximum(dist, 1e-9))
            x_new = (w[:, None] * q).sum(0) / w.sum()
            if np.linalg.norm(x_new - x) < 1e-5:
                x = x_new
                break
            x = x_new
        return x, speeds[si], dist

    def cost(idx, x, pen, cap=0.3):
        """Robust total misfit: each stain's closest (speed, point) on its curves, truncated at cap."""
        d2 = np.sum((curves[idx] - x[None, None, None]) ** 2, axis=-1) + pen[None, :, None]
        d2 = np.where(np.isnan(d2), np.inf, d2).reshape(len(idx), -1).min(axis=1)
        return float(np.minimum(d2, cap ** 2).sum())

    # With stains on one surface the curves of different speeds all pass near one line
    # (e.g. vertical above a floor pattern), so the misfit is almost flat along it and a
    # local solver stays wherever it starts. Search along that line first: from the
    # straight-line estimate (too far from the surface) toward the stains.
    if start is None:
        start = straight_line_origin(stains, bootstrap=0).point
    start = np.asarray(start, float)
    toward = np.array([s.position for s in stains]).mean(0)
    candidates = [start + tau * (toward - start) for tau in np.linspace(0.0, 0.95, 20)]

    def search(idx, pen, x_hint=None):
        cands = candidates + ([x_hint] if x_hint is not None else [])
        best = min(cands, key=lambda c: cost(idx, c, pen))
        return solve(idx, best, iters, pen)

    all_idx = np.arange(N)
    x, sp, dist = search(all_idx, pen)
    pts = []
    for _ in range(bootstrap):
        idx = rng.integers(0, N, N)
        pts.append(solve(idx, x, max(iters // 4, 10))[0])
    cov_boot = np.cov(np.array(pts).T) if len(pts) >= 5 else np.eye(3) * 1e6
    # Sensitivity to the speed assumption: re-solve favouring slow and fast impacts.
    # The spread between those answers is added as a systematic term.
    x_slow = search(all_idx, penalty(lo + 0.25 * (typ - lo)), x)[0]
    x_fast = search(all_idx, penalty(hi - 0.25 * (hi - typ)), x)[0]
    spread = x_fast - x_slow
    msd = MODEL_SD["default"] if model_sd is None else np.asarray(model_sd, float)
    cov = cov_boot + np.outer(spread, spread) + np.diag(msd) ** 2
    semi, axes = _ellipsoid(cov, CHI2_3D_95)
    notes = [f"Assumes impact speeds of {lo:g}-{hi:g} m/s (typical {typ:g}); the height depends on this assumption.",
             "Uncertainty = resampling of stains + effect of slower/faster assumed speeds + a model-error "
             "term calibrated on synthetic scenarios (evaluation/run_bpa_eval.py)."]
    return OriginEstimate(x, cov, semi, axes, N, "gravity_corrected", dist < max(3 * np.median(dist), 0.05),
                          notes, {"speeds": sp, "curve_distance": dist, "point_slow": x_slow, "point_fast": x_fast,
                                  "cov_resampling": cov_boot})


SPEED_PRIORS = {           # (low, typical, high) impact speed, m/s, per mechanism
    "impact": (1.5, 4.0, 7.5),
    "projected": (1.5, 3.0, 5.0),
    "castoff": (3.0, 6.0, 10.0),
    "drip": (1.0, 3.0, 5.0),
}
