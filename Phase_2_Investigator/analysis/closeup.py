"""Measure individual bloodstains in evidence close-ups (C5).

    obs = stains_in_frame(frame)                  # one close-up (CameraFrame with depth, K, pose)
    obs = merge_observations(obs_from_all_frames) # the same stain seen in several close-ups -> one

For each red blob of a 1920 x 1080 close-up (40 deg field of view):
- blobs touching the image border are skipped (cut off, so their ellipse is wrong), and so are blobs
  under `min_px` pixels (too few to fit an ellipse) and very large ones (pools and smears are measured
  as areas, not droplets);
- the blob goes through Phase 2's measurement, `analysis.bpa.measure.from_camera_mask`, on a crop around
  it: fit the surface plane to the depth around the stain, project the mask onto that plane (undoing
  the perspective), then fit the ellipse. The result is width and length, impact angle (W/L = sin alpha),
  travel direction and world position.
Phase 2's evaluation (evaluation/results/bpa_eval.json) measured this path on single stains. The impact
angle is accurate to a median 1.1-2.0 deg up to about 1 m from the stain, and degrades fast beyond
(12 deg at 1.5 m). So the close-up tour stays within ~1 m.

`score_against_truth` matches measured stains to ground-truth stains (simulation only).
"""
from __future__ import annotations

import cv2
import numpy as np

from analysis.bpa.measure import from_camera_mask
from analysis.bpa.projection import Intrinsics


def red_mask(rgb, s_min=90, v_min=15):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 0] <= 9) | (hsv[..., 0] >= 166)) & (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)


def stains_in_frame(frame, min_px=5, max_px=60000, border=3, pad=10, frame_id="", max_blobs=400):
    """Observations of the stains in one close-up. frame: CameraFrame (rgb, depth_m, K, T_world_cam)."""
    m = red_mask(frame.rgb)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    H, W = m.shape
    K = frame.K
    order = np.argsort(-stats[1:, 4])[:max_blobs] + 1          # the biggest blobs first
    out = []
    for i in order:
        x, y, w, h, area = stats[i]
        if area < min_px or area > max_px:
            continue
        if x <= border or y <= border or x + w >= W - border or y + h >= H - border:
            continue
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
        mask = lab[y0:y1, x0:x1] == i
        intr = Intrinsics(float(K[0, 0]), float(K[1, 1]), float(K[0, 2] - x0), float(K[1, 2] - y0), x1 - x0, y1 - y0)
        try:
            ob = from_camera_mask(f"{frame_id}:{i}", mask, frame.depth_m[y0:y1, x0:x1], intr, frame.T_world_cam)
        except (ValueError, np.linalg.LinAlgError):
            continue
        ob.meta.update(frame=frame_id, bbox=[int(x), int(y), int(w), int(h)])
        out.append(ob)
    return out


def merge_observations(obs, tol=0.004):
    """The same stain seen in several close-ups: keep the view taken most head-on (then the biggest)."""
    best = sorted(obs, key=lambda o: (o.meta.get("incidence_deg", 90), -o.meta.get("pixels", 0)))
    kept, pos = [], []
    for o in best:
        p = o.position
        r = max(tol, 0.5 * float(o.fit.width))
        if any(np.linalg.norm(p - q) < max(r, rq) for q, rq in pos):
            continue
        kept.append(o)
        pos.append((p, r))
    return kept


def _angle_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def score_against_truth(obs, gt, tol=0.006):
    """Match each usable measured stain to the nearest ground-truth stain (parent drops) and report the
    impact-angle error, the direction error and how many were matched."""
    st = [s for s in gt["stains"] if s.get("kind", "parent") == "parent"]
    if not st:
        return {"measured": len(obs), "matched": 0}
    P = np.array([s["center_world"] for s in st])
    rows = []
    for o in obs:
        if not o.usable:
            continue
        d = np.linalg.norm(P - o.position, axis=1)
        k = int(np.argmin(d))
        if d[k] > max(tol, 0.6 * st[k]["length"]):
            continue
        s = st[k]
        rows.append((abs(o.alpha_deg - s["impact_angle_deg"]), _angle_between(o.direction, s["dir_world"]),
                     o.meta.get("distance", np.nan), s["id"], s["event"]))
    if not rows:
        return {"measured_usable": int(sum(o.usable for o in obs)), "matched": 0}
    a = np.array([r[0] for r in rows])
    g = np.array([r[1] for r in rows])
    return {"measured_usable": int(sum(o.usable for o in obs)), "matched": len(rows),
            "distinct_truth_stains": len({r[3] for r in rows}),
            "alpha_abs_err_median_deg": round(float(np.median(a)), 2),
            "alpha_abs_err_p90_deg": round(float(np.percentile(a, 90)), 2),
            "direction_err_median_deg": round(float(np.median(g)), 2),
            "direction_reversed_frac": round(float((g > 90).mean()), 3),
            "distance_median_m": round(float(np.nanmedian([r[2] for r in rows])), 2),
            "events": {int(e): sum(1 for r in rows if r[4] == e) for e in {r[4] for r in rows}}}
