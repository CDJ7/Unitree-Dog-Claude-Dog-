"""Bloodstain candidate detector for the dog's camera (C2). CPU only: colour mask + region features + a
random forest. No GPU and no deep network, so it runs on the laptop next to the other robot software.

Why not colour alone: the scenes contain red distractors (towel, mug, cushion, book) in the same red as
blood, and at the dog's camera resolution most single droplets are only a few pixels. What separates them
is the *pattern* and the *geometry*:
- a stain region is many small dark-red spots (spatter) or one flat mark *lying on* a wall or floor;
- a distractor is one solid red object that *stands out* from the surface it is on.

Pipeline for one RGB frame (any size; trained on 1280 x 720, the real Go2 front-camera stream):
1. red mask in HSV (dark and washed-out reds included, because tiny stains blend with the surface);
2. regions: mask pixels closer than `link` px are grouped (closing), so a spatter cloud is one region;
3. features per region: size, spot count, fill ratio, colour statistics, shape and, when depth is known,
   the physical size, the height above the floor, the surface normal (plane fit), the flatness, and the depth
   step to the surrounding surface;
4. a random forest gives P(stain) per region.

On the real Go2 the front camera has no depth. There, the region's depth comes from the floor plane
(camera height and pitch) or from the LiDAR (C3). Without depth the forest uses only the image features
(`use_depth=False` model).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent / "models"

IMAGE_FEATURES = ["log_px", "log_hull", "fill", "n_spots", "largest_share", "spot_px_median", "h_cos", "h_sin",
                  "s_mean", "v_mean", "v_std", "s_std", "aspect", "border", "dark_frac"]
DEPTH_FEATURES = ["depth", "log_area_m2", "z_med", "nz", "flat_rms", "step", "valid_depth"]


@dataclass
class Region:
    bbox: tuple                      # x, y, w, h (pixels)
    centroid: tuple                  # u, v (pixels)
    mask: np.ndarray = field(repr=False)          # bool mask over the bbox
    feats: dict = field(default_factory=dict)
    prob: float = 0.0
    xyz: np.ndarray | None = None    # world position of the region (from depth)
    normal: np.ndarray | None = None


def red_mask(rgb, s_min=70, v_min=18, v_max=235):
    """Dark-to-bright reds, including the desaturated ones of sub-pixel stains."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= 9) | (h >= 166)) & (s >= s_min) & (v >= v_min) & (v <= v_max)), hsv


def propose(rgb, link=None, min_px=6):
    """Candidate regions: red pixels grouped by a closing of `link` px (default 1.5 % of the width)."""
    m, hsv = red_mask(rgb)
    link = link or max(3, int(round(0.015 * rgb.shape[1])))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link, link))
    grouped = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        sub = (lab[y:y + h, x:x + w] == i) & m[y:y + h, x:x + w]
        if int(sub.sum()) < min_px:
            continue
        out.append(Region(bbox=(int(x), int(y), int(w), int(h)), centroid=(float(cents[i][0]), float(cents[i][1])),
                          mask=sub))
    return out, hsv


def _image_features(r, hsv, shape):
    x, y, w, h = r.bbox
    sub = r.mask
    px = int(sub.sum())
    ys, xs = np.nonzero(sub)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts) if len(pts) >= 3 else None
    hull_area = max(float(cv2.contourArea(hull)) if hull is not None else 1.0, 1.0)
    n_sp, _, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    areas = st[1:, 4] if n_sp > 1 else np.array([px])
    vals = hsv[y:y + h, x:x + w][sub].astype(float)
    hue = np.radians(vals[:, 0] * 2.0)
    H, W = shape[:2]
    return {"log_px": np.log1p(px), "log_hull": np.log1p(hull_area), "fill": px / hull_area,
            "n_spots": np.log1p(len(areas)), "largest_share": float(areas.max()) / px,
            "spot_px_median": np.log1p(float(np.median(areas))),
            "h_cos": float(np.cos(hue).mean()), "h_sin": float(np.sin(hue).mean()),
            "s_mean": vals[:, 1].mean() / 255, "v_mean": vals[:, 2].mean() / 255, "v_std": vals[:, 2].std() / 255,
            "s_std": vals[:, 1].std() / 255, "aspect": max(w, h) / max(min(w, h), 1),
            "border": float(x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1),
            "dark_frac": float((vals[:, 2] < 90).mean())}


def _depth_features(r, frame, ring_px=6):
    """Physical size, height, surface normal, flatness and depth step from a CameraFrame with depth."""
    x, y, w, h = r.bbox
    d = frame.depth_m
    sub = r.mask
    ys, xs = np.nonzero(sub)
    dz = d[y + ys, x + xs]
    ok = np.isfinite(dz) & (dz > 0.05)
    nan = {k: 0.0 for k in DEPTH_FEATURES}
    if ok.sum() < 3:
        return nan, None, None
    u, v, z = (x + xs[ok]).astype(float), (y + ys[ok]).astype(float), dz[ok]
    P = frame.backproject(u, v, z)
    c = np.median(P, axis=0)
    Q = P - P.mean(0)
    if len(Q) >= 3:
        _, sv, vt = np.linalg.svd(Q[:: max(1, len(Q) // 400)], full_matrices=False)
        n = vt[-1]
        rms = float(sv[-1] / np.sqrt(max(len(Q[:: max(1, len(Q) // 400)]), 1)))
    else:
        n, rms = np.array([0, 0, 1.0]), 0.0
    cam = frame.T_world_cam[:3, 3]
    if np.dot(n, cam - c) < 0:
        n = -n
    # depth step: the region vs. a ring of surface just around it (flat marks have ~0; objects stand out)
    H, W = d.shape
    x0, y0, x1, y1 = max(x - ring_px, 0), max(y - ring_px, 0), min(x + w + ring_px, W), min(y + h + ring_px, H)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    ring[y - y0:y - y0 + h, x - x0:x - x0 + w] &= ~sub
    rd = d[y0:y1, x0:x1][ring]
    rd = rd[np.isfinite(rd)]
    step = float(np.median(rd) - np.median(z)) if len(rd) else 0.0
    f = frame.K[0, 0]
    zmed = float(np.median(z))
    area_m2 = float(sub.sum()) * (zmed / f) ** 2
    return ({"depth": zmed, "log_area_m2": np.log(area_m2 + 1e-6), "z_med": float(c[2]), "nz": float(abs(n[2])),
             "flat_rms": rms, "step": step, "valid_depth": float(ok.mean())}, c, n)


def region_features(frame_rgb, regions, hsv, frame=None):
    for r in regions:
        r.feats = _image_features(r, hsv, frame_rgb.shape)
        if frame is not None and getattr(frame, "depth_m", None) is not None:
            df, c, n = _depth_features(r, frame)
            r.feats.update(df)
            r.xyz, r.normal = c, n
    return regions


class StainDetector:
    """P(stain) for every red region of a frame. `model`: path to a pickle from training/train_stain_detector.py."""

    def __init__(self, model=None, use_depth=True, threshold=0.5):
        model = Path(model or MODELS / ("stain_rf_depth.pkl" if use_depth else "stain_rf_image.pkl"))
        with open(model, "rb") as fh:
            blob = pickle.load(fh)
        self.clf, self.features, self.threshold = blob["model"], blob["features"], float(blob.get("threshold", threshold))
        self.use_depth = any(f in DEPTH_FEATURES for f in self.features)

    def regions(self, frame):
        """All candidate regions of a CameraFrame (or RGB array), each with .prob."""
        rgb = frame.rgb if hasattr(frame, "rgb") else frame
        regs, hsv = propose(rgb)
        if not regs:
            return []
        region_features(rgb, regs, hsv, frame if (self.use_depth and hasattr(frame, "depth_m")) else None)
        X = np.array([[r.feats.get(f, 0.0) for f in self.features] for r in regs], float)
        p = self.clf.predict_proba(X)[:, 1]
        for r, pi in zip(regs, p):
            r.prob = float(pi)
        return regs

    def __call__(self, frame):
        """Regions judged to be stains (prob >= threshold)."""
        return [r for r in self.regions(frame) if r.prob >= self.threshold]


def draw(rgb, regions, threshold=0.5):
    out = rgb.copy()
    for r in regions:
        x, y, w, h = r.bbox
        col = (255, 40, 40) if r.prob >= threshold else (120, 120, 120)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.putText(out, f"{r.prob:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out
