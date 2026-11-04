#!/usr/bin/env python3
"""Q1 baselines: colour-histogram features + logistic regression (colour spaces RGB vs HSV).

    python q1_supervised/baselines.py

Features per frame (168x224, same frames and same by-run split as the CNNs):
- hist_rgb:      joint 8x8x8 RGB histogram (512-d), normalised to sum 1
- hist_hsv:      joint 8x8x8 HSV histogram (512-d): hue/saturation separate colour from brightness,
                 so it should be less sensitive to the per-run lighting randomisation
- hist_hsv_grid: the HSV histogram of each cell of a 2x2 grid (4 x 512-d): adds coarse spatial
                 layout (what is left/right, floor/wall), the cue a CNN exploits much better
Classifier: standardisation + multinomial logistic regression with balanced class weights; C chosen
on the validation runs, then refit on train and evaluated once on test.
Writes results/q1/<name>/test_predictions.csv and train_config.json (evaluate.py builds the tables).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from go2sim.zones import Zones  # noqa: E402
from q1_supervised.data import build, load_cfg  # noqa: E402


def rgb_to_hsv(x):
    """uint8 N x 3 x H x W -> float HSV in [0,1], same shape (vectorised)."""
    x = x.astype(np.float32) / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx, mn = x.max(1), x.min(1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rr = m & (mx == r)
    gg = m & (mx == g) & ~rr
    bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / d[rr]) % 6
    h[gg] = (b - r)[gg] / d[gg] + 2
    h[bb] = (r - g)[bb] / d[bb] + 4
    h /= 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h, s, mx], 1)


def joint_hist(ch, bins):
    """ch: N x 3 x P values in [0,1] -> N x bins^3 normalised joint histogram."""
    q = np.clip((ch * bins).astype(np.int32), 0, bins - 1)
    code = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    out = np.zeros((len(ch), bins ** 3), np.float32)
    for i in range(len(ch)):
        out[i] = np.bincount(code[i], minlength=bins ** 3)
    return out / ch.shape[2]


def features(images, kind, bins, chunk=512):
    feats = []
    for s in range(0, len(images), chunk):
        x = images[s:s + chunk].numpy()
        n, _, H, W = x.shape
        v = rgb_to_hsv(x) if kind != "hist_rgb" else x.astype(np.float32) / 255.0
        if kind == "hist_hsv_grid":
            cells = [v[:, :, i * H // 2:(i + 1) * H // 2, j * W // 2:(j + 1) * W // 2].reshape(n, 3, -1)
                     for i in range(2) for j in range(2)]
            feats.append(np.concatenate([joint_hist(c, bins) for c in cells], 1))
        else:
            feats.append(joint_hist(v.reshape(n, 3, -1), bins))
    return np.concatenate(feats)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cfg = load_cfg()
    zones = Zones()
    frames, split, cache, idx = build(cfg)
    hc = cfg["histogram"]
    for kind in ("hist_rgb", "hist_hsv", "hist_hsv_grid"):
        t0 = time.time()
        X = features(cache.images, kind, hc["bins"])
        y = frames.zone_id.to_numpy()
        scores = {}
        for C in hc["C_grid"]:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
            clf.fit(X[idx["train"]], y[idx["train"]])
            scores[C] = f1_score(y[idx["val"]], clf.predict(X[idx["val"]]), average="macro")
        C = max(scores, key=scores.get)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        clf.fit(X[idx["train"]], y[idx["train"]])
        te = idx["test"]
        prob = clf.predict_proba(X[te])
        out = TEST2 / cfg["results_dir"] / kind
        out.mkdir(parents=True, exist_ok=True)
        df = frames.iloc[te][["run", "scene", "route", "frame", "t", "base_x", "base_y", "base_yaw"]].copy()
        df["true"] = y[te]
        df["pred"] = prob.argmax(1)
        for k, n in enumerate(zones.names):
            df[f"p_{n}"] = prob[:, k].round(5)
        df.to_csv(out / "test_predictions.csv", index=False)
        info = {"model": kind, "feature_dim": int(X.shape[1]), "bins_per_channel": hc["bins"],
                "classifier": "StandardScaler + LogisticRegression(class_weight=balanced)",
                "C_selected": C, "val_macro_f1_by_C": {str(k): round(v, 4) for k, v in scores.items()},
                "seed": cfg["seed"], "split": split, "wall_time_s": round(time.time() - t0, 1)}
        (out / "train_config.json").write_text(json.dumps(info, indent=1))
        print(f"{kind}: dim {X.shape[1]}, C={C}, val macro-F1 {scores[C]:.3f}, "
              f"test acc {(df.true == df.pred).mean():.3f} ({info['wall_time_s']} s)", flush=True)


if __name__ == "__main__":
    main()
