#!/usr/bin/env python3
"""Q1: Grad-CAM heatmaps of the fine-tuned ResNet18 on test frames.

    python q1_supervised/gradcam.py                     # -> results/q1/gradcam.png

Grad-CAM (Selvaraju et al., 2017): the gradient of the predicted class score with respect to the
last convolutional block (layer4, 6x7 cells for a 168x224 input) is averaged per channel to weight
that block's activations; ReLU of the weighted sum, upsampled to the image, shows which pixels
pushed the network towards its answer. Per zone, the 3 most confident correct test frames from
different runs are shown, plus the most confident mistake (if any), with its true and predicted
zones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from go2sim.zones import Zones  # noqa: E402
from q1_supervised.data import build, load_cfg, normalize  # noqa: E402
from q1_supervised.evaluate import load_model  # noqa: E402


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        out = self.model(normalize(x))
        cls = out.argmax(1) if cls is None else cls
        out.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).max(1)[0].clamp_min(1e-8).view(-1, 1, 1)
        return cam.detach().numpy(), out.softmax(1).detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--out", default=str(TEST2 / "results/q1/gradcam.png"))
    a = ap.parse_args()
    cfg = load_cfg()
    zones = Zones()
    frames, split, cache, idx = build(cfg)
    model, _ = load_model(TEST2 / cfg["results_dir"] / a.model / "best.pt", len(zones))
    for p in model.parameters():
        p.requires_grad_(True)
    cam = GradCAM(model, model.layer4)
    pred = pd.read_csv(TEST2 / cfg["results_dir"] / a.model / "test_predictions.csv")
    pos = {(r, f): i for i, (r, f) in enumerate(zip(frames.run, frames.frame))}
    pred["conf"] = pred[[f"p_{n}" for n in zones.names]].max(axis=1)
    picks = []
    for zi, zname in enumerate(zones.names):
        ok = pred[(pred.true == zi) & (pred.pred == zi)].sort_values("conf", ascending=False)
        ok = ok.drop_duplicates("run").head(3)
        bad = pred[(pred.true == zi) & (pred.pred != zi)].sort_values("conf", ascending=False).head(1)
        picks.append([*ok.itertuples(), *bad.itertuples()])
    ncol = max(len(p) for p in picks)
    fig, axes = plt.subplots(len(zones), ncol, figsize=(ncol * 3.0, len(zones) * 2.45))
    for zi, row in enumerate(picks):
        for j in range(ncol):
            ax = axes[zi, j]
            ax.axis("off")
            if j >= len(row):
                continue
            r = row[j]
            i = pos[(r.run, r.frame)]
            x = cache.images[i:i + 1].float() / 255.0
            heat, prob = cam(x)
            img = x[0].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.imshow(heat[0], cmap="jet", alpha=0.42, vmin=0, vmax=1)
            ok = r.pred == r.true
            ax.set_title(f"{'OK' if ok else 'WRONG'}: pred {zones.names[r.pred]} ({prob[0].max():.2f})\n"
                         f"{r.run} #{r.frame}" + ("" if ok else f", true {zones.names[r.true]}"),
                         fontsize=7, color="black" if ok else "red")
        axes[zi, 0].text(-0.06, 0.5, zones.names[zi], transform=axes[zi, 0].transAxes, rotation=90, va="center",
                         ha="right", fontsize=9, color=zones.colors[zi], weight="bold")
    fig.suptitle("Grad-CAM (ResNet18 layer4) on held-out test frames: where the network looks")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
