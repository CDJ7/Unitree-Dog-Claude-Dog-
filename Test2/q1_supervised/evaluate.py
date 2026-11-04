#!/usr/bin/env python3
"""Q1: evaluate every trained zone classifier on the held-out TEST runs.

    python q1_supervised/evaluate.py

For each model folder in results/q1/ with a best.pt (CNNs) the test predictions are computed and
saved; folders that already have test_predictions.csv (histogram baselines) are read. Then:
- metrics.json: per model accuracy, macro-F1, per-class precision/recall/F1/support, accuracy on the
  held-out scene vs the held-out routes of seen scenes, accuracy by distance to a zone boundary,
  and with temporal smoothing (majority vote over the last 5 frames = 1 s, causal);
- comparison.md: the table for the README;
- confusion_<model>.png, curves_<model>.png (train/val loss + accuracy per epoch), boundary.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from go2sim.zones import Zones  # noqa: E402
from q1_supervised.data import ZoneDataset, build, load_cfg, normalize  # noqa: E402
from q1_supervised.models import build_model  # noqa: E402

ORDER = ["resnet18", "smallcnn", "resnet18_scratch", "hist_hsv_grid", "hist_hsv", "hist_rgb"]
LABEL = {"resnet18": "ResNet18 (ImageNet, fine-tuned)", "smallcnn": "Small CNN (from scratch)",
         "resnet18_scratch": "ResNet18 (random init)", "hist_hsv_grid": "HSV histogram 2x2 grid + LR",
         "hist_hsv": "HSV histogram + LR", "hist_rgb": "RGB histogram + LR"}


def load_model(ckpt_path, nc):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = build_model(ck["model"], nc) if ck["model"] != "resnet18" else _resnet_shell(nc)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def _resnet_shell(nc):
    import torchvision
    m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, nc)
    return m


@torch.no_grad()
def cnn_predictions(model, cache, idx, frames, zones, bs=128):
    dl = torch.utils.data.DataLoader(ZoneDataset(cache, idx), batch_size=bs, shuffle=False)
    probs = torch.cat([model(normalize(x)).softmax(1) for x, _ in dl]).numpy()
    df = frames.iloc[idx][["run", "scene", "route", "frame", "t", "base_x", "base_y", "base_yaw"]].copy()
    df["true"] = frames.zone_id.to_numpy()[idx]
    df["pred"] = probs.argmax(1)
    for k, n in enumerate(zones.names):
        df[f"p_{n}"] = probs[:, k].round(5)
    return df


def _acc(y, p):
    """Accuracy, or None when the subset is empty (e.g. no held-out scene in the sample config)."""
    return round(float((y == p).mean()), 4) if len(y) else None


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def boundary_distance(zones, x, y):
    """Distance (m) from (x, y) to the nearest edge of its own zone polygon."""
    out = np.full(len(x), np.nan)
    for i, (px, py) in enumerate(zip(x, y)):
        z = zones.zone_id((px, py))
        if z < 0:
            continue
        P = zones.polygons[z]
        d = np.inf
        for a, b in zip(P, np.roll(P, -1, 0)):
            ab, ap = b - a, np.array([px, py]) - a
            t = np.clip(ap @ ab / (ab @ ab), 0, 1)
            d = min(d, float(np.linalg.norm(ap - t * ab)))
        out[i] = d
    return out


def smooth(df, nc, k=5):
    """Causal majority vote over the last k frames of the same run (k=5 = 1 s at 5 Hz)."""
    out = np.empty(len(df), int)
    for _, g in df.groupby("run", sort=False):
        p = g.sort_values("frame")
        votes = p.pred.to_numpy()
        res = [np.bincount(votes[max(0, i - k + 1):i + 1], minlength=nc).argmax() for i in range(len(votes))]
        out[df.index.get_indexer(p.index)] = res
    return out


def per_class(y, p, names):
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(names)), zero_division=0)
    return {n: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4),
                "support": int(sup[i])} for i, n in enumerate(names)}


def confusion_plot(y, p, names, title, path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p, labels=range(len(names)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted zone")
    ax.set_ylabel("true zone (dog position)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm.tolist()


def curves_plot(hist, title, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], "o-", label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy loss")
    ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in hist], "o-", label="train acc (augmented)")
    ax[1].plot(ep, [h["val_acc"] for h in hist], "o-", label="val acc")
    ax[1].plot(ep, [h["val_macro_f1"] for h in hist], "s--", label="val macro-F1")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1)
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = load_cfg()
    zones = Zones()
    nc = len(zones)
    res_dir = TEST2 / cfg["results_dir"]
    frames, split, cache, idx = build(cfg)
    test_scenes = set(cfg["split"]["test_scenes"])
    metrics = {"split": {k: v for k, v in split.items()}, "n_test_frames": int(len(idx["test"])),
               "test_scenes_held_out": sorted(test_scenes), "models": {}}
    for name in ORDER:
        d = res_dir / name
        if (d / "best.pt").exists():
            model, ck = load_model(d / "best.pt", nc)
            df = cnn_predictions(model, cache, idx["test"], frames, zones)
            df.to_csv(d / "test_predictions.csv", index=False)
        elif (d / "test_predictions.csv").exists():
            df = pd.read_csv(d / "test_predictions.csv")
        else:
            continue
        df = df.reset_index(drop=True)
        y, p = df.true.to_numpy(), df.pred.to_numpy()
        from sklearn.metrics import f1_score
        ps = smooth(df, nc)
        unseen = df.scene.isin(test_scenes).to_numpy()
        bd = boundary_distance(zones, df.base_x.to_numpy(), df.base_y.to_numpy())
        bins = [0, 0.3, 0.6, 1.0, 9]
        by_bd = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (bd >= lo) & (bd < hi)
            by_bd[f"{lo}-{hi} m"] = {"acc": _acc(y[m], p[m]), "n": int(m.sum())}
        rec = {"label": LABEL[name], "accuracy": round(float((y == p).mean()), 4),
               "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
               "accuracy_unseen_scene": _acc(y[unseen], p[unseen]),
               "accuracy_unseen_routes_seen_scenes": _acc(y[~unseen], p[~unseen]),
               "accuracy_smoothed_1s": round(float((y == ps).mean()), 4),
               "macro_f1_smoothed_1s": round(float(f1_score(y, ps, average="macro")), 4),
               "accuracy_by_distance_to_zone_boundary": by_bd,
               "per_class": per_class(y, p, zones.names)}
        rec["confusion"] = confusion_plot(y, p, zones.names, f"{LABEL[name]}: test acc {rec['accuracy']:.3f}",
                                          res_dir / f"confusion_{name}.png")
        tc = d / "train_config.json"
        if tc.exists():
            info = json.loads(tc.read_text())
            rec["train"] = {k: info[k] for k in ("params", "weights_source", "epochs", "best_epoch", "best_val_macro_f1",
                                                 "wall_time_s", "C_selected", "feature_dim") if k in info}
        if (d / "history.json").exists():
            curves_plot(json.loads((d / "history.json").read_text()), LABEL[name], res_dir / f"curves_{name}.png")
        metrics["models"][name] = rec
        print(f"{name:16s} acc {rec['accuracy']:.3f}  macro-F1 {rec['macro_f1']:.3f}  unseen scene "
              f"{_fmt(rec['accuracy_unseen_scene'])}  smoothed {rec['accuracy_smoothed_1s']:.3f}", flush=True)
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    # comparison table
    lines = ["| Model | Test acc | Macro-F1 | Unseen scene acc | Unseen routes acc | Acc, 1 s smoothing |",
             "|---|---|---|---|---|---|"]
    for name, r in metrics["models"].items():
        lines.append(f"| {r['label']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {_fmt(r['accuracy_unseen_scene'])} | "
                     f"{_fmt(r['accuracy_unseen_routes_seen_scenes'])} | {r['accuracy_smoothed_1s']:.3f} |")
    best = next(iter(metrics["models"]))
    lines += ["", f"Per-class ({metrics['models'][best]['label']}):", "",
              "| Zone | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for n, c in metrics["models"][best]["per_class"].items():
        lines.append(f"| {n} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |")
    (res_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    # boundary plot
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, r in metrics["models"].items():
        b = r["accuracy_by_distance_to_zone_boundary"]
        ax.plot(range(len(b)), [v["acc"] for v in b.values()], "o-", label=r["label"])
    ax.set_xticks(range(len(b)), list(b.keys()))
    ax.set_xlabel("dog's distance to the edge of its zone")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Frames near zone boundaries are the hard ones")
    fig.tight_layout()
    fig.savefig(res_dir / "boundary.png", dpi=120)
    plt.close(fig)
    print((res_dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
