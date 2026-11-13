#!/usr/bin/env python3
"""Evaluate the real-world zone classifier on the held-out recording session (real-world Q1, step 3b).

    python real_go2/evaluate_real.py                    # results/real_q1/resnet18/best.pt

Uses the split from real_config.yaml (last session = test). Writes to results/real_q1/:
metrics.json (accuracy, macro-F1, per-zone precision/recall/F1, confusion counts, 1 s smoothed accuracy),
confusion_resnet18.png, curves_resnet18.png and test_predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
CONFIG = TEST2 / "real_go2/real_config.yaml"
os.environ.setdefault("Q1_CONFIG", str(CONFIG))

from q1_supervised.data import build, load_cfg  # noqa: E402
from q1_supervised.evaluate import cnn_predictions, confusion_plot, curves_plot, load_model, per_class, smooth  # noqa: E402


class _Names:
    """Minimal stand-in for go2sim.zones.Zones in cnn_predictions (only .names is used)."""

    def __init__(self, names):
        self.names = list(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="val: only write <model>/val_predictions.csv (used to tune the zone-map localiser, "
                         "real_go2/zone_map.py --tune); the test metrics are not touched")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    names = cfg["classes"]
    out = TEST2 / cfg["results_dir"]
    frames, split, cache, idx = build(cfg)
    model, ck = load_model(out / a.model / "best.pt", len(names))
    assert list(ck["classes"]) == list(names), f"model classes {ck['classes']} != config classes {names}"
    if a.split == "val":
        df = cnn_predictions(model, cache, idx["val"], frames, _Names(names)).reset_index(drop=True)
        df.to_csv(out / a.model / "val_predictions.csv", index=False)
        print(f"val session {split['val']}: {len(df)} frames, accuracy {(df.true == df.pred).mean():.3f} "
              f"-> {(out / a.model / 'val_predictions.csv').relative_to(TEST2)}")
        return
    df = cnn_predictions(model, cache, idx["test"], frames, _Names(names)).reset_index(drop=True)
    df.to_csv(out / a.model / "test_predictions.csv", index=False)
    y, p = df.true.to_numpy(), df.pred.to_numpy()
    from sklearn.metrics import f1_score
    ps = smooth(df, len(names))
    met = {"split": split, "classes": names, "n_test_frames": int(len(df)),
           "accuracy": round(float((y == p).mean()), 4),
           "macro_f1": round(float(f1_score(y, p, average="macro", labels=range(len(names)))), 4),
           "accuracy_smoothed_1s": round(float((y == ps).mean()), 4),
           "per_class": per_class(y, p, names),
           "confusion": confusion_plot(y, p, names, f"Real Go2, ResNet18: test acc {(y == p).mean():.3f}",
                                       out / f"confusion_{a.model}.png")}
    hist = out / a.model / "history.json"
    if hist.exists():
        curves_plot(json.loads(hist.read_text()), "ResNet18 on real Go2 frames", out / f"curves_{a.model}.png")
    (out / "metrics.json").write_text(json.dumps(met, indent=1))
    print(f"test session {split['test']}: {len(df)} frames, accuracy {met['accuracy']:.3f}, "
          f"macro-F1 {met['macro_f1']:.3f}, 1 s smoothed {met['accuracy_smoothed_1s']:.3f} -> {out}")
    for n, c in met["per_class"].items():
        print(f"  {n:12s} P {c['precision']:.3f} R {c['recall']:.3f} F1 {c['f1']:.3f} (n={c['support']})")


if __name__ == "__main__":
    main()
