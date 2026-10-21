#!/usr/bin/env python3
"""Train the mechanism classifier on synthetic ground truth and report accuracy.

    python training/train_mechanism_classifier.py               # 600 scenarios
    python training/train_mechanism_classifier.py --n 2000 --seed 1

Steps: random scenarios (mechanisms balanced) -> generate spatter -> simulated
measurement with varied resolution and occlusion -> stain groups -> features
-> gradient boosting. Splits are by scenario (no group of a test scenario is
seen in training). Writes:
  analysis/classifier/models/mechanism_gb.pkl     (gitignored)
  evaluation/results/mechanism_confusion.png
  evaluation/results/mechanism_classifier.json    accuracy, confusion, importances
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from analysis.bpa.measure import group_observations  # noqa: E402
from analysis.classifier.features import FEATURE_NAMES, group_features  # noqa: E402
from analysis.classifier.model import DEFAULT_MODEL, MechanismClassifier  # noqa: E402
from evaluation.observe import label_group, observe  # noqa: E402
from sim_env.scenarios import scenario_paths  # noqa: E402
from sim_env.scenarios.random_scenario import MECHS, random_scenario  # noqa: E402
from sim_env.spatter import generate, load_scenario  # noqa: E402

RESULTS = ROOT / "evaluation" / "results"


def groups_of(gt, rng, min_group=3):
    """(features, label, purity) for every stain group of one scenario."""
    res = rng.uniform(1.5e-4, 4e-4)
    obs, _ = observe(gt, res=res, drop=rng.uniform(0.0, 0.3), rng=rng)
    out = []
    for g in group_observations(obs, link=0.3):
        if len(g) < min_group and not (len(g) == 1 and g[0].reason == "smear"):
            continue
        mech, _, purity = label_group(g, gt)
        out.append((group_features(g), mech, purity))
    return out


def build(seeds, base_seed):
    X, y, sid = [], [], []
    rng = np.random.default_rng(base_seed)
    for s in seeds:
        scn = random_scenario(int(s), mechanism=MECHS[int(s) % len(MECHS)])
        gt = generate(scn)
        for f, m, purity in groups_of(gt, rng):
            if purity < 0.7:          # mixed groups have no single right answer
                continue
            X.append([f[k] for k in FEATURE_NAMES])
            y.append(m)
            sid.append(int(s))
    return np.array(X), np.array(y), np.array(sid)


def confusion(y_true, y_pred, classes):
    M = np.zeros((len(classes), len(classes)), int)
    ix = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        M[ix[t], ix[p]] += 1
    return M


def plot_confusion(M, classes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    norm = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600, help="number of random training scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    a = ap.parse_args()
    t0 = time.time()
    seeds = np.arange(a.seed * 100000, a.seed * 100000 + a.n)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(seeds)
    n_tr, n_va = int(0.7 * a.n), int(0.15 * a.n)
    Xtr, ytr, _ = build(seeds[:n_tr], a.seed + 1)
    Xva, yva, _ = build(seeds[n_tr:n_tr + n_va], a.seed + 2)
    Xte, yte, _ = build(seeds[n_tr + n_va:], a.seed + 3)
    print(f"groups: train {len(ytr)}, val {len(yva)}, test {len(yte)}  ({time.time() - t0:.0f}s)")

    clf = MechanismClassifier.train(Xtr, ytr, seed=a.seed)
    classes = list(clf.classes)
    acc_va = float(np.mean(clf.model.predict(Xva) == yva))
    pred = clf.model.predict(Xte)
    acc_te = float(np.mean(pred == yte))
    M = confusion(yte, pred, classes)

    # the four hand-written scenarios, as a separate showcase check
    showcase = {}
    rng = np.random.default_rng(99)
    for p in scenario_paths():
        gt = generate(load_scenario(p))
        res = []
        for f, m, purity in groups_of(gt, rng):
            probs = clf.predict_proba_features(f)
            res.append({"true": m, "pred": max(probs, key=probs.get), "conf": round(max(probs.values()), 3),
                        "purity": round(purity, 2), "n": int(round(np.exp(f["log_n"])))})
        showcase[p.stem] = res

    path = clf.save(a.out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_confusion(M, classes, f"Mechanism classifier, test groups (accuracy {acc_te:.0%})",
                   RESULTS / "mechanism_confusion.png")
    report = {"n_scenarios": a.n, "seed": a.seed, "groups": {"train": len(ytr), "val": len(yva), "test": len(yte)},
              "accuracy": {"val": acc_va, "test": acc_te}, "classes": classes, "confusion_test": M.tolist(),
              "feature_importance": clf.feature_importance(), "showcase": showcase,
              "note": "Synthetic data with idealised segmentation; real-image accuracy will be lower."}
    (RESULTS / "mechanism_classifier.json").write_text(json.dumps(report, indent=1))
    print(f"val accuracy {acc_va:.3f}  test accuracy {acc_te:.3f}")
    print("confusion (rows true, cols predicted):", classes)
    print(M)
    print("top features:", list(clf.feature_importance())[:6])
    for k, v in showcase.items():
        print(k, [(r["true"], r["pred"], r["conf"], r["n"]) for r in v])
    print(f"model -> {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
