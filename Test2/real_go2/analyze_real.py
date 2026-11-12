#!/usr/bin/env python3
"""Figures and error analysis for the real-Go2 zone classifier (REPORT.md section 7.6). No robot, no GPU.

    python real_go2/analyze_real.py                       # images from data_real/ (for the sample-frames figure)
    python real_go2/analyze_real.py --data /path/to/runs  # recordings somewhere else

Reads results/real_q1/{metrics.json, resnet18/history.json, resnet18/test_predictions.csv} and the per-session
states.csv / metadata.json (data_real/run_*/ or the copies in results/real_q1/dataset/). Writes to results/real_q1/:
  confusion_resnet18.png, curves_resnet18.png   (same plots as evaluate_real.py, from the saved predictions)
  dataset_summary.json                          frames per zone and session, durations
  error_analysis.json                           accuracy vs time to the nearest zone change, vs confidence,
                                                vs the dog's heading (odometry yaw), longest error streaks
  real_sample_frames.png                        two training frames per zone (needs the images)
  real_test_session.png                         test session: odometry path coloured by true zone with the
                                                errors marked, and true vs predicted zone over time
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from q1_supervised.evaluate import confusion_plot, curves_plot  # noqa: E402

ZC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # zone colours (lab_1, lab_2, open_area, shelves)


def sessions(data, out):
    """{run: (states DataFrame, metadata dict, images dir or None)} from data_real/ or results/.../dataset/."""
    res = {}
    for st in sorted(data.glob("run_*/states.csv")):
        r = st.parent
        res[r.name] = (pd.read_csv(st), json.loads((r / "metadata.json").read_text()), r / "images")
    if not res:
        for st in sorted((out / "dataset").glob("run_*_states.csv")):
            n = st.name[:7]
            res[n] = (pd.read_csv(st), json.loads((out / "dataset" / f"{n}_metadata.json").read_text()), None)
    return res


def dataset_summary(sess, classes, split):
    rows = {}
    for n, (st, meta, _) in sess.items():
        lab = st[st.zone.notna() & (st.zone != "")]
        part = next((k for k, v in split.items() if n in v), "-")
        rows[n] = {"split": part, "frames": int(len(lab)),
                   "per_zone": {c: int((lab.zone == c).sum()) for c in classes},
                   "labelled_walk_s": round(float(lab.t.max() - lab.t.min()), 1),
                   "session_s": meta.get("duration_s"), "started": meta.get("started"), "source": meta.get("source")}
    tot = {c: sum(r["per_zone"][c] for r in rows.values()) for c in classes}
    return {"sessions": rows, "total_frames": sum(tot.values()), "total_per_zone": tot}


def error_analysis(df, classes):
    df = df.sort_values("frame").reset_index(drop=True)
    ok = (df.true == df.pred).to_numpy()
    y = df.true.to_numpy()
    ch = np.flatnonzero(y[1:] != y[:-1]) + 1
    idx = np.arange(len(df))
    dist = np.min(np.abs(idx[:, None] - ch[None, :]), 1) if len(ch) else np.full(len(df), 10 ** 6)
    hz = 1.0 / float(np.median(np.diff(df.t)))
    by_dist = []
    for lo, hi in [(0, 5), (5, 15), (15, 50), (50, 10 ** 9)]:
        m = (dist >= lo) & (dist < hi)
        by_dist.append({"frames_from_zone_change": f"{lo}-{hi if hi < 10 ** 9 else 'inf'}", "n": int(m.sum()),
                        "accuracy": round(float(ok[m].mean()), 4) if m.any() else None})
    pcols = [f"p_{c}" for c in classes]
    conf = df[pcols].to_numpy().max(1)
    by_conf = [{"min_conf": th, "share_of_frames": round(float((conf >= th).mean()), 4),
                "accuracy": round(float(ok[conf >= th].mean()), 4)} for th in (0.0, 0.5, 0.7, 0.9)]
    deg = (np.degrees(df.base_yaw.to_numpy()) // 45 * 45).astype(int)
    by_heading = {}
    for k, c in enumerate(classes):
        m = y == k
        by_heading[c] = {int(h): {"n": int((m & (deg == h)).sum()),
                                  "accuracy": round(float(ok[m & (deg == h)].mean()), 3)}
                         for h in sorted(set(deg[m]))}
    streaks, s = [], 0
    for v in ~ok:
        if v:
            s += 1
        elif s:
            streaks.append(s)
            s = 0
    if s:
        streaks.append(s)
    return {"frames": int(len(df)), "accuracy": round(float(ok.mean()), 4), "frame_rate_hz": round(hz, 2),
            "zone_changes": int(len(ch)), "accuracy_by_distance_to_zone_change": by_dist,
            "accuracy_by_confidence": by_conf,
            "median_conf_correct": round(float(np.median(conf[ok])), 4),
            "median_conf_wrong": round(float(np.median(conf[~ok])), 4),
            "accuracy_by_heading_45deg_bins": by_heading,
            "error_streaks": {"count": len(streaks), "median_frames": float(np.median(streaks)) if streaks else 0,
                              "longest_frames": int(max(streaks)) if streaks else 0,
                              "longest_s": round(max(streaks) / hz, 1) if streaks else 0}}


def sample_frames(sess, classes, split, path):
    train = [n for n in split.get("train", []) if n in sess and sess[n][2] is not None]
    if not train:
        return False
    import cv2
    fig, ax = plt.subplots(2, len(classes), figsize=(3.2 * len(classes), 5.0))
    rng = np.random.default_rng(6010)
    for k, c in enumerate(classes):
        cand = []
        for n in train:
            st, _, img = sess[n]
            for f in st.frame[st.zone == c]:
                if (img / f"{int(f):06d}.jpg").exists():
                    cand.append(img / f"{int(f):06d}.jpg")
        for i, p in enumerate(rng.choice(cand, 2, replace=False) if len(cand) >= 2 else cand):
            ax[i, k].imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            ax[i, k].set_xticks([]), ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(f"{k + 1}  {c}", color=ZC[k % len(ZC)], fontsize=12, fontweight="bold")
    fig.suptitle("Real Go2 head camera: training frames per zone (320 x 240 centred crop)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def test_session_plot(df, classes, path):
    df = df.sort_values("frame")
    ok = df.true == df.pred
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    for k, c in enumerate(classes):
        m = df.true == k
        ax[0].scatter(df.base_x[m], df.base_y[m], s=6, color=ZC[k % len(ZC)], label=c)
    ax[0].scatter(df.base_x[~ok], df.base_y[~ok], s=22, facecolors="none", edgecolors="black", lw=0.7,
                  label="misclassified")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("odometry x (m)"), ax[0].set_ylabel("odometry y (m)")
    ax[0].set_title("Test session path, coloured by the true zone", fontsize=10)
    ax[0].legend(fontsize=8, markerscale=1.5, loc="best")
    t = df.t - df.t.min()
    ax[1].plot(t, df.true, color="black", lw=2, label="true zone (operator key)")
    ax[1].scatter(t[ok], df.pred[ok], s=5, color="#1baf7a", label="prediction, correct")
    ax[1].scatter(t[~ok], df.pred[~ok], s=7, color="#d1495b", label="prediction, wrong")
    ax[1].set_yticks(range(len(classes)), classes)
    ax[1].set_xlabel("time in session (s)")
    ax[1].set_title(f"True vs predicted zone, frame by frame (accuracy {ok.mean():.3f})", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--data", help="folder with run_*/ (default: data_dir from the config)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    classes, out = cfg["classes"], TEST2 / cfg["results_dir"]
    met = json.loads((out / "metrics.json").read_text())
    df = pd.read_csv(out / "resnet18/test_predictions.csv")
    y, p = df.true.to_numpy(), df.pred.to_numpy()
    confusion_plot(y, p, classes, f"Real Go2, ResNet18: test acc {(y == p).mean():.3f}", out / "confusion_resnet18.png")
    curves_plot(json.loads((out / "resnet18/history.json").read_text()), "ResNet18 on real Go2 frames",
                out / "curves_resnet18.png")
    sess = sessions(Path(a.data) if a.data else TEST2 / cfg["data_dir"], out)
    summ = dataset_summary(sess, classes, met["split"])
    (out / "dataset_summary.json").write_text(json.dumps(summ, indent=1))
    ea = error_analysis(df, classes)
    (out / "error_analysis.json").write_text(json.dumps(ea, indent=1))
    test_session_plot(df, classes, out / "real_test_session.png")
    made = sample_frames(sess, classes, met["split"], out / "real_sample_frames.png")
    print(f"{summ['total_frames']} labelled frames in {len(sess)} sessions; test accuracy {ea['accuracy']} "
          f"(metrics.json {met['accuracy']}); sample frames {'written' if made else 'skipped (no images)'} -> {out}")


if __name__ == "__main__":
    main()
