#!/usr/bin/env python3
"""Figures for REPORT.md that are not already produced by the Q1/Q2/Q3 scripts.

    python report/make_report_figures.py            # -> report/figures/*.png

Every value is read from a results file (no hard-coded numbers):
- system_overview.png   block diagram of how Q2, Q1 and Q3 connect
- q1_models.png         test accuracy and unseen-scene accuracy of the 5 Q1 models (results/q1/metrics.json)
- q3_runs.png           the four PPO runs: 2-D test success and MuJoCo success (results/q3/**)
- q3_sim_gap.png        PPO, planner and greedy: success in 2-D vs on the MuJoCo Go2 (deploy_eval_metrics.json)
- still_*.png           single frames of the GIFs, for the PDF version (GIFs do not animate in a PDF)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

TEST2 = Path(__file__).resolve().parents[1]
OUT = TEST2 / "report/figures"
R = TEST2 / "results"

# validated categorical slots 1-3 (light surface), neutral ink for all text
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})


def load(p):
    return json.loads((R / p).read_text())


def grouped_hbar(ax, labels, series, colors, names, xlabel, legend_loc="lower right"):
    """Horizontal grouped bars, 2 px surface gap between bars, value labels at the bar ends."""
    n = len(series)
    h = 0.8 / n
    for k, (vals, col, name) in enumerate(zip(series, colors, names)):
        ys = [i + (k - (n - 1) / 2) * h for i in range(len(labels))]
        bars = ax.barh(ys, [v if v is not None else 0 for v in vals], height=h, color=col, label=name,
                       edgecolor=SURF, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text((v or 0) + 0.012, b.get_y() + b.get_height() / 2, "n/a" if v is None else f"{v:.3f}",
                    va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9)


def q1_models():
    m = load("q1/metrics.json")["models"]
    keys = ["resnet18", "smallcnn", "hist_hsv_grid", "hist_hsv", "hist_rgb"]
    labels = ["ResNet18 (ImageNet,\nfine-tuned)", "Small CNN\n(from scratch)", "HSV histogram\n2x2 grid + LR",
              "HSV histogram + LR", "RGB histogram + LR"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grouped_hbar(ax, labels, [[m[k]["accuracy"] for k in keys], [m[k]["accuracy_unseen_scene"] for k in keys]],
                 [C1, C2], ["all 4,276 test frames", "unseen scene r1854 only"], "test accuracy")
    ax.set_title("Q1: zone classification accuracy on held-out test runs", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "q1_models.png")
    plt.close(fig)


def q3_runs():
    runs = [("v1: no route hint", "q3/v1_obs48", None), ("v2: + route hint\n(disc body)", "q3/v2_disc", "q3/v2_disc"),
            ("v3: + real body\nshape", "q3/v3_rect", "q3/v3_rect"), ("v4: + curriculum\n(reported)", "q3", "q3")]
    two_d, mj = [], []
    for _, ev, de in runs:
        two_d.append(load(f"{ev}/eval_metrics.json")["policies"]["ppo"]["success_rate"])
        mj.append(load(f"{de}/deploy_eval_metrics.json")["policies"]["ppo"]["mujoco"]["success_rate"] if de else None)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    grouped_hbar(ax, [r[0] for r in runs], [two_d, mj], [C1, C2],
                 ["2-D test (60 episodes, each run's own body model)", "MuJoCo Go2 (same 60 starts)"],
                 "PPO success rate (all evidence found)", legend_loc="upper right")
    ax.set_title("Q3: the four PPO runs", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "q3_runs.png")
    plt.close(fig)


def q3_sim_gap():
    d = load("q3/deploy_eval_metrics.json")["policies"]
    pols = [("PPO (v4)", "ppo"), ("planner", "planner"), ("greedy", "greedy")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    grouped_hbar(ax, [p[0] for p in pols], [[d[k]["2d"]["success_rate"] for _, k in pols],
                                            [d[k]["mujoco"]["success_rate"] for _, k in pols]],
                 [C1, C2], ["2-D training simulator", "MuJoCo Go2 (legged, real contacts)"], "success rate",
                 legend_loc="lower right")
    ax.set_title("Q3: the same 60 test starts in both simulators", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "q3_sim_gap.png")
    plt.close(fig)


def system_overview():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor="white", edgecolor=col, linewidth=1.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=8.2, color=INK2, linespacing=1.35)

    def arrow(a, b, text=None, dy=1.2):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=INK2, linewidth=1.3))
        if text:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", fontsize=7.8, color=INK2)

    box(1, 34, 24, 20, "MuJoCo crime scene", "house 185, Phase 2 simulator\nscenarios S1, S2 + random\nfurniture, mannequins,\nprops, stain clusters", INK2)
    box(38, 34, 24, 20, "Q2: Go2 + data", "learned locomotion policy\nset_velocity_command\nA* + pure-pursuit routes\nRGB 5 Hz, state 50 Hz", C1)
    box(75, 34, 24, 20, "Q1: zone classifier", "ResNet18 fine-tuned vs\nsmall CNN, RGB/HSV\nhistograms; Grad-CAM;\nlive demo in MuJoCo", C2)
    box(1, 3, 24, 20, "Exported layouts", "28 scenes: collision\nraster + evidence\nclusters", INK2)
    box(38, 3, 24, 20, "Q3: RL search", "2-D twin, Go2 body,\ncurriculum; PPO vs\nrandom / greedy /\nplanner baselines", C3)
    box(75, 3, 24, 20, "Deployment", "policy -> velocity\ncommands -> legged\nMuJoCo Go2; 60-start\nevaluation", C3)
    arrow((25.8, 44), (37.2, 44), "physics,\nscene", dy=1.6)
    arrow((62.8, 44), (74.2, 44), "17,123\nframes", dy=1.6)
    arrow((13, 33.2), (13, 23.8))
    ax.text(14.2, 28.5, "export", fontsize=7.8, color=INK2, va="center")
    arrow((25.8, 13), (37.2, 13), "walls,\nevidence", dy=1.6)
    arrow((62.8, 13), (74.2, 13), "trained\npolicy", dy=1.6)
    arrow((87, 23.8), (50, 33.2))
    ax.text(70, 27.2, "commands drive the Q2 Go2", fontsize=7.8, color=INK2, ha="center")
    fig.savefig(OUT / "system_overview.png", bbox_inches="tight")
    plt.close(fig)


def stills():
    frames = {"still_q2_walk.png": ("q2/go2_crime_scene.gif", 0.5), "still_q1_live.png": ("q1/demo_live_r1854.gif", 0.55),
              "still_q3_s2.png": ("q3/deploy_s2.gif", 1.0), "still_q3_r1854.png": ("q3/deploy_r1854.gif", 1.0),
              "still_q3_greedy_s2.png": ("q3/deploy_s2_greedy.gif", 1.0)}
    for name, (gif, where) in frames.items():
        im = Image.open(R / gif)
        im.seek(min(int(where * (im.n_frames - 1)), im.n_frames - 1))
        im.convert("RGB").save(OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    system_overview()
    q1_models()
    q3_runs()
    q3_sim_gap()
    stills()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
