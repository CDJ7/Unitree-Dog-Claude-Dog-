#!/usr/bin/env python3
"""Evaluate the BPA analysis against simulator ground truth (no MuJoCo needed).

    python evaluation/run_bpa_eval.py                 # 4 built-in + 30 random scenarios
    python evaluation/run_bpa_eval.py --random 80

Measures
  1. impact angle and direction error per stain (idealised segmentation,
     close-up resolution 0.25 mm per pixel)
  2. area-of-origin error per event (impact / projected), straight-line vs
     gravity-corrected, using the stains of that event
  3. camera path: angle error versus standoff distance for the simulator's
     front camera (320x240, 60 deg) and a close-up evidence camera
     (1920x1080, 40 deg), with the stain seen through the full
     mask + depth + pose pipeline
Writes PNGs and bpa_eval.json to evaluation/results/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from analysis.bpa.measure import from_camera_mask  # noqa: E402
from analysis.bpa.origin import MODEL_SD, SPEED_PRIORS, gravity_corrected_origin, straight_line_origin  # noqa: E402
from analysis.bpa.projection import Intrinsics, look_at  # noqa: E402
from analysis.bpa.synthetic import render_plane_view  # noqa: E402
from evaluation.observe import observe  # noqa: E402
from sim_env.scenarios import analytic_paths as scenario_paths  # noqa: E402
from sim_env.scenarios.random_scenario import random_scenario  # noqa: E402
from sim_env.spatter import generate, load_scenario  # noqa: E402
from sim_env.spatter.geometry import Rect  # noqa: E402

RESULTS = ROOT / "evaluation" / "results"


def wrap(d):
    return (d + 180) % 360 - 180


def eval_scenario(gt, rng, res=2.5e-4):
    obs, _ = observe(gt, res=res, drop=0.0, rng=rng, include_satellites=False)
    angle = [(o.meta["true_alpha"], o.alpha_deg, wrap(o.fit.gamma_deg - o.meta["true_gamma"]))
             for o in obs if o.usable]
    origins = []
    for ev in gt["events"]:
        if ev["mechanism"] not in ("impact", "projected"):
            continue
        ms = [o.measured() for o in obs if o.usable and o.meta["event"] == ev["index"]]
        if len(ms) < 6:
            continue
        # a projected source moves during the spurts: score against the centre of its path
        true = np.array(ev.get("origin_centroid") or ev["origin"])
        sl = straight_line_origin(ms, rng=rng)
        gc = gravity_corrected_origin(ms, start=sl.point, speed_prior=SPEED_PRIORS[ev["mechanism"]], rng=rng,
                                      model_sd=MODEL_SD[ev["mechanism"]])
        d = gc.point - true
        within = float(d @ np.linalg.solve(gc.cov, d)) <= 7.815
        origins.append({"scenario": gt["scenario_id"], "event": ev["index"], "mechanism": ev["mechanism"],
                        "n_stains": len(ms), "true": true.tolist(),
                        "straight": sl.point.tolist(), "gravity": gc.point.tolist(),
                        "err_straight": float(np.linalg.norm(sl.point - true)),
                        "err_gravity": float(np.linalg.norm(gc.point - true)),
                        "err_gravity_xy": float(np.linalg.norm(d[:2])), "err_gravity_z": float(d[2]),
                        "err_straight_z": float(sl.point[2] - true[2]),
                        "ellipsoid_95_max": float(gc.semi_axes[0]), "truth_inside_95": bool(within)})
    return angle, origins


def camera_curve(rng, n=25):
    wall = Rect.from_normal("w", "wall", (5, 2, 1.3), (-1, 0, 0), 2, 1.3)
    cams = {"sim front camera 320x240, 60 deg": Intrinsics.from_fovy(320, 240, 60),
            "evidence close-up 1920x1080, 40 deg": Intrinsics.from_fovy(1920, 1080, 40)}
    dists = [0.3, 0.45, 0.6, 0.8, 1.0, 1.5]
    out = {}
    for name, intr in cams.items():
        rows = []
        for dist in dists:
            errs, lost = [], 0
            for _ in range(n):
                a = rng.uniform(15, 60)
                g = rng.uniform(-180, 180)
                W = rng.uniform(2.5e-3, 4.5e-3)
                L = W / np.sin(np.radians(a))
                st = dict(center_uv=[0.0, 0.0], gamma_deg=g, length=L, width=W,
                          tail_length=L * 0.4 * np.cos(np.radians(a)))
                inc = rng.uniform(0, 40)
                target = wall.to_world([0, 0])
                cam = target + dist * (np.cos(np.radians(inc)) * wall.normal + np.sin(np.radians(inc)) * wall.u)
                T = look_at(cam, target)
                lab, dep = render_plane_view(wall, [st], intr, T)
                m = lab == 0
                if m.sum() < 6:
                    lost += 1
                    continue
                try:
                    o = from_camera_mask("x", m, dep, intr, T)
                except ValueError:
                    lost += 1
                    continue
                errs.append((abs(o.alpha_deg - a), abs(wrap(o.fit.gamma_deg - g))))
            e = np.array(errs) if errs else None
            rows.append({"distance": dist,
                         "alpha_abs_median": float(np.median(e[:, 0])) if errs else None,
                         "alpha_abs_p90": float(np.percentile(e[:, 0], 90)) if errs else None,
                         "direction_wrong_frac": float(np.mean(e[:, 1] > 90)) if errs else 1.0,
                         "lost_frac": lost / n})
        out[name] = rows
    return out


def plots(angles, origins, cam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = np.array(angles)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(A[:, 0], A[:, 1] - A[:, 0], s=4, alpha=0.4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("true impact angle (deg)"), ax[0].set_ylabel("measured - true (deg)")
    ax[0].set_title(f"Impact angle error, {len(A)} stains (median |err| {np.median(np.abs(A[:, 1] - A[:, 0])):.1f} deg)",
                    fontsize=9)
    ax[1].hist(np.abs(A[:, 2]), bins=np.linspace(0, 30, 31))
    ax[1].set_xlabel("|direction error| (deg)"), ax[1].set_ylabel("stains")
    ax[1].set_title(f"Direction (gamma) error, {np.mean(np.abs(A[:, 2]) > 90):.1%} reversed", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "angle_error.png", dpi=130)
    plt.close(fig)

    if origins:
        es = np.array([o["err_straight"] for o in origins])
        eg = np.array([o["err_gravity"] for o in origins])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        bins = np.linspace(0, max(es.max(), eg.max()) * 1.05, 20)
        ax[0].hist(es, bins=bins, alpha=0.6, label=f"straight line (median {np.median(es):.2f} m)")
        ax[0].hist(eg, bins=bins, alpha=0.6, label=f"gravity corrected (median {np.median(eg):.2f} m)")
        ax[0].set_xlabel("area-of-origin error (m)"), ax[0].set_ylabel("events")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"Origin error, {len(origins)} events", fontsize=9)
        zs = np.array([o["err_straight_z"] for o in origins])
        zg = np.array([o["err_gravity_z"] for o in origins])
        ax[1].scatter([o["true"][2] for o in origins], zs, s=14, label="straight line")
        ax[1].scatter([o["true"][2] for o in origins], zg, s=14, label="gravity corrected")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xlabel("true origin height (m)"), ax[1].set_ylabel("height error (m)")
        ax[1].set_title("Straight lines over-estimate height; gravity correction removes most of it", fontsize=9)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "origin_error.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, rows in cam.items():
        d = [r["distance"] for r in rows]
        ax[0].plot(d, [np.nan if r["alpha_abs_median"] is None else r["alpha_abs_median"] for r in rows], "o-",
                   label=name)
        ax[1].plot(d, [r["lost_frac"] + r["direction_wrong_frac"] * (1 - r["lost_frac"]) for r in rows], "o-",
                   label=name)
    ax[0].set_xlabel("camera standoff (m)"), ax[0].set_ylabel("median |impact angle error| (deg)")
    ax[0].set_title("Angle error through the camera pipeline (3 mm stains)", fontsize=9)
    ax[1].set_xlabel("camera standoff (m)"), ax[1].set_ylabel("fraction lost or direction reversed")
    ax[1].set_title("Stains that cannot be used", fontsize=9)
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "camera_standoff.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=30, help="random impact/projected scenarios to add")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    gts = [generate(load_scenario(p)) for p in scenario_paths()]
    for i in range(a.random):
        gts.append(generate(random_scenario(50000 + i, mechanism="impact" if i % 3 else "projected")))
    angles, origins = [], []
    for gt in gts:
        an, orr = eval_scenario(gt, rng)
        angles += an
        origins += orr
        print(f"{gt['scenario_id']}: {len(an)} usable stains, origins "
              + ", ".join(f"{o['mechanism']} straight {o['err_straight']:.2f} m / gravity {o['err_gravity']:.2f} m"
                          for o in orr))
    cam = camera_curve(rng)
    RESULTS.mkdir(parents=True, exist_ok=True)
    plots(angles, origins, cam)
    A = np.array(angles)
    es = np.array([o["err_straight"] for o in origins])
    eg = np.array([o["err_gravity"] for o in origins])
    summary = {
        "scenarios": len(gts),
        "stains": {"n": len(A), "alpha_abs_err_median_deg": float(np.median(np.abs(A[:, 1] - A[:, 0]))),
                   "alpha_abs_err_p90_deg": float(np.percentile(np.abs(A[:, 1] - A[:, 0]), 90)),
                   "gamma_abs_err_median_deg": float(np.median(np.abs(A[:, 2]))),
                   "direction_reversed_frac": float(np.mean(np.abs(A[:, 2]) > 90))},
        "origin": {"events": len(origins), "straight_median_m": float(np.median(es)),
                   "gravity_median_m": float(np.median(eg)), "gravity_p90_m": float(np.percentile(eg, 90)),
                   "gravity_xy_median_m": float(np.median([o["err_gravity_xy"] for o in origins])),
                   "truth_inside_95pct_ellipsoid": float(np.mean([o["truth_inside_95"] for o in origins]))},
        "camera_standoff": cam, "events": origins,
        "notes": ["Idealised segmentation (stains rasterised from ground truth at 0.25 mm/px for 1-2).",
                  "Origins use the stains of the true event (perfect grouping)."],
        "runtime_s": time.time() - t0,
    }
    (RESULTS / "bpa_eval.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k in ("stains", "origin")}, indent=1))
    print(f"wrote {RESULTS} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
