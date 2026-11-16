#!/usr/bin/env python3
"""Where are the training pictures thin? Coverage of the lab by position AND viewing direction, and a plan for the
next recording session. No robot needed.

    python real_go2/coverage.py                 # -> results/real_q1/coverage.json, coverage.png, recording_plan.md
    python real_go2/coverage.py --spots 10      # suggest more survey spots

Why: to know where it is from the camera, the classifier needs pictures of each part of the lab from many
directions. Driving tours give lots of frames looking ALONG the route and few looking across it, so a zone seen
from an unusual direction is easily mistaken for another (the held-out session confused open_area with lab_2
83 times out of 465). This script:
  1. puts every recorded frame on the zone map (results/real_q1/zone_map.npz; each session's odometry is
     rotated + shifted into the map frame by the transforms saved there) with the direction the camera faced;
  2. counts frames per 0.5 m cell and per 45 deg viewing sector (8 sectors);
  3. scores each walkable cell by its missing sectors, weighted up in zones the classifier confuses
     (results/real_q1/metrics.json) and near zone borders (where mistakes happen);
  4. picks the best SURVEY SPOTS (at least --min-sep apart) and writes recording_plan.md + coverage.png with the
     spots numbered on the map. At each spot: drive there, press its zone number, press T in
     record_real.py (one slow full turn on the spot, recorded).
Map frame = the odometry frame of the reference session (the one with the most frames, see zone_map.json info):
the PNG is the thing to read in the lab.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.zone_map import ZoneMap, apply, load_runs, map_cfg  # noqa: E402

SECTORS = 8
SECTOR_NAMES = ["+x", "+x+y", "+y", "-x+y", "-x", "-x-y", "-y", "+x-y"]


def frame_poses(zm, runs):
    """(xy (N,2), heading (N,), zone (N,), run name per frame) in the map frame, for the runs the map knows."""
    tfs = (zm.info or {}).get("transforms", {})
    xs, hs, zs, names = [], [], [], []
    for n, (xy, yaw, lab, _) in runs.items():
        if n not in tfs:
            continue
        th, t = math.radians(tfs[n]["rotation_deg"]), np.array(tfs[n]["shift_m"], float)
        xs.append(apply(xy, th, t))
        hs.append((yaw + th) % (2 * math.pi))
        zs.append(lab)
        names += [n] * len(xy)
    if not xs:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int), []
    return np.concatenate(xs), np.concatenate(hs), np.concatenate(zs), names


def sector_of(heading):
    return (np.floor((np.asarray(heading) % (2 * math.pi)) / (2 * math.pi) * SECTORS + 0.5).astype(int)) % SECTORS


class Coverage:
    """Frame counts per coarse cell and viewing sector over the map's walkable area."""

    def __init__(self, zm, xy, heading, cell_m=0.5):
        self.zm, self.cell_m = zm, float(cell_m)
        self.lo = zm.lo
        ext = np.array(zm.shape) * zm.res
        self.shape = tuple(np.ceil(ext / cell_m).astype(int))
        self.counts = np.zeros(self.shape + (SECTORS,), int)
        if len(xy):
            ij = np.floor((xy - self.lo) / cell_m).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
            np.add.at(self.counts, (ij[ok, 0], ij[ok, 1], sector_of(heading)[ok]), 1)
        # a coarse cell is walkable if most of its fine map cells are
        f = max(1, int(round(cell_m / zm.res)))
        w = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                blk = zm.walk[i * f:(i + 1) * f, j * f:(j + 1) * f]
                w[i, j] = blk.mean() if blk.size else 0.0
        self.walk = w >= 0.5

    def centre(self, ij):
        return self.lo + (np.asarray(ij, float) + 0.5) * self.cell_m

    def sectors_covered(self, min_frames):
        return (self.counts >= min_frames).sum(2)

    def summary(self, min_frames, classes):
        cov = self.sectors_covered(min_frames)
        w = self.walk
        zone = np.array([[self.zm.zone_at(self.centre((i, j))) for j in range(self.shape[1])]
                         for i in range(self.shape[0])])
        per_zone = {}
        for k, c in enumerate(classes):
            m = w & (zone == k)
            per_zone[c] = {"cells": int(m.sum()),
                           "mean_sectors_covered": round(float(cov[m].mean()), 2) if m.any() else None,
                           "cells_with_all_sectors": int((cov[m] == SECTORS).sum()),
                           "frames": int(self.counts[m].sum())}
        return {"cells": int(w.sum()), "min_frames_per_sector": int(min_frames),
                "mean_sectors_covered": round(float(cov[w].mean()), 2) if w.any() else 0.0,
                "share_cells_all_sectors": round(float((cov[w] == SECTORS).mean()), 3) if w.any() else 0.0,
                "per_zone": per_zone}, zone


def zone_weights(classes, metrics):
    """1 + the share of a zone's test frames that were misclassified (more weight where the classifier is weak)."""
    w = np.ones(len(classes))
    if metrics and "confusion" in metrics and metrics.get("classes") == list(classes):
        cm = np.array(metrics["confusion"], float)
        w = 1.0 + (1.0 - np.diag(cm) / np.maximum(cm.sum(1), 1))
    return w


def confusions(classes, metrics, top=3):
    """The largest off-diagonal confusions: (true zone, predicted zone, count, share of the true zone)."""
    if not metrics or "confusion" not in metrics or metrics.get("classes") != list(classes):
        return []
    cm = np.array(metrics["confusion"], float)
    out = [(classes[i], classes[j], int(cm[i, j]), float(cm[i, j] / max(cm[i].sum(), 1)))
           for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    return sorted(out, key=lambda r: -r[2])[:top]


def plan_spots(cov, zone, weights, min_frames, n_spots=8, min_sep=1.2):
    """Greedy survey spots: score = missing sectors x zone weight x (1.5 near a zone border); spots at least
    min_sep apart. Returns a list of dicts (map xy, zone index, missing sector indices, score)."""
    covered = cov.counts >= min_frames
    missing = SECTORS - covered.sum(2)
    border = np.zeros(cov.shape, bool)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        border |= zone != np.roll(np.roll(zone, di, 0), dj, 1)
    score = missing * weights[zone] * np.where(border, 1.5, 1.0) * cov.walk
    # don't send the dog to the very edge of where it has walked
    f = max(1, int(round(cov.cell_m / cov.zm.res)))
    clear = np.array([[cov.zm.clear[min(i * f + f // 2, cov.zm.shape[0] - 1), min(j * f + f // 2, cov.zm.shape[1] - 1)]
                       for j in range(cov.shape[1])] for i in range(cov.shape[0])])
    score = score * (clear >= 0.3)
    spots = []
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    for i, j in order:
        if score[i, j] <= 0 or len(spots) >= n_spots:
            break
        c = cov.centre((i, j))
        if any(np.hypot(*(c - s["xy"])) < min_sep for s in spots):
            continue
        spots.append({"xy": c, "zone": int(zone[i, j]), "missing": [int(s) for s in np.flatnonzero(~covered[i, j])],
                      "frames": int(cov.counts[i, j].sum()), "score": float(score[i, j])})
    return spots


def plot(cov, zone, classes, spots, path, min_frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a7de1", "#e8602c", "#1bb27f", "#f0a30a", "#8e44ad", "#7f8c8d"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ext = [cov.lo[0], cov.lo[0] + cov.shape[0] * cov.cell_m, cov.lo[1], cov.lo[1] + cov.shape[1] * cov.cell_m]
    rgb = np.ones(cov.shape + (3,))
    for k in range(len(classes)):
        rgb[cov.walk & (zone == k)] = 0.4 * np.array(matplotlib.colors.to_rgb(cols[k % len(cols)])) + 0.6
    axes[0].imshow(rgb.transpose(1, 0, 2), origin="lower", extent=ext)
    for k, c in enumerate(classes):
        axes[0].plot([], [], "s", color=cols[k % len(cols)], label=c)
    for n, s in enumerate(spots, 1):
        axes[0].plot(*s["xy"], "o", ms=14, mfc="white", mec="black")
        axes[0].text(*s["xy"], str(n), ha="center", va="center", fontsize=9, weight="bold")
    axes[0].set_title("zones (map) and suggested survey spots", fontsize=10)
    axes[0].legend(fontsize=8, loc="best")
    cv = np.where(cov.walk, cov.sectors_covered(min_frames), np.nan)
    im = axes[1].imshow(cv.T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=SECTORS)
    fig.colorbar(im, ax=axes[1], label=f"viewing directions with >= {min_frames} frames (of {SECTORS})")
    for n, s in enumerate(spots, 1):
        axes[1].text(*s["xy"], str(n), ha="center", va="center", fontsize=9, weight="bold", color="white")
    axes[1].set_title("how many directions each 0.5 m cell was seen from", fontsize=10)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("map x (m)"), ax.set_ylabel("map y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_plan(path, classes, summary, spots, conf, ref):
    L = ["# Recording plan: where the training pictures are thin", "",
         "Made by `python real_go2/coverage.py` from the recorded sessions and the zone map. Read it with "
         "`coverage.png` open: the numbers there are the spots below.", "",
         f"- Walkable 0.5 m cells: {summary['cells']}; on average each was seen from "
         f"{summary['mean_sectors_covered']} of {SECTORS} directions (>= {summary['min_frames_per_sector']} frames each).",
         f"- Cells seen from all {SECTORS} directions: {summary['share_cells_all_sectors']:.0%}.", ""]
    L += ["| zone | cells | directions covered (mean) | cells with all directions | frames |", "|---|---|---|---|---|"]
    for c in classes:
        z = summary["per_zone"][c]
        L.append(f"| {c} | {z['cells']} | {z['mean_sectors_covered']} | {z['cells_with_all_sectors']} | {z['frames']} |")
    if conf:
        L += ["", "## What the classifier confuses (held-out session)", ""]
        for t, p, n, share in conf:
            L.append(f"- **{t}** taken for **{p}**: {n} frames ({share:.0%} of {t}). Record {t} from more directions, "
                     f"especially looking towards {p}, and {p} from where it looks most like {t}.")
    L += ["", "## Survey spots (in order of value)", "",
          f"Map frame: the odometry frame of session `{ref}` (x = the way the dog faced when that session started).",
          "At each spot: drive there, press the zone's number, then press **T** in `record_real.py`: the dog turns "
          "once on the spot, slowly, recording all the way round. Any driving key stops the turn.", "",
          "| spot | zone | map x, y (m) | directions missing | frames there now |", "|---|---|---|---|---|"]
    for n, s in enumerate(spots, 1):
        miss = ", ".join(SECTOR_NAMES[m] for m in s["missing"]) or "-"
        L.append(f"| {n} | {classes[s['zone']]} | {s['xy'][0]:.1f}, {s['xy'][1]:.1f} | {miss} | {s['frames']} |")
    L += ["", "Then rebuild and retrain:", "", "```bash", "python real_go2/zone_map.py",
          "python q1_supervised/train.py --config real_go2/real_config.yaml", "python real_go2/evaluate_real.py",
          "python real_go2/coverage.py      # check the gaps closed", "```", ""]
    Path(path).write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--min-frames", type=int, default=5, help="frames needed for a viewing sector to count")
    ap.add_argument("--spots", type=int, default=8)
    ap.add_argument("--min-sep", type=float, default=1.2, help="survey spots at least this far apart (m)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    classes, out = cfg["classes"], TEST2 / cfg["results_dir"]
    mpath = TEST2 / map_cfg(cfg["navigation"])["file"]
    if not mpath.exists():
        raise SystemExit(f"No zone map at {mpath}: record sessions, then python real_go2/zone_map.py")
    zm = ZoneMap.load(mpath)
    runs = load_runs(TEST2 / cfg["data_dir"], classes)
    if not runs:
        raise SystemExit(f"No recordings in {TEST2 / cfg['data_dir']} (data_real/ is not in git: copy it here)")
    xy, hd, zs, names = frame_poses(zm, runs)
    missing_runs = sorted(set(runs) - set((zm.info or {}).get("transforms", {})))
    cov = Coverage(zm, xy, hd)
    summary, zone = cov.summary(a.min_frames, classes)
    mp = out / "metrics.json"
    metrics = json.loads(mp.read_text()) if mp.exists() else None
    spots = plan_spots(cov, zone, zone_weights(classes, metrics), a.min_frames, a.spots, a.min_sep)
    conf = confusions(classes, metrics)
    plot(cov, zone, classes, spots, out / "coverage.png", a.min_frames)
    write_plan(out / "recording_plan.md", classes, summary, spots, conf, (zm.info or {}).get("reference", "?"))
    res = {"frames_on_map": int(len(xy)), "runs_on_map": sorted(set(names)), "runs_not_in_map": missing_runs,
           **summary, "confusions": [list(c) for c in conf],
           "spots": [{**s, "xy": [round(float(v), 2) for v in s["xy"]], "zone": classes[s["zone"]]} for s in spots]}
    (out / "coverage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("frames_on_map", "mean_sectors_covered", "share_cells_all_sectors")}))
    if missing_runs:
        print(f"Not in the map yet (rebuild with python real_go2/zone_map.py): {', '.join(missing_runs)}")
    print(f"-> {(out / 'recording_plan.md').relative_to(TEST2)}, {(out / 'coverage.png').relative_to(TEST2)}")


if __name__ == "__main__":
    main()
