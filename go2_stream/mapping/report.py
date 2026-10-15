"""Write results/: object_map.json, top-down map plot, per-object error chart, accuracy CSV."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def write_all(out_dir, objmap, world, evaluation, meta, x_range=(-3.05, 3.05), y_range=(-2.55, 2.55)):
    os.makedirs(out_dir, exist_ok=True)
    doc = {"meta": meta, "objects": objmap.to_dict(), "robot_path": [[round(x, 3), round(y, 3)] for x, y in world.trail[::3]],
           "evaluation": evaluation}
    with open(os.path.join(out_dir, "object_map.json"), "w") as f:
        json.dump(doc, f, indent=2)

    with open(os.path.join(out_dir, "accuracy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "class", "detected", "n_observations", "true_x", "true_y", "true_z", "est_x", "est_y", "est_z",
                    "error_3d_m", "error_xy_m", "surface_error_3d_m"])
        for r in evaluation["per_object"]:
            est = r.get("est_xyz", [None] * 3)
            w.writerow([r["object"], r["class"], r["detected"], r.get("n_observations", 0), *r["true_xyz"], *est,
                        r.get("error_3d_m"), r.get("error_xy_m"), r.get("surface_error_3d_m")])

    _plot_map(os.path.join(out_dir, "map_topdown.png"), objmap, world, evaluation, meta, x_range, y_range)
    _plot_errors(os.path.join(out_dir, "error_per_object.png"), evaluation, meta)
    return doc


def _plot_map(out_path, objmap, world, ev, meta, xr, yr):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot([xr[0], xr[1], xr[1], xr[0], xr[0]], [yr[0], yr[0], yr[1], yr[1], yr[0]], "k-", lw=2, label="room outline")
    for o in world.objects:                                  # true footprints
        ax.add_patch(plt.Rectangle((o.aabb_min[0], o.aabb_min[1]), *(o.aabb_max[:2] - o.aabb_min[:2]),
                                   fc="#cfe8cf", ec="#5a9a5a", lw=1, zorder=1))
    path = np.array(world.trail)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "-", color="#ff8c00", lw=1.5, label="robot path", zorder=2)
        ax.plot(*path[0], "o", color="#ff8c00", ms=8, zorder=3)
    first_t = first_e = True
    for r in ev["per_object"]:
        t = r["true_xyz"]
        ax.plot(t[0], t[1], "s", color="#2e7d32", ms=9, zorder=4, label="true position" if first_t else None)
        first_t = False
        ax.annotate(r["object"].replace("obj_", ""), (t[0], t[1]), textcoords="offset points", xytext=(6, 6), fontsize=7)
        if r["detected"]:
            e = r["est_xyz"]
            ax.plot(e[0], e[1], "x", color="#c62828", ms=10, mew=2.5, zorder=5, label="estimated position" if first_e else None)
            first_e = False
            ax.plot([t[0], e[0]], [t[1], e[1]], "-", color="#c62828", lw=1, zorder=4)
    for i, fp in enumerate(ev["false_positives"]):
        ax.plot(fp["est_xyz"][0], fp["est_xyz"][1], "D", mfc="none", mec="#6a1b9a", ms=10, zorder=5,
                label="false positive" if i == 0 else None)
    s = ev["summary"]
    ax.set_title(f"Object map vs ground truth ({meta.get('labels', '?')})\n"
                 f"detected {s['n_detected']}/{s['n_true']}, mean 3D error {s['mean_error_3d_m']} m, "
                 f"false positives {s['n_false_positives']}", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _plot_errors(path, ev, meta):
    rows = ev["per_object"]
    names = [r["object"].replace("obj_", "") for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(rows))
    corr = [r.get("error_3d_m", 0) for r in rows]
    surf = [r.get("surface_error_3d_m", 0) for r in rows]
    ax.bar(x - 0.2, surf, 0.4, label="visible-surface point", color="#90a4ae")
    ax.bar(x + 0.2, corr, 0.4, label="with centre correction", color="#1565c0")
    for i, r in enumerate(rows):
        if not r["detected"]:
            ax.text(i, 0.01, "missed", ha="center", va="bottom", rotation=90, color="#c62828", fontsize=9)
        else:
            ax.text(i + 0.2, corr[i] + 0.005, f"{corr[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("3D position error (m)")
    ax.set_title(f"Position error per object ({meta.get('labels', '?')})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
