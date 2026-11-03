#!/usr/bin/env python3
"""Q2: statistics and figures for a collected dataset.

    python q2_data_simulation/analyze.py                 # data/full -> results/q2/
    python q2_data_simulation/analyze.py --data data/sample --out results/q2/sample

Writes:
    dataset_stats.json         frames per zone / scene / route, durations, distances, collisions,
                               velocity-tracking error of the locomotion controller
    zone_counts.png            frames per zone (stacked by scene)
    trajectories.png           every run's base path over the house-185 top-down map + zone outlines
    sample_frames.png          4 frames per zone from different runs
    sensors_<run>.png          IMU, joint angles, foot-contact gait diagram, command vs measured velocity
The background map (house185_topdown.png + house185_map.npz) is rendered once from scenario S1
(`--make-map`, done automatically if missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

from go2sim.zones import Zones  # noqa: E402

EXTENT = (-0.5, 8.7, -0.5, 5.0)
MAP_DIR = TEST2 / "results" / "q2"


def make_map(out_dir=MAP_DIR):
    """Render the orthographic top view of S1 and save the static occupancy grid."""
    from go2sim.go2_env import CrimeSceneGo2
    from go2sim.static_map import StaticMap
    dog = CrimeSceneGo2("1", controller="scripted", settle_s=0.2)
    try:
        dog.teleport(-5, -5, 0, settle_s=0.0)          # move the dog out of the picture
        img = dog.render_topdown(EXTENT, px_per_m=100)
        Image.fromarray(img).save(out_dir / "house185_topdown.png")
        dog.teleport(5.0, 1.3, np.pi, settle_s=0.0)
        sm = StaticMap(dog.model, dog.data, dog.sim.robot_geoms)
        np.savez_compressed(out_dir / "house185_map.npz", blocked=sm.blocked, free=sm.free,
                            extent=np.array(sm.extent()))
    finally:
        dog.close()


def draw_map(ax, zones, alpha_zones=0.9, labels=True):
    bg = MAP_DIR / "house185_topdown.png"
    if bg.exists():
        ax.imshow(Image.open(bg), extent=EXTENT, origin="upper")
    for i, p in enumerate(zones.polygons):
        ax.add_patch(Polygon(p, fill=False, ec=zones.colors[i], lw=2, alpha=alpha_zones))
        if labels:
            c = zones.centroid(i)
            ax.text(c[0], c[1], zones.names[i], color="white", ha="center", va="center", fontsize=8,
                    bbox=dict(fc=zones.colors[i], ec="none", alpha=0.75, pad=1.5))
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def load_runs(data):
    runs = []
    for p in sorted(Path(data).glob("run_*/metadata.json")):
        m = json.loads(p.read_text())
        m["dir"] = p.parent
        runs.append(m)
    return runs


def tracking_error(df, lag_s=0.3):
    """RMS error between the command and the measured body velocity, after a fixed response lag."""
    dt = float(np.median(np.diff(df.t)))
    k = max(1, int(round(lag_s / dt)))
    cmd = df[["cmd_vx", "cmd_vy", "cmd_yaw_rate"]].to_numpy()[:-k]
    meas = df[["base_vx_body", "base_vy_body", "base_wz_body"]].to_numpy()[k:]
    moving = np.abs(cmd).sum(1) > 1e-3
    e = (meas - cmd)[moving]
    return {"rmse_vx": float(np.sqrt((e[:, 0] ** 2).mean())), "rmse_vy": float(np.sqrt((e[:, 1] ** 2).mean())),
            "rmse_yaw_rate": float(np.sqrt((e[:, 2] ** 2).mean())), "n": int(moving.sum())}


def sensor_plot(run, out):
    df = pd.read_csv(run["dir"] / "states.csv")
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for c, lab in (("imu_gyro_x", "x"), ("imu_gyro_y", "y"), ("imu_gyro_z", "z")):
        ax[0].plot(df.t, df[c], lw=0.8, label=lab)
    ax[0].set_ylabel("IMU gyro (rad/s)")
    ax[0].legend(loc="upper right", ncol=3)
    for c, lab in (("imu_acc_x", "x"), ("imu_acc_y", "y"), ("imu_acc_z", "z")):
        ax[1].plot(df.t, df[c], lw=0.8, label=lab)
    ax[1].set_ylabel("IMU accel (m/s²)")
    ax[1].legend(loc="upper right", ncol=3)
    for j in ("hip", "thigh", "calf"):
        ax[2].plot(df.t, df[f"q_FL_{j}"], lw=0.8, label=f"FL {j}")
        ax[2].plot(df.t, df[f"q_RR_{j}"], lw=0.8, ls="--", label=f"RR {j}")
    ax[2].set_ylabel("joint angle (rad)")
    ax[2].legend(loc="upper right", ncol=3, fontsize=7)
    for i, leg in enumerate(("FL", "FR", "RL", "RR")):
        c = df[f"foot_contact_{leg}"].to_numpy()
        ax[3].fill_between(df.t, i + 0.1, i + 0.1 + 0.8 * c, step="post", color=f"C{i}")
    ax[3].set_yticks([0.5, 1.5, 2.5, 3.5], ["FL", "FR", "RL", "RR"])
    ax[3].set_ylabel("foot contact")
    ax[4].plot(df.t, df.cmd_vx, "k--", lw=1, label="cmd vx")
    ax[4].plot(df.t, df.base_vx_body, "C0", lw=0.8, label="measured vx (body)")
    ax[4].plot(df.t, df.cmd_yaw_rate, "r--", lw=1, label="cmd yaw rate")
    ax[4].plot(df.t, df.base_wz_body, "C3", lw=0.8, label="measured yaw rate")
    ax[4].set_ylabel("velocity")
    ax[4].set_xlabel("time (s)")
    ax[4].legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle(f"{run['run']} ({run['scene']}, {run['route']['name']}): Go2 sensor log at 50 Hz")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    # zoom: 3 s of gait
    t0 = float(df.t[df.cmd_vx.abs() > 0.1].iloc[len(df) // 20]) if (df.cmd_vx.abs() > 0.1).any() else 5.0
    z = df[(df.t >= t0) & (df.t <= t0 + 3)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for j in ("hip", "thigh", "calf"):
        ax[0].plot(z.t, z[f"q_FL_{j}"], label=f"FL {j}")
    ax[0].set_ylabel("FL joint angle (rad)")
    ax[0].legend(ncol=3, fontsize=8)
    for i, leg in enumerate(("FL", "FR", "RL", "RR")):
        ax[1].fill_between(z.t, i + 0.1, i + 0.1 + 0.8 * z[f"foot_contact_{leg}"], step="post", color=f"C{i}")
    ax[1].set_yticks([0.5, 1.5, 2.5, 3.5], ["FL", "FR", "RL", "RR"])
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("diagonal pairs (FL+RR, FR+RL) alternate: the dog trots")
    fig.tight_layout()
    fig.savefig(str(out).replace(".png", "_gait_zoom.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(TEST2 / "data/full"))
    ap.add_argument("--out", default=str(TEST2 / "results/q2"))
    ap.add_argument("--make-map", action="store_true")
    ap.add_argument("--sensor-run", default=None, help="run id for the sensor plots (default: first)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.make_map or not (MAP_DIR / "house185_topdown.png").exists():
        make_map()
    zones = Zones()
    runs = load_runs(a.data)
    if not runs:
        raise SystemExit(f"no runs in {a.data}")
    scenes = sorted({r["scene"] for r in runs}, key=lambda s: [r["scene"] for r in runs].index(s))

    # ---------------------------------------------------------------- statistics
    per_zone = {n: int(sum(r["frames_per_zone"][n] for r in runs)) for n in zones.names}
    per_scene_zone = {s: {n: int(sum(r["frames_per_zone"][n] for r in runs if r["scene"] == s))
                          for n in zones.names} for s in scenes}
    track = []
    for r in runs:
        df = pd.read_csv(r["dir"] / "states.csv", usecols=["t", "cmd_vx", "cmd_vy", "cmd_yaw_rate", "base_vx_body",
                                                           "base_vy_body", "base_wz_body", "upright"])
        e = tracking_error(df)
        e["min_upright"] = float(df.upright.min())
        track.append(e)
    n = sum(e["n"] for e in track)
    total = sum(per_zone.values())
    stats = {
        "data": str(Path(a.data).resolve().relative_to(TEST2)) if str(Path(a.data).resolve()).startswith(str(TEST2)) else a.data,
        "n_runs": len(runs), "n_scenes": len(scenes), "n_frames": total,
        "n_state_rows": int(sum(r["n_states"] for r in runs)),
        "frames_per_zone": per_zone,
        "zone_share": {k: round(v / total, 4) for k, v in per_zone.items()},
        "balance_min_over_max": round(min(per_zone.values()) / max(per_zone.values()), 3),
        "frames_per_scene_zone": per_scene_zone,
        "sim_time_s": round(sum(r["duration_s"] for r in runs), 1),
        "distance_m": round(sum(r["distance_m"] for r in runs), 1),
        "goals_reached": int(sum(r["goals_reached"] for r in runs)),
        "goals_total": int(sum(len(r["route"]["order"]) for r in runs)),
        "collision_state_rows": int(sum(r["collision_state_rows"] for r in runs)),
        "runs_with_collisions": int(sum(r["collision_state_rows"] > 0 for r in runs)),
        "frames_outside_zones": int(sum(r["n_frames"] for r in runs) - total),
        "velocity_tracking_rmse_lag0.3s": {
            k: round(float(np.sqrt(sum(e[k] ** 2 * e["n"] for e in track) / n)), 4)
            for k in ("rmse_vx", "rmse_vy", "rmse_yaw_rate")},
        "min_upright_cos": round(min(e["min_upright"] for e in track), 4),
        "controller": runs[0]["controller"],
        "image_size": runs[0]["camera"]["image_size"], "image_hz": runs[0]["camera"]["image_hz"],
        "state_hz": runs[0]["state_hz"],
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in ("n_runs", "n_frames", "frames_per_zone", "balance_min_over_max",
                                            "goals_reached", "goals_total", "collision_state_rows",
                                            "velocity_tracking_rmse_lag0.3s", "min_upright_cos")}, indent=1))

    # ---------------------------------------------------------------- zone counts
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(zones))
    for i, s in enumerate(scenes):
        v = np.array([per_scene_zone[s][z] for z in zones.names])
        ax.bar(zones.names, v, bottom=bottom, label=s, color=plt.cm.tab20(i))
        bottom += v
    for i, v in enumerate(bottom):
        ax.text(i, v + total * 0.005, str(int(v)), ha="center", fontsize=9)
    ax.set_ylabel("frames")
    ax.set_title(f"Q2 dataset: {total} labelled frames, {len(runs)} runs, {len(scenes)} scenes")
    ax.legend(title="scene", fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(out / "zone_counts.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- trajectories
    fig, ax = plt.subplots(figsize=(11, 6.6))
    draw_map(ax, zones)
    for i, r in enumerate(runs):
        df = pd.read_csv(r["dir"] / "states.csv", usecols=["base_x", "base_y"])
        ax.plot(df.base_x, df.base_y, lw=0.8, alpha=0.75, color=plt.cm.tab20(scenes.index(r["scene"])))
        ax.plot(df.base_x.iloc[0], df.base_y.iloc[0], "o", ms=3, color="k")
    for i, s in enumerate(scenes):
        ax.plot([], [], color=plt.cm.tab20(i), label=s)
    ax.legend(title="scene", loc="lower right", fontsize=8)
    ax.set_title(f"Go2 base trajectories, {len(runs)} runs (dots = start poses)")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- sample frames
    rng = np.random.default_rng(0)
    per = 4
    fig, axes = plt.subplots(len(zones), per, figsize=(per * 2.6, len(zones) * 2.1))
    for zi, zname in enumerate(zones.names):
        cands = []
        for r in runs:
            df = pd.read_csv(r["dir"] / "states.csv", usecols=["frame", "zone_id"])
            f = df[(df.frame >= 0) & (df.zone_id == zi)].frame.to_numpy()
            if len(f):
                cands.append((r, f))
        picks = [cands[k] for k in rng.choice(len(cands), size=min(per, len(cands)), replace=False)]
        for j in range(per):
            ax = axes[zi, j]
            ax.axis("off")
            if j < len(picks):
                r, f = picks[j]
                fr = int(rng.choice(f))
                ax.imshow(Image.open(r["dir"] / "images" / f"{fr:06d}.jpg"))
                ax.set_title(f"{r['run']} #{fr}", fontsize=7)
        axes[zi, 0].text(-0.08, 0.5, zname, transform=axes[zi, 0].transAxes, rotation=90, va="center",
                         ha="right", fontsize=9, color=zones.colors[zi], weight="bold")
    fig.suptitle("Front-camera frames per zone (lighting and camera noise randomised per run)")
    fig.tight_layout()
    fig.savefig(out / "sample_frames.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- sensors
    run = next((r for r in runs if r["run"] == a.sensor_run), runs[0])
    sensor_plot(run, out / f"sensors_{run['run']}.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
