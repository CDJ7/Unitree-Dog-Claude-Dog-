#!/usr/bin/env python3
"""Q2: walk the Go2 through the crime scene and record a labelled dataset.

    python q2_data_simulation/collect_dataset.py                       # full dataset (config/collect.yaml)
    python q2_data_simulation/collect_dataset.py --scenes s1 --routes clockwise_tour --output data/test
    python q2_data_simulation/collect_dataset.py --list                # print the run plan only

Output (one folder per run, layout modelled on unitree_go2_create_dataset's recorder):
    <output>/run_XXX/images/000000.jpg ...   front camera, 320x240, 5 Hz (camera noise applied)
    <output>/run_XXX/states.csv              50 Hz: time, base pose/velocity, IMU, 12 joints q/dq/tau,
                                             foot forces + contact flags, velocity command, zone label,
                                             collision count, `frame` (image index or -1)
    <output>/run_XXX/metadata.json           scene, route, seeds, lighting, camera noise, camera mount,
                                             controller, per-zone frame counts, events
    <output>/runs.csv                        one line per run (for splits and statistics)

One worker process per scene: the scene is loaded once, then each of its routes starts by
teleporting the standing dog to the route's start pose.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from go2sim import paths  # noqa: E402,F401
from go2sim.zones import Zones  # noqa: E402


def run_plan(cfg, scenes=None, routes=None):
    """Deterministic list of runs: dicts with run id, scene and route spec."""
    zones = Zones()
    plan, k = [], 0
    rng = np.random.default_rng(cfg["seed"])
    for si, sc in enumerate(cfg["scenes"]):
        items = [dict(r, kind="hand") for r in cfg["hand_routes"]]
        rr = cfg["random_route"]
        for j in range(cfg["random_routes_per_scene"]):
            n = int(rng.integers(rr["n_goals"][0], rr["n_goals"][1] + 1))
            # balanced: cycle through a shuffled zone list, never the same zone twice in a row
            order = []
            while len(order) < n:
                perm = list(rng.permutation(len(zones)))
                if order and perm[0] == order[-1]:
                    perm = perm[1:] + perm[:1]
                order += perm
            order = [zones.names[i] for i in order[:n]]
            start = zones.names[int(rng.integers(len(zones)))]
            items.append({"name": f"random_{j}", "kind": "random", "start": start, "order": order,
                          "speed": round(float(rng.uniform(*rr["speed"])), 2), "spins": "random",
                          "spin_prob": rr["spin_prob"]})
        for ri, r in enumerate(items):
            run = {"run": f"run_{k:03d}", "scene_index": si, "scene": sc["name"], "scene_spec": sc,
                   "route_index": ri, "route": r, "seed": int(cfg["seed"] * 1000 + k)}
            k += 1
            if (scenes and sc["name"] not in scenes) or (routes and r["name"] not in routes):
                continue
            plan.append(run)
    return plan


def _scenario(sc):
    if "random" in sc:
        from sim_env.scenarios.random_sim import write_random
        return str(write_random(int(sc["random"]), env="scenesmith_house_185"))
    return str(sc["scenario"])


STATE_COLS = None


def record_run(dog, smap, zones, run, cfg, out_root, zone_counts=None):
    """Execute one route with the recorder attached; returns the run summary."""
    from go2sim.randomization import apply_camera_noise, sample_camera_noise, sample_lighting
    from go2sim.waypoint_follower import WaypointFollower

    rng = np.random.default_rng(run["seed"])
    route = run["route"]
    out = Path(out_root) / run["run"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    lighting = sample_lighting(rng, cfg["lighting"])
    noise = sample_camera_noise(rng, cfg["camera_noise"])
    dog.set_lighting(**lighting)
    reach = smap.reachable_from(dog.pose2d()[:2])
    zi = {n: i for i, n in enumerate(zones.names)}
    start = smap.sample_free(rng, zones.polygons[zi[route["start"]]], cfg["min_clearance"], reach)
    if start is None:
        start = smap.sample_free(rng, None, cfg["min_clearance"], reach)
    yaw0 = float(rng.uniform(-np.pi, np.pi))
    dog.teleport(start[0], start[1], yaw0)

    rows, frames = [], []
    every = int(round((1.0 / cfg["image_hz"]) / 0.02))
    state_every = max(1, int(round((1.0 / cfg["state_hz"]) / 0.02)))
    tick = {"n": 0, "frame": 0, "goal": -1}
    t0 = dog.time

    def on_tick(d):
        n = tick["n"]
        tick["n"] += 1
        img = n % every == 0
        if not img and n % state_every:
            return
        row = d.read_state()
        row["t"] = round(d.time - t0, 4)
        z = zones.zone_id((row["base_x"], row["base_y"]))
        row["zone_id"], row["zone"] = z, zones.names[z] if z >= 0 else "outside"
        row["collisions"] = len(d.collisions())
        row["goal_index"] = tick["goal"]
        row["frame"] = -1
        if img:
            jpg = apply_camera_noise(d.render_front(tuple(cfg["image_size"])), noise, rng)
            name = f"{tick['frame']:06d}.jpg"
            (out / "images" / name).write_bytes(jpg)
            row["frame"] = tick["frame"]
            frames.append((tick["frame"], row["t"], z))
            tick["frame"] += 1
        rows.append(row)

    fol = WaypointFollower(dog, smap, on_tick=on_tick)
    goals, reached = [], 0
    fol.hold(0.4)
    for gi, zname in enumerate(route["order"]):
        tick["goal"] = gi
        g = smap.sample_free(rng, zones.polygons[zi[zname]], cfg["min_clearance"], reach)
        if g is None:
            fol.events.append((dog.time, "no_free_goal", zname))
            continue
        goals.append({"zone": zname, "xy": [round(float(v), 3) for v in g]})
        ok = fol.go_to(g, speed=route["speed"])
        reached += int(ok)
        spin = route["spins"] == "all" or (route["spins"] == "random" and rng.random() < route.get("spin_prob", 0.5))
        if not spin and cfg.get("balance_dwell") and zone_counts is not None:
            # adaptive balancing: also look around in zones that are under-represented so far
            seen = dict(zone_counts)
            for _, _, z in frames:
                if z >= 0:
                    seen[zones.names[z]] += 1
            spin = seen[zname] < 0.8 * np.mean(list(seen.values()))
        if spin:
            fol.spin(float(rng.choice([-1, 1]) * rng.uniform(np.pi, 2 * np.pi)))
    tick["goal"] = len(route["order"])
    fol.hold(0.5)

    keys = list(rows[0].keys())
    with open(out / "states.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    counts = {n: 0 for n in zones.names}
    for _, _, z in frames:
        if z >= 0:
            counts[zones.names[z]] += 1
    path = np.array([[r["base_x"], r["base_y"]] for r in rows])
    dist = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    col_ticks = int(sum(r["collisions"] > 0 for r in rows))
    meta = {
        "run": run["run"], "scene": run["scene"], "scenario_id": dog.scenario["id"],
        "scenario_title": dog.scenario.get("title", ""), "environment": dog.scenario["environment"],
        "route": route, "seed": run["seed"], "start_pose": [float(start[0]), float(start[1]), yaw0],
        "goals": goals, "goals_reached": reached, "lighting": lighting, "camera_noise": noise,
        "camera": {**dog.camera_mount, "image_size": cfg["image_size"], "image_hz": cfg["image_hz"],
                   "model": "pinhole, fx = fy = (H/2)/tan(fovy/2), cx = W/2, cy = H/2"},
        "controller": dog.controller_name, "state_hz": cfg["state_hz"],
        "duration_s": round(rows[-1]["t"], 2), "n_frames": len(frames), "n_states": len(rows),
        "frames_per_zone": counts, "distance_m": round(dist, 2),
        "collision_state_rows": col_ticks,
        "events": [[round(t, 2), k, v] for t, k, v in fol.events],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    return meta


def worker(args):
    scene_runs, cfg, out_root = args
    from go2sim.go2_env import CrimeSceneGo2
    from go2sim.static_map import StaticMap
    zones = Zones()
    sc = scene_runs[0]["scene_spec"]
    t = time.time()
    dog = CrimeSceneGo2(_scenario(sc), controller=cfg["controller"], camera_size=tuple(cfg["image_size"]))
    smap = StaticMap(dog.model, dog.data, dog.sim.robot_geoms)
    print(f"[{sc['name']}] loaded in {time.time() - t:.0f} s", flush=True)
    metas = []
    counts = {n: 0 for n in zones.names}
    try:
        for run in scene_runs:
            t = time.time()
            m = record_run(dog, smap, zones, run, cfg, out_root, counts)
            metas.append(m)
            for k, v in m["frames_per_zone"].items():
                counts[k] += v
            print(f"[{sc['name']}] {run['run']} {run['route']['name']}: {m['n_frames']} frames, "
                  f"{m['duration_s']} s sim, {m['goals_reached']}/{len(run['route']['order'])} goals, "
                  f"collision rows {m['collision_state_rows']}, {time.time() - t:.0f} s wall", flush=True)
    finally:
        dog.close()
    return metas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(TEST2 / "config/collect.yaml"))
    ap.add_argument("--output")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--routes", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out_root = Path(a.output or cfg["output_dir"])
    if not out_root.is_absolute():
        out_root = TEST2 / out_root
    plan = run_plan(cfg, a.scenes, a.routes)
    if a.list:
        for r in plan:
            print(r["run"], r["scene"], r["route"]["name"], r["route"]["start"], "->", r["route"]["order"],
                  r["route"]["speed"])
        return
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "collect_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    by_scene = {}
    for r in plan:
        by_scene.setdefault(r["scene"], []).append(r)
    jobs = [(runs, cfg, str(out_root)) for runs in by_scene.values()]
    t = time.time()
    if a.workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            metas = [m for ms in pool.imap_unordered(worker, jobs) for m in ms]
    else:
        metas = [m for j in jobs for m in worker(j)]
    # merge with runs already in the folder (collection can be done in parts)
    for p in sorted(out_root.glob("run_*/metadata.json")):
        m = json.loads(p.read_text())
        if m["run"] not in {x["run"] for x in metas}:
            metas.append(m)
    metas.sort(key=lambda m: m["run"])
    zones = Zones()
    with open(out_root / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "scene", "route", "speed", "n_frames", "duration_s", "distance_m", "goals_reached",
                    "n_goals", "collision_state_rows"] + zones.names)
        for m in metas:
            w.writerow([m["run"], m["scene"], m["route"]["name"], m["route"]["speed"], m["n_frames"],
                        m["duration_s"], m["distance_m"], m["goals_reached"], len(m["route"]["order"]),
                        m["collision_state_rows"]] + [m["frames_per_zone"][n] for n in zones.names])
    tot = {n: sum(m["frames_per_zone"][n] for m in metas) for n in zones.names}
    print(f"done: {len(metas)} runs, {sum(tot.values())} frames in {time.time() - t:.0f} s; per zone {tot}")


if __name__ == "__main__":
    main()
