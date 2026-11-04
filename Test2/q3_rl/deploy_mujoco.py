#!/usr/bin/env python3
"""Q3: deploy the trained search policy on the full MuJoCo Go2 (learned locomotion controller from Q2).

    python q3_rl/deploy_mujoco.py                          # test layout s2, results/q3/ppo/best_model.zip
    python q3_rl/deploy_mujoco.py --layout r1854 --seed 3 --policy greedy

The 2-D env is used only as the policy's *sensor model*: every 0.2 s the MuJoCo dog's true pose and
measured body velocity are written into it, the observation (lidar on the scene's collision raster,
nearest unfound evidence, found flags...) is built from them, and the policy's action is sent to
CrimeSceneGo2.set_velocity_command - the same (vx, vy, yaw_rate) interface as in Q2. The legs are
driven by the simulator's learned locomotion policy; nothing about the RL policy is changed.
Evidence is marked found with the same camera-view rule as in training. Contamination is measured
physically: steps where a FOOT of the MuJoCo dog lands inside a floor stain cluster (plus the env's
footprint rule for comparison). Collisions are MuJoCo contacts between the robot and the scene.
Writes results/q3/deploy_<layout>.gif (chase camera | map with trail), deploy_<layout>.json and .csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

import go2sim.paths  # noqa: E402,F401
from PIL import Image, ImageDraw  # noqa: E402

from q3_rl.crime_scene_env import CrimeSceneSearchEnv, load_env_config  # noqa: E402
from q3_rl.evaluate import PPOPolicy, _rel, greedy  # noqa: F401  # noqa: E402
from q3_rl.extract_layouts import scenario_spec  # noqa: E402

OUT = TEST2 / "results/q3"


def run_on_dog(dog, env, policy, max_s, on_step=None):
    """Run one episode of `policy` on the MuJoCo dog, starting from its current pose.

    Every control step (env dt) the dog's true pose and measured body velocity are written into the 2-D
    env (its sensor model); evidence found, footprint-on-stain, feet inside floor stains and robot-scene
    contacts are measured; then the policy's action is sent to set_velocity_command. `on_step(k, row,
    stats)` is called before each action (used for the GIF). Returns (per-step rows, stats dict)."""
    L, dt = env.layout, env.cfg["robot"]["dt"]
    floor = L.floor
    rows = []
    st_ = {"feet_on": 0, "coll_steps": 0, "entries": 0}
    was_on = -1
    t0 = dog.time
    k = 0
    while True:
        x, y, yaw = dog.pose2d()
        st = dog.read_state()
        env.xy, env.yaw = np.array([x, y]), float(yaw)
        env.v = np.array([st["base_vx_body"], st["base_vy_body"], st["base_wz_body"]])
        env.found |= env._visible()
        on = env._on_stain()
        st_["entries"] += int(on >= 0 and on != was_on)
        was_on = on
        feet = np.array([dog.data.geom_xpos[g][:2] for g in dog.foot_geom.values()])
        foot_in = bool(len(floor)) and bool((np.linalg.norm(feet[:, None] - floor[None, :, :2], axis=2)
                                              < floor[None, :, 2]).any())
        st_["feet_on"] += int(foot_in)
        n_coll = len(dog.collisions())
        st_["coll_steps"] += int(n_coll > 0)
        t = dog.time - t0
        row = {"t": round(t, 2), "x": x, "y": y, "yaw": yaw, "found": int(env.found.sum()),
               "foot_in_stain": int(foot_in), "footprint_on_stain": int(on >= 0), "contacts": n_coll,
               "upright": st["upright"]}
        rows.append(row)
        done = bool(env.found.all())
        if on_step:
            on_step(k, row, st_, done)
        if done or t >= max_s:
            break
        env.last_collided = n_coll > 0          # MuJoCo contact = the env's "bumped last step" input
        act = np.clip(np.asarray(policy(env._obs()), float), -1, 1)
        env.last_action = act                    # the env's "previous action" input (as in training)
        dog.set_velocity_command(*(act * env.limits))
        dog.step(dt)
        k += 1
    df = pd.DataFrame(rows)
    stats = {"n_items": int(len(env.found)), "n_found": int(env.found.sum()), "success": bool(env.found.all()),
             "time_s": round(float(df.t.iloc[-1]), 2), "control_steps": int(len(df) - 1),
             "steps_with_foot_in_stain": int(st_["feet_on"]), "footprint_stain_entries": int(st_["entries"]),
             "steps_with_robot_scene_contact": int(st_["coll_steps"]), "min_upright": round(float(df.upright.min()), 3)}
    return df, stats


def make_policy(name, model=None):
    from q3_rl.evaluate import planner
    return {"ppo": lambda: PPOPolicy(model), "greedy": lambda: greedy, "planner": lambda: planner}[name]()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="s2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", default="ppo", choices=["ppo", "greedy", "planner"])
    ap.add_argument("--out", default=str(OUT), help="output folder (v3: results/q3/v3_rect)")
    ap.add_argument("--model", help="default: <out>/ppo/best_model.zip")
    ap.add_argument("--env-config", help="environment config (default config/q3_env.yaml)")
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--gif-every", type=int, default=2, help="render every k-th control step")
    a = ap.parse_args()
    from go2sim.go2_env import CrimeSceneGo2
    out = Path(a.out)
    a.model = a.model or str(out / "ppo/best_model.zip")
    cfg = load_env_config(a.env_config)
    env = CrimeSceneSearchEnv(layouts=[a.layout], config=cfg, randomize=False, render_mode="rgb_array")
    env.reset(seed=a.seed, options={"layout": 0})
    x0, y0, yaw0 = env.xy[0], env.xy[1], env.yaw
    policy = make_policy(a.policy, a.model)
    dog = CrimeSceneGo2(scenario_spec(a.layout))
    dog.set_lighting(headlight_ambient=0.8, headlight_diffuse=0.9, lamps=(0.3, 0.2, 0.3))
    dog.teleport(x0, y0, yaw0)
    L = env.layout
    dt = cfg["robot"]["dt"]
    max_s = a.max_s or cfg["episode"]["max_steps"] * dt
    frames, trail = [], []
    wall0 = time.time()
    S = 3  # env.render scale

    def to_px(x, y):
        return ((x - L.origin[0]) / L.res * S, (L.H - (y - L.origin[1]) / L.res) * S)

    def gif_frame(k, row, st, done):
        trail.append(to_px(row["x"], row["y"]))
        if k % a.gif_every and not done:
            return
        chase = Image.fromarray(dog.render_chase((480, 360)))
        m = Image.fromarray(env.render())
        d = ImageDraw.Draw(m)
        if len(trail) > 1:
            d.line(trail, fill=(30, 90, 220), width=2)
        m = m.resize((int(m.width * 360 / m.height), 360))
        canvas = Image.new("RGB", (480 + m.width, 392), (20, 20, 20))
        canvas.paste(chase, (0, 32))
        canvas.paste(m, (480, 32))
        d = ImageDraw.Draw(canvas)
        d.text((6, 4), f"{a.policy.upper()} policy on MuJoCo Go2 | {a.layout} | t={row['t']:5.1f}s | "
               f"evidence found {row['found']}/{len(env.found)} | foot-in-stain steps {st['feet_on']} | "
               f"contact steps {st['coll_steps']}", fill=(255, 255, 255))
        frames.append(canvas)

    try:
        df, stats = run_on_dog(dog, env, policy, max_s, on_step=gif_frame)
    finally:
        dog.close()
    res = {"layout": a.layout, "policy": a.policy, "model": _rel(policy.path) if a.policy == "ppo" else None,
           "seed": a.seed, "spawn": [round(float(x0), 3), round(float(y0), 3), round(float(yaw0), 3)],
           **stats, "wall_time_s": round(time.time() - wall0, 1)}
    name = f"deploy_{a.layout}" + ("" if a.policy == "ppo" else f"_{a.policy}")
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    pal = frames[len(frames) // 2].quantize(colors=96, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out / f"{name}.gif", save_all=True, append_images=q[1:] + [q[-1]] * 8,
              duration=int(1000 * dt * a.gif_every / 2), loop=0, optimize=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
