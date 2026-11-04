#!/usr/bin/env python3
"""Q3: evaluate the PPO policy against random and greedy baselines on the held-out TEST layouts.

    python q3_rl/evaluate.py                          # results/q3/ppo/best_model.zip, 60 episodes
    python q3_rl/evaluate.py --episodes 120 --model results/q3/ppo/final_model.zip

Every policy is run on exactly the same episodes (same layout, spawn, evidence subset and domain
randomisation: episode k uses reset(seed=BASE+k)), split evenly over the 6 test layouts that were
never used in training or model selection.
Policies:
  random  uniform random velocity commands
  greedy  reactive: of the lidar directions open for > 0.8 m (+-30 deg window), steer along the one
          closest to the bearing of the nearest unfound evidence; drive when aligned and the
          robot-wide corridor ahead is clear (no map memory, no stain avoidance) - uses only the
          same observation vector as PPO
  planner map-following: steer along the observation's route hint (shortest path on the known floor
          plan to the nearest unfound evidence) with the same corridor check; no stain avoidance
  ppo     the trained policy (deterministic)
Writes results/q3/eval_episodes.csv, eval_metrics.json, comparison.md, trajectories.png,
learning_curves.png (from the EvalCallback log and the training Monitor files).
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

from q3_rl.crime_scene_env import CrimeSceneSearchEnv, load_env_config  # noqa: E402

BASE_SEED = 900_000
OUT = TEST2 / "results/q3"


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, 3)


RAY_ANGLES = np.arange(24) * 2 * np.pi / 24     # lidar ray directions in the body frame (0 = ahead, CCW)


def greedy(obs, open_m=0.8, front_m=0.35, half_width=0.30):
    """Reactive go-to-nearest-evidence (see module docstring). A direction counts as open when the
    lidar rays within +-30 deg of it are longer than `open_m` (or than the evidence distance); the
    dog turns to the open direction closest to the evidence bearing, and drives forward only when
    roughly aligned and no lidar return lies inside the robot-wide corridor ahead (`front_m` deep,
    +-`half_width`). Uses only the PPO observation vector; no map memory, no stain avoidance."""
    lid = obs[7:31] * 3.0
    dx, dy = obs[31] * 5.0, obs[32] * 5.0
    bearing = np.arctan2(dy, dx)
    win = np.min(np.stack([np.roll(lid, k) for k in (-2, -1, 0, 1, 2)]), 0)
    ok = win > min(open_m, np.hypot(dx, dy))
    if not ok.any():
        return np.array([0.0, 0.0, 1.0])
    score = np.where(ok, np.cos(RAY_ANGLES - bearing), -np.inf)
    e = (RAY_ANGLES[int(np.argmax(score))] + np.pi) % (2 * np.pi) - np.pi
    ahead = np.cos(RAY_ANGLES) > 0
    px, py = lid * np.cos(RAY_ANGLES), lid * np.sin(RAY_ANGLES)
    blocked = (ahead & (px < front_m + 0.24) & (np.abs(py) < half_width)).any()
    vx = np.cos(e) ** 2 if (abs(e) < 0.6 and not blocked) else 0.0
    return np.array([vx, 0.0, np.clip(2.0 * e, -1, 1)])


def planner(obs, front_m=0.10, half_width=0.26):
    """Map-following baseline: steer along the route hint (shortest path on the known floor plan to
    the nearest unfound evidence, observation 48-49), drive when aligned and the robot-wide corridor
    ahead is clear. Knows the map like PPO, but nothing about stains."""
    lid = obs[7:31] * 3.0
    e = float(np.arctan2(obs[49], obs[48])) if np.any(obs[48:50]) else float(np.arctan2(obs[32], obs[31]))
    px, py = lid * np.cos(RAY_ANGLES), lid * np.sin(RAY_ANGLES)
    blocked = ((np.cos(RAY_ANGLES) > 0) & (px < front_m + 0.24) & (np.abs(py) < half_width)).any()
    vx = np.cos(e) ** 2 if (abs(e) < 0.6 and not blocked) else 0.0
    return np.array([vx, 0.0, np.clip(2.0 * e, -1, 1)])


class PPOPolicy:
    def __init__(self, path):
        from stable_baselines3 import PPO
        path = Path(path)
        if not path.exists() and path.name == "best_model.zip" and (path.parent / "final_model.zip").exists():
            # the EvalCallback writes best_model.zip only after its first evaluation (eval_every steps)
            print(f"[ppo] {path} not written yet (training shorter than eval_every); using final_model.zip", flush=True)
            path = path.parent / "final_model.zip"
        self.path = path
        self.model = PPO.load(path, device="cpu")

    def __call__(self, obs):
        return self.model.predict(obs, deterministic=True)[0]


def run_episode(env, policy, seed, layout):
    obs, info = env.reset(seed=seed, options={"layout": layout})
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        ret += r
        done = term or trunc
    return {**{k: v for k, v in info.items() if k not in ("reward_terms",)}, "return": ret,
            "traj": np.array(env.traj), "targets": env.tgt.copy(), "found": env.found.copy()}


def _rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(TEST2)) if p.is_relative_to(TEST2) else str(p)


def summary(df):
    s = df[df.success]
    return {"episodes": int(len(df)), "success_rate": round(float(df.success.mean()), 4),
            "found_fraction": round(float((df.n_found / df.n_items).mean()), 4),
            "time_s_successful_mean": round(float(s.time_s.mean()), 2) if len(s) else None,
            "time_s_all_mean": round(float(df.time_s.mean()), 2),
            "contamination_entries_mean": round(float(df.contamination_entries.mean()), 3),
            "contamination_steps_mean": round(float(df.contamination_steps.mean()), 2),
            "episodes_with_contamination": round(float((df.contamination_entries > 0).mean()), 4),
            "collisions_mean": round(float(df.collisions.mean()), 2),
            "severe_collision_rate": round(float(df.severe_collision.mean()), 4),
            "path_m_mean": round(float(df.path_m.mean()), 2), "return_mean": round(float(df["return"].mean()), 2)}


MONITOR_COLS = ["r", "l", "t", "success", "n_found", "n_items", "contamination_entries", "collisions", "severe_collision"]


def read_monitor(path):
    """SB3 Monitor CSV; tolerant of a missing or repeated header (resumed runs append)."""
    rows = [ln.split(",") for ln in Path(path).read_text().splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("r,")]
    d = pd.DataFrame(rows, columns=MONITOR_COLS)
    for c in MONITOR_COLS:
        d[c] = d[c].map(lambda v: v == "True") if c in ("success", "severe_collision") else pd.to_numeric(d[c])
    return d


def learning_curves(ppo_dir, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    mons = sorted((ppo_dir / "monitor").glob("*.monitor.csv"))
    if mons:
        d = pd.concat([read_monitor(m) for m in mons]).sort_values("t").reset_index(drop=True)
        q = d.tail(500)                       # how well the policy does on its own training scenes at the end
        (Path(path).parent / "train_summary.json").write_text(json.dumps({
            "ppo_dir": _rel(ppo_dir), "training_episodes": int(len(d)), "training_steps": int(d.l.sum()),
            "last_500_episodes": {"success_rate": round(float(q.success.mean()), 4),
                                  "found_fraction": round(float((q.n_found / q.n_items).mean()), 4),
                                  "severe_collision_rate": round(float(q.severe_collision.mean()), 4),
                                  "stain_entries_mean": round(float(q.contamination_entries.mean()), 3)}}, indent=1))
        n_envs = len(mons)
        d["steps"] = d.l.cumsum()
        roll = lambda s: s.rolling(200, min_periods=20).mean()  # noqa: E731
        ax[0].plot(d.steps, roll(d.r), lw=1)
        ax[0].set_title(f"training episodes (rolling 200 of {len(d)}, {n_envs} envs)")
        ax[0].set_ylabel("episode return")
        ax[1].plot(d.steps, roll(d.success.astype(float)), label="success rate")
        ax[1].plot(d.steps, roll(d.n_found / d.n_items), label="found fraction")
        ax[1].plot(d.steps, roll((d.contamination_entries > 0).astype(float)), label="episodes w/ contamination")
        ax[1].set_ylim(0, 1)
        ax[1].legend(fontsize=8)
        ax[1].set_title("training episodes")
        for a_ in ax[:2]:
            a_.set_xlabel("environment steps")
    ev = ppo_dir / "evaluations.npz"
    if ev.exists():
        e = np.load(ev)
        ax[2].plot(e["timesteps"], e["results"].mean(1), "o-", ms=3, label="mean return")
        ax[2].fill_between(e["timesteps"], e["results"].mean(1) - e["results"].std(1),
                           e["results"].mean(1) + e["results"].std(1), alpha=0.2)
        ax[2].set_ylabel("return")
        if "successes" in e and e["successes"].size:
            a2 = ax[2].twinx()
            a2.plot(e["timesteps"], e["successes"].mean(1), "s--", color="C2", ms=3, label="success rate")
            a2.set_ylim(0, 1.05)
            a2.set_ylabel("success rate", color="C2")
        ax[2].set_xlabel("environment steps")
        ax[2].set_title("EvalCallback on validation layouts (deterministic)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def trajectory_plot(env, eps, path, layouts):
    pols = list(eps)
    fig, axes = plt.subplots(len(layouts), len(pols), figsize=(4.6 * len(pols), 2.75 * len(layouts)))
    axes = np.atleast_2d(axes)
    for r, (li, k) in enumerate(layouts):
        L = env.layouts[li]
        for c, p in enumerate(pols):
            e = eps[p][k]
            ax = axes[r, c]
            ext = (L.origin[0], L.origin[0] + L.W * L.res, L.origin[1], L.origin[1] + L.H * L.res)
            ax.imshow(np.where(L.blocked, 0.35, np.where(L.reach, 1.0, 0.85)), cmap="gray", vmin=0, vmax=1,
                      origin="lower", extent=ext)
            for x, y, rad in L.floor:
                ax.add_patch(plt.Circle((x, y), rad, color="darkred", alpha=0.35))
            t = e["traj"]
            ax.plot(t[:, 0], t[:, 1], "-", color="C0", lw=1.2)
            ax.plot(t[0, 0], t[0, 1], "o", color="C0", ms=6)
            for (x, y, _z), f in zip(e["targets"], e["found"]):
                ax.plot(x, y, "*", ms=12, color="limegreen" if f else "orange", mec="k", mew=0.6)
            ax.set_xlim(-0.2, 8.4)
            ax.set_ylim(-0.2, 4.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{p} | {L.name}: found {e['n_found']}/{e['n_items']}, {e['time_s']:.0f} s, "
                         f"contam {e['contamination_entries']}, coll {e['collisions']}", fontsize=7.5)
    fig.suptitle("Test episodes (same spawn & evidence for every policy). star: evidence (green = found), "
                 "red discs: floor stains", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppo-dir", default=str(OUT / "ppo"), help="training output folder (e.g. on Google Drive)")
    ap.add_argument("--model", help="default: <ppo-dir>/best_model.zip")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--policies", default="random,greedy,planner,ppo")
    ap.add_argument("--env-config", help="environment config (default config/q3_env.yaml)")
    ap.add_argument("--out", default=str(OUT), help="output folder for the tables and figures")
    a = ap.parse_args()
    a.model = a.model or str(Path(a.ppo_dir) / "best_model.zip")
    cfg = load_env_config(a.env_config)
    env = CrimeSceneSearchEnv(split="test", config=cfg)
    out = Path(a.out)
    pols = {"random": lambda: RandomPolicy(1), "greedy": lambda: greedy, "planner": lambda: planner,
            "ppo": lambda: PPOPolicy(a.model)}
    names = [p for p in a.policies.split(",")]
    rows, eps = [], {}
    pol_used = {}
    for p in names:
        pol = pol_used[p] = pols[p]()
        eps[p] = []
        for k in range(a.episodes):
            li = k % len(env.layouts)
            e = run_episode(env, pol, BASE_SEED + k, li)
            eps[p].append(e)
            rows.append({"policy": p, "episode": k, "seed": BASE_SEED + k,
                         **{c: e[c] for c in ("layout", "n_items", "n_found", "success", "time_s", "collisions",
                                              "severe_collision", "contamination_steps", "contamination_entries", "path_m", "return")}})
        d = pd.DataFrame([r for r in rows if r["policy"] == p])
        print(f"{p:7s} success {d.success.mean():.3f}  found {(d.n_found / d.n_items).mean():.3f}  time {d.time_s.mean():6.1f} s"
              f"  contam entries {d.contamination_entries.mean():.2f}  collisions {d.collisions.mean():.1f}", flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eval_episodes.csv", index=False)
    met = {"test_layouts": [L.name for L in env.layouts], "episodes_per_policy": a.episodes, "base_seed": BASE_SEED,
           "model": _rel(pol_used["ppo"].path) if "ppo" in names else None,
           "policies": {p: summary(df[df.policy == p]) for p in names},
           "per_layout_success": {p: df[df.policy == p].groupby("layout").success.mean().round(3).to_dict() for p in names}}
    (out / "eval_metrics.json").write_text(json.dumps(met, indent=1))
    lines = ["| Policy | Success | Evidence found | Time, successful eps (s) | Contamination entries / ep | "
             "Eps with contamination | Collision steps / ep | Severe-collision ends | Path (m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p, s in met["policies"].items():
        ts = f"{s['time_s_successful_mean']:.1f}" if s["time_s_successful_mean"] is not None else "-"
        lines.append(f"| {p} | {s['success_rate']:.3f} | {s['found_fraction']:.3f} | {ts} | "
                     f"{s['contamination_entries_mean']:.2f} | {s['episodes_with_contamination']:.2f} | "
                     f"{s['collisions_mean']:.1f} | {s['severe_collision_rate']:.2f} | {s['path_m_mean']:.1f} |")
    (out / "comparison.md").write_text("\n".join(lines) + f"\n\n{a.episodes} episodes per policy on the test layouts "
                                       f"{met['test_layouts']}; max episode length {cfg['episode']['max_steps'] * cfg['robot']['dt']:.0f} s.\n")
    print("\n".join(lines))
    picks = [(li, li) for li in range(min(3, len(env.layouts)))]
    trajectory_plot(env, eps, out / "trajectories.png", picks)
    learning_curves(Path(a.ppo_dir), out / "learning_curves.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
