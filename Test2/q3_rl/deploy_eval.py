#!/usr/bin/env python3
"""Q3: statistical evaluation of the search policies on the full MuJoCo Go2, and the 2-D -> MuJoCo gap.

    python q3_rl/deploy_eval.py                         # 6 test scenes x 10 starts x {ppo, greedy, planner}
    python q3_rl/deploy_eval.py --episodes 3 --policies ppo --layouts s2

For every test scene the MuJoCo scene is opened once (one process per scene). Episode k uses
CrimeSceneSearchEnv.reset(seed=BASE_SEED + k) with randomize=False: the start pose comes from the seed and
ALL of the scene's evidence items must be found (no subset, no jitter, no noise). Each episode is run twice
from the identical start:
  * 2-D: in CrimeSceneSearchEnv (the training simulator);
  * MuJoCo: on the legged Go2 with the learned locomotion controller (deploy_mujoco.run_on_dog).
Writes results/q3/deploy_eval_episodes.csv (one row per episode and simulator), deploy_eval_metrics.json
and deploy_eval.md (the README table).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

BASE_SEED = 700_000
OUT = TEST2 / "results/q3"
TEST_LAYOUTS = ["s2", "r1854", "r3101", "r3102", "r3103", "r3104"]


def worker(args):
    layout, policies, episodes, model = args
    import go2sim.paths  # noqa: F401
    from go2sim.go2_env import CrimeSceneGo2
    from q3_rl.crime_scene_env import CrimeSceneSearchEnv, load_env_config
    from q3_rl.deploy_mujoco import make_policy, run_on_dog
    from q3_rl.evaluate import run_episode
    from q3_rl.extract_layouts import scenario_spec
    cfg = load_env_config()
    max_s = cfg["episode"]["max_steps"] * cfg["robot"]["dt"]
    env_mj = CrimeSceneSearchEnv(layouts=[layout], config=cfg, randomize=False)
    env_2d = CrimeSceneSearchEnv(layouts=[layout], config=cfg, randomize=False)
    dog = CrimeSceneGo2(scenario_spec(layout))
    rows = []
    try:
        for p in policies:
            pol = make_policy(p, model)
            for k in range(episodes):
                seed = BASE_SEED + k
                e = run_episode(env_2d, pol, seed, 0)
                rows.append({"layout": layout, "policy": p, "episode": k, "seed": seed, "sim": "2d",
                             "n_items": e["n_items"], "n_found": e["n_found"], "success": e["success"],
                             "time_s": round(e["time_s"], 2), "stain_entries": e["contamination_entries"],
                             "collision_steps": e["collisions"], "severe_collision": e["severe_collision"]})
                env_mj.reset(seed=seed, options={"layout": 0})
                assert np.allclose(env_mj.xy, e["traj"][0][:2]), "2-D and MuJoCo starts differ"
                dog.teleport(env_mj.xy[0], env_mj.xy[1], env_mj.yaw)
                t0 = time.time()
                _, st = run_on_dog(dog, env_mj, pol, max_s)
                rows.append({"layout": layout, "policy": p, "episode": k, "seed": seed, "sim": "mujoco",
                             "n_items": st["n_items"], "n_found": st["n_found"], "success": st["success"],
                             "time_s": st["time_s"], "stain_entries": st["footprint_stain_entries"],
                             "collision_steps": st["steps_with_robot_scene_contact"],
                             "foot_in_stain_steps": st["steps_with_foot_in_stain"], "min_upright": st["min_upright"],
                             "wall_s": round(time.time() - t0, 1)})
                print(f"{layout} {p} ep{k}: 2-D {'OK ' if e['success'] else 'FAIL'} {e['time_s']:5.1f}s | MuJoCo "
                      f"{'OK ' if st['success'] else 'FAIL'} {st['time_s']:5.1f}s found {st['n_found']}/{st['n_items']}",
                      flush=True)
    finally:
        dog.close()
    return rows


def summarise(df):
    out = {}
    for (p, sim), g in df.groupby(["policy", "sim"]):
        s = g[g.success]
        out.setdefault(p, {})[sim] = {
            "episodes": int(len(g)), "success_rate": round(float(g.success.mean()), 4),
            "found_fraction": round(float((g.n_found / g.n_items).mean()), 4),
            "time_s_successful_mean": round(float(s.time_s.mean()), 2) if len(s) else None,
            "stain_entries_mean": round(float(g.stain_entries.mean()), 3),
            "collision_steps_mean": round(float(g.collision_steps.mean()), 2),
            **({"foot_in_stain_steps_mean": round(float(g.foot_in_stain_steps.mean()), 2),
                "min_upright": round(float(g.min_upright.min()), 3)} if sim == "mujoco" else {})}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", default=",".join(TEST_LAYOUTS))
    ap.add_argument("--policies", default="ppo,greedy,planner")
    ap.add_argument("--episodes", type=int, default=10, help="start poses per scene")
    ap.add_argument("--model", default=str(OUT / "ppo/best_model.zip"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    layouts, pols = a.layouts.split(","), a.policies.split(",")
    t0 = time.time()
    jobs = [(L, pols, a.episodes, a.model) for L in layouts]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(worker, jobs) for r in part]
    df = pd.DataFrame(rows).sort_values(["policy", "layout", "episode", "sim"]).reset_index(drop=True)
    df.to_csv(OUT / "deploy_eval_episodes.csv", index=False)
    met = {"layouts": layouts, "episodes_per_layout": a.episodes, "base_seed": BASE_SEED, "randomize": False,
           "policies": summarise(df),
           "agreement_2d_vs_mujoco": {p: round(float((g[g.sim == "2d"].success.to_numpy()
                                                       == g[g.sim == "mujoco"].success.to_numpy()).mean()), 4)
                                      for p, g in df.groupby("policy")},
           "per_layout_success": {p: {sim: g[g.sim == sim].groupby("layout").success.mean().round(2).to_dict()
                                      for sim in ("2d", "mujoco")} for p, g in df.groupby("policy")},
           "wall_time_s": round(time.time() - t0, 1)}
    (OUT / "deploy_eval_metrics.json").write_text(json.dumps(met, indent=1))
    lines = ["| Policy | Success 2-D | Success MuJoCo | Same outcome | Time, successful (2-D / MuJoCo, s) | "
             "Stain entries / ep (2-D / MuJoCo) | Steps with a foot in a stain / ep | Contact steps / ep (2-D / MuJoCo) |",
             "|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    for p in pols:
        d, m = met["policies"][p]["2d"], met["policies"][p]["mujoco"]
        lines.append(f"| {p} | {d['success_rate']:.3f} | {m['success_rate']:.3f} | {met['agreement_2d_vs_mujoco'][p]:.2f} | "
                     f"{f(d['time_s_successful_mean'])} / {f(m['time_s_successful_mean'])} | "
                     f"{d['stain_entries_mean']:.2f} / {m['stain_entries_mean']:.2f} | {m['foot_in_stain_steps_mean']:.1f} | "
                     f"{d['collision_steps_mean']:.1f} / {m['collision_steps_mean']:.1f} |")
    txt = "\n".join(lines) + (f"\n\n{a.episodes} start poses x {len(layouts)} test scenes = {a.episodes * len(layouts)} "
                              f"episodes per policy and simulator; all evidence of the scene must be found; 180 s limit.\n")
    (OUT / "deploy_eval.md").write_text(txt)
    print(txt)
    print(f"done in {met['wall_time_s']} s -> {OUT}")


if __name__ == "__main__":
    main()
