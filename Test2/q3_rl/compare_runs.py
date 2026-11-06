#!/usr/bin/env python3
"""Q3: paired comparison of two PPO runs on the MuJoCo Go2 (same scenes, same start poses).

    python q3_rl/compare_runs.py --a results/q3/v2_disc --b results/q3 --out results/q3/paired_v4_vs_v2.json

Reads the deploy_eval_episodes.csv of both runs (written by deploy_eval.py with the same seeds), pairs
the MuJoCo PPO episodes by (scene, start), and reports how often each run succeeds where the other
fails, with an exact two-sided McNemar (binomial) test, plus how the steps with a foot in a stain are
spread over the episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def mujoco_ppo(run_dir):
    d = pd.read_csv(Path(run_dir) / "deploy_eval_episodes.csv")
    return d[(d.sim == "mujoco") & (d.policy == "ppo")].set_index(["layout", "episode"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run folder A (e.g. results/q3/v2_disc)")
    ap.add_argument("--b", required=True, help="run folder B (e.g. results/q3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = mujoco_ppo(a.a), mujoco_ppo(a.b)
    j = A[["seed", "success", "foot_in_stain_steps"]].join(B[["seed", "success", "foot_in_stain_steps"]],
                                                           lsuffix="_a", rsuffix="_b", how="inner")
    assert (j.seed_a == j.seed_b).all(), "the two runs used different start seeds"
    only_a = int((j.success_a & ~j.success_b).sum())
    only_b = int((~j.success_a & j.success_b).sum())
    disc = only_a + only_b

    def stains(f):
        return {"total_steps": int(f.sum()), "episodes_with_any": int((f > 0).sum()), "median": float(f.median()),
                "top3_episodes": [int(v) for v in f.nlargest(3)]}
    res = {"a": a.a, "b": a.b, "paired_episodes": int(len(j)),
           "success_rate_a": round(float(j.success_a.mean()), 4), "success_rate_b": round(float(j.success_b.mean()), 4),
           "both_succeed": int((j.success_a & j.success_b).sum()), "only_a_succeeds": only_a,
           "only_b_succeeds": only_b, "neither": int((~j.success_a & ~j.success_b).sum()),
           "mcnemar_exact_p": round(float(binomtest(only_b, disc, 0.5).pvalue), 4) if disc else 1.0,
           "foot_in_stain_a": stains(j.foot_in_stain_steps_a), "foot_in_stain_b": stains(j.foot_in_stain_steps_b)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
