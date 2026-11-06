#!/usr/bin/env python3
"""Q3: train PPO (stable-baselines3) on CrimeSceneSearchEnv.

    python q3_rl/train_ppo.py                         # config/q3_ppo.yaml
    python q3_rl/train_ppo.py --timesteps 200000      # shorter run
    python q3_rl/train_ppo.py --resume                # continue from the latest checkpoint in out_dir
    python q3_rl/train_ppo.py --out /content/drive/MyDrive/test2_q3   # e.g. Colab + Google Drive

- vectorised envs (SubprocVecEnv by default) with domain randomisation, each wrapped in Monitor
- TensorBoard logs (tensorboard --logdir results/q3/tb)
- CheckpointCallback every `checkpoint_every` steps (model + VecNormalize-free, so a zip is enough)
- EvalCallback on the VALIDATION layouts (never the test layouts): best_model.zip + evaluations.npz
- all hyper-parameters + env config + versions saved to <out>/train_config.json
Resume: loads the newest checkpoints/ppo_<steps>_steps.zip and continues the step counter, so the
learning curve and TensorBoard continue where they stopped (useful when Colab disconnects).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from q3_rl.crime_scene_env import CrimeSceneSearchEnv, load_env_config  # noqa: E402


class InfoLogger:
    """Factory for an SB3 callback that logs episode success/contamination/collisions to TensorBoard."""

    @staticmethod
    def make():
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.buf = []

            def _on_step(self):
                for done, info in zip(self.locals["dones"], self.locals["infos"]):
                    if done:
                        self.buf.append(info)
                if len(self.buf) >= 32:
                    b = self.buf
                    self.logger.record("search/success_rate", np.mean([i["success"] for i in b]))
                    self.logger.record("search/found_fraction", np.mean([i["n_found"] / max(i["n_items"], 1) for i in b]))
                    self.logger.record("search/contamination_entries", np.mean([i["contamination_entries"] for i in b]))
                    self.logger.record("search/collisions", np.mean([i["collisions"] for i in b]))
                    self.logger.record("search/time_s", np.mean([i["time_s"] for i in b]))
                    self.buf = []
                return True
        return _CB()


def make_env(split, env_cfg, seed, rank, monitor_dir=None, curriculum_level=None):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        env = CrimeSceneSearchEnv(split=split, config=env_cfg, curriculum_level=curriculum_level)
        fn = str(Path(monitor_dir) / f"rank{rank:02d}") if monitor_dir else None
        fresh = fn is None or not Path(fn + ".monitor.csv").exists()     # append when resuming
        env = Monitor(env, filename=fn, override_existing=fresh, info_keywords=("success", "n_found", "n_items", "contamination_entries", "collisions", "severe_collision"))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _f


def curriculum_callback(n_levels, level, path, threshold, window):
    """SB3 callback for the environment curriculum (config `curriculum`): when the success rate of the
    last `window` finished training episodes reaches `threshold`, every training env moves to the next
    level; after the last level comes the full task. The level and its history are saved to `path`, so
    --resume continues at the same level."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.level, self.buf = level, []
            st = json.loads(Path(path).read_text()) if Path(path).exists() else {}
            self.history = st.get("history", [[0, level]])

        def _save(self):
            Path(path).write_text(json.dumps({"level": self.level, "n_levels": n_levels, "full_task_level": n_levels,
                                              "threshold": threshold, "window": window,
                                              "history": self.history}, indent=1))

        def _on_training_start(self):
            self._save()

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.buf.append(float(info["success"]))
            if self.level < n_levels and len(self.buf) >= window and np.mean(self.buf[-window:]) >= threshold:
                self.level += 1
                self.training_env.env_method("set_curriculum_level", self.level)
                self.history.append([int(self.num_timesteps), self.level])
                self.buf = []
                self._save()
                print(f"curriculum -> level {self.level}{' (full task)' if self.level == n_levels else ''} "
                      f"at {self.num_timesteps} steps", flush=True)
            self.logger.record("curriculum/level", self.level)
            return True
    return _CB()


def trim_monitors(mon_dir, steps_per_env):
    """Keep only the episodes each env finished before the checkpoint (cumulative length <= steps_per_env),
    so a run interrupted after its last checkpoint does not count those steps twice."""
    for f in Path(mon_dir).glob("*.monitor.csv"):
        keep, total = [], 0
        for ln in f.read_text().splitlines():
            if ln.startswith("#") or ln.startswith("r,") or not ln:
                keep.append(ln)
                continue
            total += int(ln.split(",")[1])
            if total > steps_per_env:
                break
            keep.append(ln)
        f.write_text("\n".join(keep) + "\n")


def latest_checkpoint(out):
    cks = sorted((out / "checkpoints").glob("ppo_*_steps.zip"), key=lambda p: int(re.findall(r"(\d+)_steps", p.name)[0]))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(TEST2 / "config/q3_ppo.yaml"))
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.timesteps:
        cfg["total_timesteps"] = a.timesteps
    if a.n_envs:
        cfg["n_envs"] = a.n_envs
    out = Path(a.out) if a.out else TEST2 / cfg["out_dir"]
    tb = (out / "tb") if a.out else TEST2 / cfg["tensorboard_dir"]
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = load_env_config(TEST2 / cfg["env_config"])
    seed = cfg["seed"]
    set_random_seed(seed)
    torch.set_num_threads(1)
    n = cfg["n_envs"]
    Vec = SubprocVecEnv if cfg.get("vec_env", "subproc") == "subproc" and n > 1 else DummyVecEnv
    (out / "monitor").mkdir(exist_ok=True)
    ck0 = latest_checkpoint(out) if a.resume else None
    if ck0 is None:                        # fresh run: start new Monitor logs
        for f in (out / "monitor").glob("*.monitor.csv"):
            f.unlink()
    else:                                  # resume: drop episodes logged after the checkpoint, then append
        trim_monitors(out / "monitor", int(re.findall(r"(\d+)_steps", ck0.name)[0]) // n)
    cur = env_cfg.get("curriculum")
    cur_path = out / "curriculum.json"
    level = None
    if cur:
        level = 0
        if a.resume and ck0 and cur_path.exists():      # the level reached by the checkpoint's step
            ck_steps = int(re.findall(r"(\d+)_steps", ck0.name)[0])
            hist = [h for h in json.loads(cur_path.read_text())["history"] if h[0] <= ck_steps]
            level = hist[-1][1] if hist else 0
            cur_path.write_text(json.dumps({**json.loads(cur_path.read_text()), "level": level, "history": hist}))
        elif cur_path.exists():
            cur_path.unlink()
    venv = Vec([make_env("train", env_cfg, seed, i, out / "monitor", level) for i in range(n)])
    eval_env = DummyVecEnv([make_env("val", env_cfg, seed + 10_000, 0)])
    p = dict(cfg["ppo"])
    policy_kwargs = {"net_arch": p.pop("net_arch"), "log_std_init": p.pop("log_std_init")}
    ck = latest_checkpoint(out) if a.resume else None
    if ck:
        model = PPO.load(ck, env=venv, tensorboard_log=str(tb), device="cpu")
        print(f"resumed from {ck} at {model.num_timesteps} steps", flush=True)
    else:
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0, tensorboard_log=str(tb), device="cpu",
                    policy_kwargs=policy_kwargs, **p)
    per_env = lambda k: max(k // n, 1)  # noqa: E731  callbacks count calls, one call = n env steps
    evcb = EvalCallback(eval_env, n_eval_episodes=cfg["eval_episodes"], eval_freq=per_env(cfg["eval_every"]),
                        best_model_save_path=str(out), log_path=str(out), deterministic=True, verbose=0)
    if ck and (out / "evaluations.npz").exists():      # keep the eval history and best score across resumes
        e = np.load(out / "evaluations.npz")
        keep = e["timesteps"] <= model.num_timesteps
        evcb.evaluations_timesteps = list(e["timesteps"][keep])
        evcb.evaluations_results = [list(r) for r in e["results"][keep]]
        evcb.evaluations_length = [list(r) for r in e["ep_lengths"][keep]]
        if "successes" in e:
            evcb._is_success_buffer = []
            evcb.evaluations_successes = [list(r) for r in e["successes"][keep]]
        if keep.any():
            evcb.best_mean_reward = float(e["results"][keep].mean(1).max())
    cbs = CallbackList([
        CheckpointCallback(per_env(cfg["checkpoint_every"]), str(out / "checkpoints"), name_prefix="ppo"),
        evcb, InfoLogger.make(),
        *([curriculum_callback(len(cur["levels"]), level, cur_path, cur["threshold"], cur["window"])] if cur else [])])
    info = {"config": cfg, "env_config": env_cfg, "policy_kwargs": policy_kwargs, "vec_env": Vec.__name__,
            "resumed_from": str(ck) if ck else None, "torch": torch.__version__,
            "sb3": __import__("stable_baselines3").__version__, "gymnasium": __import__("gymnasium").__version__,
            "n_train_layouts": len(env_cfg["layouts"]["train"])}
    (out / "train_config.json").write_text(json.dumps(info, indent=1))
    t0 = time.time()
    remaining = cfg["total_timesteps"] - (model.num_timesteps if ck else 0)
    print(f"training {remaining} steps with {n} x {Vec.__name__} -> {out}", flush=True)
    model.learn(total_timesteps=max(remaining, 0), callback=cbs, reset_num_timesteps=not ck,
                tb_log_name="ppo", progress_bar=False)
    model.save(out / "final_model")
    info["wall_time_s"] = round(time.time() - t0, 1)
    info["final_timesteps"] = int(model.num_timesteps)
    (out / "train_config.json").write_text(json.dumps(info, indent=1))
    print(f"done: {model.num_timesteps} steps in {info['wall_time_s']} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
