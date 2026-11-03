"""Q3 environment tests: Gymnasium/SB3 API compliance, determinism, reward terms, episode ends."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q3_rl import crime_scene_env as cse  # noqa: E402
from q3_rl.crime_scene_env import CrimeSceneSearchEnv, load_env_config  # noqa: E402


def test_env_checkers():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    check_env(CrimeSceneSearchEnv(split="train"))
    sb3_check(CrimeSceneSearchEnv(split="test"), warn=True)


def test_seed_determinism():
    a, b = CrimeSceneSearchEnv(split="train"), CrimeSceneSearchEnv(split="train")
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(50):
        act = rng.uniform(-1, 1, 3)
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]


def test_reward_terms_signs():
    w = load_env_config()["reward"]
    assert cse.reward_find(w, 2) == 2 * w["find"]
    assert cse.reward_progress(w, 3.0, 2.8) > 0 > cse.reward_progress(w, 2.8, 3.0)
    assert cse.reward_progress(w, np.inf, 1.0) == 0.0
    assert cse.reward_contamination(w, True, 1) == -(w["contamination"] + w["contamination_entry"])
    assert cse.reward_collision(w, True) < 0 == cse.reward_collision(w, False)
    assert cse.reward_collision(w, True, severe=True) == -(w["collision"] + w["severe_collision"])
    assert cse.reward_time(w) < 0 and cse.reward_all_found(w, True) == w["all_found"]


def test_spawn_clear_and_truncation():
    cfg = load_env_config()
    cfg["episode"]["max_steps"] = 20
    env = CrimeSceneSearchEnv(split="train", config=cfg)
    for s in range(20):
        env.reset(seed=s)
        assert env._on_stain() < 0 and env.layout.reach[env.layout.cell(env.xy)]
    term = trunc = False
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.zeros(3))
    assert trunc and not term and info["time_s"] == 20 * cfg["robot"]["dt"]


def test_find_terminates():
    """Teleport in front of every item: each is found and the episode terminates with the bonus."""
    env = CrimeSceneSearchEnv(split="test", randomize=False)
    env.reset(seed=0, options={"layout": 0})
    L = env.layout
    for i in range(len(env.tgt)):
        f = L.field(int(env.items[i]))
        r, c = np.unravel_index(np.argmin(np.where(np.isfinite(f), f, np.inf)), f.shape)
        env.xy = L.cell_xy(r, c)
        d = env.tgt[i, :2] - (env.xy)
        env.yaw = float(np.arctan2(d[1], d[0]))
        _, rew, term, _, info = env.step(np.zeros(3))
        assert env.found[i], f"item {i} not found from its viewpoint"
    assert term and info["success"]


def test_severe_collision_terminates():
    """Drive straight into a wall: the episode terminates as a failure after severe_collision_steps."""
    cfg = load_env_config()
    env = CrimeSceneSearchEnv(split="train", config=cfg)
    env.reset(seed=3)
    term, n = False, 0
    while not term and n < 400:
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0]))
        n += 1
    assert term and info["severe_collision"] and not info["success"]
