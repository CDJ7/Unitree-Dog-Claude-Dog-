"""Q3: reinforcement-learning evidence search (CrimeSceneSearchEnv + PPO)."""
import gymnasium as gym

if "CrimeSceneSearch-v0" not in gym.registry:
    gym.register(id="CrimeSceneSearch-v0", entry_point="q3_rl.crime_scene_env:CrimeSceneSearchEnv")
