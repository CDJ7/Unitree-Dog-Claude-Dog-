"""Command-line options shared by every script that starts the simulator.

    --scenario N|id|path   a scenario config (sim_env/scenarios/configs, 1-based)
    --env ID               a simulator environment (default: scenario's, else house 185)
    --spawn X Y YAW        override the robot's start pose
"""
from __future__ import annotations

import argparse

import yaml

from sim_env.runtime import DEFAULT_ENVIRONMENT, SimSession
from sim_env.scenarios import resolve_scenario


def add_sim_args(parser: argparse.ArgumentParser):
    parser.add_argument("--scenario", help="scenario number (1-based), id or YAML path")
    parser.add_argument("--env", help="simulator environment id, e.g. scenesmith_house_186")
    parser.add_argument("--spawn", nargs=3, type=float, metavar=("X", "Y", "YAW"))
    return parser


def session_from_args(args, **kwargs):
    """(SimSession, scenario dict or None) for the parsed options."""
    scenario = None
    env = args.env
    if args.scenario:
        path = resolve_scenario(args.scenario)
        scenario = yaml.safe_load(path.read_text())
        env = env or (scenario.get("binding") or {}).get("environment")
    sim = SimSession(env or DEFAULT_ENVIRONMENT, spawn=args.spawn, **kwargs)
    return sim, scenario
