"""Command-line options shared by every script that starts the simulator.

    --scenario N|id|path   a scenario (sim_env/scenarios/configs, 1-based): props, stains, spawn
    --random SEED          a seeded random scenario (sim_env/scenarios/random_sim.py)
    --env ID               an empty simulator environment (default house 185 when no scenario)
    --spawn X Y YAW        override the robot's start pose
    --no-stains            load the scenario without its stains (e.g. for comparisons)
    --rebuild              regenerate the scenario's spatter even if cached
"""
from __future__ import annotations

import argparse

from sim_env.runtime import DEFAULT_ENVIRONMENT, SimSession


def add_sim_args(parser: argparse.ArgumentParser):
    parser.add_argument("--scenario", help="scenario number (1-based), id or YAML path")
    parser.add_argument("--random", type=int, metavar="SEED", help="seeded random scenario")
    parser.add_argument("--env", help="simulator environment id, e.g. scenesmith_house_186")
    parser.add_argument("--spawn", nargs=3, type=float, metavar=("X", "Y", "YAW"))
    parser.add_argument("--no-stains", action="store_true", help="scenario without its stains")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the scenario's spatter")
    return parser


def scenario_from_args(args):
    """The scenario config dict for the options, or None."""
    from sim_env.scenario_scene import load_scenario_config
    if getattr(args, "random", None) is not None:
        from sim_env.scenarios.random_sim import write_random
        return load_scenario_config(str(write_random(args.random)))
    if args.scenario:
        return load_scenario_config(args.scenario)
    return None


def session_from_args(args, **kwargs):
    """(SimSession, ground truth dict or None) for the parsed options."""
    scn = scenario_from_args(args)
    if scn is not None and not args.env:
        from sim_env.scenario_scene import open_scenario
        sim, gt = open_scenario(scn, show_stains=not args.no_stains, spawn=args.spawn,
                                force=args.rebuild, **kwargs)
        print(f"scenario {scn['id']}: {scn.get('title', '')}")
        return sim, gt
    sim = SimSession(args.env or DEFAULT_ENVIRONMENT, spawn=args.spawn, **kwargs)
    return sim, None
