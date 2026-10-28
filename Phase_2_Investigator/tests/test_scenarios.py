"""A3: scenario configs, the scenario build (props + spatter + ground truth) and random scenarios."""
import numpy as np
import pytest
import yaml

import sim_env  # noqa: F401
from sim_env.runtime import SIM_ROOT
from sim_env.scenarios import scenario_paths

needs_sim = pytest.mark.skipif(not (SIM_ROOT / "sim/native/controller.py").exists(),
                               reason="simulator submodule not fetched")
MECHS = {"impact", "castoff", "projected", "drip", "transfer"}


@pytest.mark.parametrize("path", scenario_paths(), ids=lambda p: p.stem)
def test_config_schema(path):
    s = yaml.safe_load(path.read_text())
    for key in ("id", "environment", "spawn", "mannequins", "obstacles", "distractors", "events", "summary"):
        assert key in s, key
    assert len(s["spawn"]) == 3
    assert any(m["role"] == "victim" for m in s["mannequins"])
    assert len(s["obstacles"]) >= 1 and len(s["distractors"]) >= 2
    for e in s["events"]:
        assert e["mechanism"] in MECHS
        assert e.get("weapon_category") in ("blunt", "edged", "none")


def test_five_scenarios_cover_all_mechanisms_and_weapon_categories():
    mechs, cats = set(), set()
    for p in scenario_paths():
        s = yaml.safe_load(p.read_text())
        mechs |= {e["mechanism"] for e in s["events"]}
        cats |= {e["weapon_category"] for e in s["events"]}
    assert len(scenario_paths()) == 5
    assert mechs == MECHS
    assert cats == {"blunt", "edged", "none"}


@needs_sim
def test_simple_room_scenario_builds_with_ground_truth():
    from sim_env.scenario_scene import build_scenario, load_scenario_config
    import json
    scn = load_scenario_config(5)
    out = build_scenario(scn)
    gt = json.loads((out / "ground_truth.json").read_text())
    assert gt["environment"] == "csi_simple_room" and gt["spawn"] == scn["spawn"]
    landed = [d for d in gt["droplets"] if d["landed"]]
    assert len(landed) > 0.9 * len(gt["droplets"])
    for d in landed:
        assert {"impact_pos", "impact_angle_deg", "gamma_deg", "dir_world", "surface_id"} <= set(d)
    assert gt["smears"] and gt["clusters"]
    assert {"origin", "mechanism", "weapon_category"} <= set(gt["events"][0])
    assert all(isinstance(x["z"], float) and x["z"] > 0.3 for x in gt["distractors"])   # on furniture
    xml = (out / "stains.xml").read_text()
    assert xml.count("<geom") >= len(gt["stains"])
    # the largest cluster is the impact pattern on the south wall
    c0 = gt["clusters"][0]
    assert c0["surface_kind"] == "wall" and "impact" in c0["mechanisms"] and abs(c0["center"][1]) < 0.1


@needs_sim
def test_random_scenarios_are_seeded_and_valid():
    from sim_env.scenarios.random_sim import random_sim_scenario
    a = random_sim_scenario(21, env="csi_simple_room")
    b = random_sim_scenario(21, env="csi_simple_room")
    c = random_sim_scenario(22, env="csi_simple_room")
    assert a == b and a != c
    for s in (a, c):
        assert s["events"] and s["mannequins"][0]["role"] == "victim"
        assert len(s["obstacles"]) == 2 and len(s["distractors"]) == 3
        v = np.array(s["mannequins"][0]["event_position"])
        assert np.hypot(*(v - np.array(s["spawn"][:2]))) > 1.5
