"""Fast checks (no simulator needed): zone lookup and the deterministic Q2 run plan."""
import sys
from pathlib import Path

import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from go2sim.zones import Zones  # noqa: E402


def test_zone_lookup():
    z = Zones()
    assert z.names == ["victim_area", "bed_area", "lounge_area", "desk_entry_area", "bathroom"]
    assert z.zone_name((1.0, 1.0)) == "victim_area"
    assert z.zone_name((3.0, 3.0)) == "bed_area"
    assert z.zone_name((3.0, 1.0)) == "lounge_area"
    assert z.zone_name((5.0, 2.0)) == "desk_entry_area"
    assert z.zone_name((7.0, 2.0)) == "bathroom"
    assert z.zone_id((9.0, 9.0)) == -1


def test_zones_do_not_overlap():
    z = Zones()
    import numpy as np
    xs, ys = np.meshgrid(np.arange(0.02, 8.2, 0.1), np.arange(0.02, 4.5, 0.1))
    for x, y in zip(xs.ravel(), ys.ravel()):
        hits = [i for i, p in enumerate(z.polygons) if Zones._inside(p, x, y)]
        assert len(hits) <= 1


def test_run_plan_deterministic_and_unique():
    from q2_data_simulation.collect_dataset import run_plan
    cfg = yaml.safe_load((TEST2 / "config/collect.yaml").read_text())
    a, b = run_plan(cfg), run_plan(cfg)
    assert [r["route"] for r in a] == [r["route"] for r in b]
    ids = [r["run"] for r in a]
    assert len(ids) == len(set(ids)) == len(cfg["scenes"]) * (len(cfg["hand_routes"]) + cfg["random_routes_per_scene"])
