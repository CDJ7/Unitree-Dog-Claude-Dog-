"""Go-to-zone navigator logic on scripted inputs (no robot, no camera)."""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_go2.goto_zone import ZoneNavigator, wrap  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "real_go2/real_config.yaml").read_text())
CLASSES, NAV = CFG["classes"], CFG["navigation"]


def prob_for(target, conf, n=4):
    p = np.full(n, (1 - conf) / (n - 1))
    p[target] = conf
    return p


def run_look(nav, t, h, view_fn, dt=0.1):
    """Simulate turning on the spot: heading follows the commanded rate, one frame per step."""
    for _ in range(400):
        vx, wz = nav.update(t, h, view_fn(h))
        if nav.state != "look":
            return t, h
        t, h = t + dt, wrap(h + wz * dt)
    raise AssertionError("look-around never finished")


def test_finds_target_direction_then_walks():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    shelves = CLASSES.index("shelves")
    nav.start(shelves, 0.0, 0.0)
    # the shelves are seen only when looking around +90 deg; elsewhere the dog sees open_area
    view = lambda h: prob_for(shelves, 0.9) if abs(wrap(h - math.pi / 2)) < 0.4 else prob_for(2, 0.8)
    t, h = run_look(nav, 0.0, 0.0, view)
    assert nav.state == "turn" and nav.where == 2
    assert abs(wrap(nav.best_heading - math.pi / 2)) < math.radians(20)
    for _ in range(200):                                   # turn to the best heading
        vx, wz = nav.update(t, h, None)
        if nav.state != "turn":
            break
        t, h = t + 0.1, wrap(h + wz * 0.1)
    assert nav.state == "walk" and nav.legs == 1
    vx, wz = nav.update(t, h, prob_for(shelves, 0.6), obstacle_m=None)
    assert vx == NAV["approach_speed"]


def test_obstacle_ends_leg_and_arrives_when_target_dominates():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.start(0, 0.0, 0.0)
    nav.state, nav.best_heading, nav._t0 = "walk", 0.0, 0.0
    vx, _ = nav.update(0.1, 0.0, prob_for(0, 0.6), obstacle_m=0.3)
    assert vx == 0.0 and nav.state == "look" and "obstacle" in nav.msg
    run_look(nav, 0.2, 0.0, lambda h: prob_for(0, 0.85))   # lab_1 in every direction
    assert nav.state == "arrived" and nav.share >= NAV["arrive_share"]


def test_not_seen_fails_and_no_lidar_limits():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=False)
    assert nav.nav["approach_speed"] <= 0.15 and nav.nav["leg_s"] <= 2.0
    nav.start(1, 0.0, 0.0)
    run_look(nav, 0.0, 0.0, lambda h: prob_for(2, 0.95))   # lab_2 never seen
    assert nav.state == "failed" and "not seen" in nav.msg
    nav.cancel()
    assert nav.state == "failed"                           # finished states are not overwritten
