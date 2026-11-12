"""Go-to-zone navigator on scripted inputs (no robot, no camera): zone finding and obstacle avoidance."""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_go2.goto_zone import ZoneNavigator, free_profile, wrap  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "real_go2/real_config.yaml").read_text())
CLASSES, NAV = CFG["classes"], CFG["navigation"]
NB, LOOK = NAV["heading_bins"], NAV["lookahead_m"]
CORRIDOR = {"x_min": 0.05, "half_width_m": 0.3, "z_min": -0.15, "z_max": 0.45, "min_points": 3}
SHELVES, OPEN = CLASSES.index("shelves"), CLASSES.index("open_area")


def prob_for(target, conf, n=4):
    p = np.full(n, (1 - conf) / (n - 1))
    p[target] = conf
    return p


def body_free(world_free_fn, heading):
    """Body-frame profile seen at `heading` from a function of absolute direction."""
    return np.array([world_free_fn(heading + i * 2 * math.pi / NB) for i in range(NB)])


def run_look(nav, t, h, view_fn, world_free_fn=lambda a: LOOK, dt=0.1):
    for _ in range(400):
        vx, wz = nav.update(t, h, view_fn(h), body_free(world_free_fn, h))
        if nav.state != "look":
            return t, h
        t, h = t + dt, wrap(h + wz * dt)
    raise AssertionError("look-around never finished")


def run_turn(nav, t, h, dt=0.1):
    for _ in range(300):
        vx, wz = nav.update(t, h, None, None)
        if nav.state != "turn":
            return t, h
        t, h = t + dt, wrap(h + wz * dt)
    raise AssertionError("turn never finished")


def near(a, b, deg=20):
    return abs(wrap(a - b)) < math.radians(deg)


def shelves_at_90(h):
    """The shelves are seen only when looking around +90 deg; elsewhere the dog sees the open area."""
    return prob_for(SHELVES, 0.9) if abs(wrap(h - math.pi / 2)) < 0.4 else prob_for(OPEN, 0.8)


def test_free_profile_wall_ahead():
    wall = np.array([[1.0, y, 0.2] for y in np.linspace(-0.25, 0.25, 11)])   # wall 1 m ahead, across the corridor
    side = np.array([[0.6, 0.6, 0.2]] * 5)                                     # beside the corridor: ignored ahead
    f = free_profile(np.vstack([wall, side]), NB, CORRIDOR, LOOK)
    assert abs(f[0] - 1.0) < 1e-6
    assert f[NB // 2] == LOOK                                                  # behind: clear
    assert f[0] < f[NB // 4]                                                   # left (+90): no wall there
    floor = np.array([[1.0, 0.0, -0.4]] * 10)                                  # below z_min: floor, ignored
    assert free_profile(floor, NB, CORRIDOR, LOOK)[0] == LOOK


def test_finds_target_direction_when_clear():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.start(SHELVES, 0.0, 0.0)
    t, h = run_look(nav, 0.0, 0.0, shelves_at_90)
    assert nav.state == "turn" and nav.where == OPEN and not nav.detour
    assert near(nav.best_heading, math.pi / 2)
    t, h = run_turn(nav, t, h)
    assert nav.state == "walk" and nav.legs == 1
    vx, wz = nav.update(t, h, prob_for(SHELVES, 0.6), body_free(lambda a: LOOK, h))
    assert vx == NAV["approach_speed"] and nav.avoiding == 0


def test_detour_when_the_targets_way_is_blocked():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.start(SHELVES, 0.0, 0.0)
    # a desk blocks the way at 60..120 deg (0.4 m free); everything else is clear
    wall = lambda a: 0.4 if abs(wrap(a - math.pi / 2)) <= math.radians(31) else LOOK
    run_look(nav, 0.0, 0.0, shelves_at_90, wall)
    assert nav.state == "turn" and nav.detour
    assert not near(nav.best_heading, math.pi / 2, 25)            # not straight into the desk...
    assert near(nav.best_heading, math.pi / 2, 70)                # ...but the free way closest to the target
    assert nav.free[nav.bin_of(nav.best_heading)] >= NAV["want_free_m"]


def test_steers_round_obstacle_then_remembers_blocked_direction():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.start(SHELVES, 0.0, 0.0)
    nav.state, nav.best_heading, nav._t0, nav.legs = "walk", 0.0, 0.0, 1
    # obstacle 0.8 m ahead, the left is free, the right is not: slow down and steer left
    free = np.full(NB, LOOK)
    free[0], free[-1], free[-2] = 0.8, 0.6, 0.6
    vx, wz = nav.update(0.1, 0.0, prob_for(SHELVES, 0.6), free)
    assert nav.avoiding == 1 and wz > 0 and 0 < vx < NAV["approach_speed"]
    # closer than stop_obstacle_m: the leg ends and that direction is remembered as blocked
    free[0] = 0.3
    vx, wz = nav.update(0.2, 0.0, prob_for(SHELVES, 0.6), free)
    assert vx == 0.0 and nav.state == "look" and "blocked" in nav.msg and nav.blocked == {0: 1}
    # next look-around: the target still looks likeliest straight ahead (0 deg), which just blocked: go round
    run_look(nav, 0.3, 0.0, lambda h: prob_for(SHELVES, 0.9) if abs(wrap(h)) < 0.4 else prob_for(OPEN, 0.8))
    assert nav.state == "turn" and not near(nav.best_heading, 0.0, 20)


def test_boxed_in_and_arrival():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.start(SHELVES, 0.0, 0.0)
    run_look(nav, 0.0, 0.0, shelves_at_90, lambda a: 0.3)          # walls all round
    assert nav.state == "failed" and "boxed in" in nav.msg
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.start(0, 0.0, 0.0)
    run_look(nav, 0.0, 0.0, lambda h: prob_for(0, 0.85))          # lab_1 in every direction
    assert nav.state == "arrived" and nav.share >= NAV["arrive_share"]


def test_not_seen_and_no_lidar_limits():
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=False)
    assert nav.nav["approach_speed"] <= 0.15 and nav.nav["leg_s"] <= 2.0
    nav.start(1, 0.0, 0.0)
    run_look(nav, 0.0, 0.0, lambda h: prob_for(OPEN, 0.95))       # lab_2 never seen
    assert nav.state == "failed" and "not seen" in nav.msg
    nav.cancel()
    assert nav.state == "failed"                                  # finished states are not overwritten


from real_go2 import nav_sim  # noqa: E402


def test_closed_loop_goes_round_a_desk_to_the_zone():
    """A low desk blocks the walk straight to the zone, but not the view over it (real_go2/nav_sim.py)."""
    desk = nav_sim.box(2.3, 2.7, 1.0, 3.0)
    nav, x, y, clear = nav_sim.run(CLASSES, NAV, desk, [])
    assert nav.state == "arrived", f"ended {nav.state}: {nav.msg} at ({x:.2f}, {y:.2f})"
    assert x > 4.5 and clear > 0.2, f"too close to an obstacle: {clear:.2f} m"


def test_closed_loop_explores_when_the_zone_is_hidden():
    """A tall shelf unit hides the zone from the start: it explores round it and arrives without collisions."""
    shelf = nav_sim.box(2.3, 2.7, 0.8, 3.2)
    nav, x, y, clear = nav_sim.run(CLASSES, NAV, shelf, shelf)
    assert nav.state == "arrived", f"ended {nav.state}: {nav.msg} at ({x:.2f}, {y:.2f})"
    assert nav.explored >= 1 and clear > 0.2, f"explored {nav.explored}, closest {clear:.2f} m"


def test_turn_reaches_heading_despite_turn_deadband():
    """The Go2 barely turns on the spot below ~0.2 rad/s. A purely proportional turn stalls there; the minimum turn
    rate gets it within turn_tolerance_rad, quickly, from any starting error."""
    for err0 in (math.radians(d) for d in (8, 30, -45, 170)):
        nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
        nav.state, nav.target, nav.best_heading = "turn", SHELVES, 0.0
        t, h = 0.0, -err0
        for _ in range(300):
            vx, wz = nav.update(t, h)
            if nav.state != "turn":
                break
            if abs(wz) < 0.2:                       # deadband
                wz = 0.0
            t, h = t + 0.1, wrap(h + wz * 0.1)
        assert nav.state == "walk", f"start error {math.degrees(err0):+.0f} deg: still {nav.state} after {t:.1f} s"
        assert abs(wrap(h)) <= NAV["turn_tolerance_rad"] + 0.05 and t < 10, (math.degrees(h), t)


def test_turn_timeout_looks_again():
    """If the dog cannot turn at all (stuck), the turn gives up and looks around again instead of spinning forever."""
    nav = ZoneNavigator(CLASSES, NAV, have_lidar=True)
    nav.state, nav.target, nav.best_heading = "turn", SHELVES, math.pi / 2
    t = 0.0
    while nav.state == "turn" and t < 60:
        nav.update(t, 0.0)
        t += 0.1
    assert nav.state == "look" and t <= NAV["turn_timeout_s"] + 0.2


def test_closed_loop_with_turn_deadband():
    """Same desk layout, but turns on the spot below 0.2 rad/s do nothing: it still arrives."""
    desk = nav_sim.box(2.3, 2.7, 1.0, 3.0)
    nav, x, y, clear = nav_sim.run(CLASSES, NAV, desk, [], turn_deadband=0.2)
    assert nav.state == "arrived", f"ended {nav.state}: {nav.msg} at ({x:.2f}, {y:.2f})"
    assert clear > 0.2


def test_floor_below_body_is_not_an_obstacle():
    """Body frame as the dog reports it standing (odom cloud via cloud_to_body): floor at about -0.31 m. An open
    floor gives the full lookahead everywhere; a box 1 m ahead is seen only ahead. (The raw rt/utlidar/cloud has
    z pointing down, which put the floor at +0.39 m - inside the corridor - and read 0.05 m in every direction.)"""
    rng = np.random.default_rng(0)
    r, a = rng.uniform(0.37, 3.0, 5000), rng.uniform(0, 2 * math.pi, 5000)
    floor = np.column_stack([r * np.cos(a), r * np.sin(a), rng.normal(-0.33, 0.02, 5000)])
    assert np.all(free_profile(floor, NB, CORRIDOR, LOOK) == LOOK)
    box = np.column_stack([rng.uniform(1.0, 1.2, 200), rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.3, 0.2, 200)])
    p = free_profile(np.vstack([floor, box]), NB, CORRIDOR, LOOK)
    assert abs(p[0] - 1.0) < 0.1 and p[NB // 4] == LOOK and p[NB // 2] == LOOK
    flipped = floor * np.array([1, 1, -1])            # the raw sensor frame's mistake
    assert np.all(free_profile(flipped, NB, CORRIDOR, LOOK) < 0.5)   # blocked in every direction
