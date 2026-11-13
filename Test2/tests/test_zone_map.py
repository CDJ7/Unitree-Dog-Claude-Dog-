"""Zone map, localiser and map-based navigation (real_go2/zone_map.py, goto_zone.py map mode), on a synthetic
8 x 8 m lab with four zones (one per quadrant) walked in a grid pattern. No robot, no data_real needed."""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.goto_zone import ZoneNavigator  # noqa: E402
from real_go2.nav_sim import ProbStream, run_map, true_zone  # noqa: E402
from real_go2.zone_map import ZoneLocalizer, ZoneMap, align, apply, map_cfg  # noqa: E402

CFG = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
CLASSES = CFG["classes"]


def lab_session():
    """Grid of walked lines (every 1 m) over 0..8 m; zone = quadrant."""
    pts = []
    for c in np.arange(0.5, 8.0, 1.0):
        line = np.arange(0.3, 7.7, 0.1)
        pts += [np.stack([line, np.full_like(line, c)], 1), np.stack([np.full_like(line, c), line], 1)]
    xy = np.concatenate(pts)
    lab = (xy[:, 0] > 4).astype(int) + 2 * (xy[:, 1] > 4).astype(int)
    return xy, lab


def lab_map():
    xy, lab = lab_session()
    return ZoneMap.from_points(CLASSES, xy, lab)


def noisy_prob(z, rng, right=0.8):
    k = z if rng.random() < right else int(rng.integers(len(CLASSES)))
    p = np.full(len(CLASSES), 0.1 / 3)
    p[k] = 0.9
    return p


def test_align_recovers_rotation_and_shift():
    xy, lab = lab_session()
    th, t = math.radians(70), np.array([3.0, -2.0])
    moved = apply(xy, -th, 0) - t                     # a session recorded in another odometry frame
    sc, th_hat, t_hat = align(moved, lab, xy, lab, len(CLASSES))
    back = apply(moved, th_hat, t_hat)
    assert sc > 0.8                                   # < 1: zone borders are blurred
    assert np.median(np.hypot(*(back - xy).T)) < 0.3


def test_plan_stays_walkable_and_ends_inside_the_target_zone():
    zm = lab_map()
    path = zm.plan((1.5, 1.5), CLASSES.index("shelves"))           # quadrant 3: x > 4, y > 4
    assert path is not None and len(path) > 5
    assert all(zm.walk[zm.cell(p)] for p in path)
    assert zm.zone_at(path[-1]) == 3 and path[-1][0] > 4.5 and path[-1][1] > 4.5


def test_localizer_converges_while_walking():
    zm = lab_map()
    rng = np.random.default_rng(0)
    loc = ZoneLocalizer(zm, map_cfg(CFG["navigation"]))
    th_o, t_o = 2.0, np.array([5.0, 1.0])                          # unknown odometry frame
    route = [(1.5, 1.5), (6.5, 1.5), (6.5, 6.5), (1.5, 6.5)]
    t = 0.0
    for a, b in zip(route, route[1:]):
        for f in np.linspace(0, 1, 60):
            p = np.array(a) + f * (np.array(b) - np.array(a))
            loc.update(apply(p[None], th_o, t_o)[0], 0.0, noisy_prob(true_zone(zm, p), rng), t)
            t += 0.3
    e = loc.estimate()
    assert loc.localized
    assert np.hypot(*(e["xy"] - np.array(route[-1]))) < 1.0
    assert int(np.argmax(loc.zone_belief())) == true_zone(zm, route[-1])


def test_camera_alone_cannot_declare_arrival_with_a_map():
    """The camera says 'target' all round, but the map (localised elsewhere) disagrees: no arrival."""
    zm = lab_map()
    rng = np.random.default_rng(1)
    nav = ZoneNavigator(CLASSES, CFG["navigation"], have_lidar=True, zmap=zm)
    t = 0.0
    # drive by hand lab_1 -> lab_2 -> shelves and back to lab_2 (a straight line alone is ambiguous: any
    # parallel line would fit), then stand at (6.5, 1.0)
    route = [(1.5, 1.5), (6.5, 1.5), (6.5, 6.5), (6.5, 1.0)]
    for a, b in zip(route, route[1:]):
        for f in np.linspace(0, 1, 60):
            p = np.array(a) + f * (np.array(b) - np.array(a))
            nav.update(t, 0.0, noisy_prob(true_zone(zm, p), rng, 0.95), pos=p)
            t += 0.3
    assert nav.loc.localized
    target = CLASSES.index("open_area")                            # quadrant 2: far away
    nav.start(target, t, 0.0)
    h = 0.0
    lie = np.full(len(CLASSES), 0.02)
    lie[target] = 0.94
    for _ in range(400):                                           # one look-around, camera fooled
        vx, wz = nav.update(t, h, lie, np.full(24, 2.5), pos=np.array([6.5, 1.0]))
        if nav.state != "look":
            break
        h, t = h + wz * 0.1, t + 0.1
    assert nav.state != "arrived"
    assert nav.state in ("route", "turn")                          # it plans a route / explores instead


def test_closed_loop_on_the_map_arrives_in_the_right_zone():
    zm = lab_map()
    rng = np.random.default_rng(2)
    pools = {}
    for z in range(len(CLASSES)):
        pools[z] = np.array([noisy_prob(z, rng, 0.8) for _ in range(300)])
    target = CLASSES.index("shelves")
    nav, xy, secs, zend, closest = run_map(zm, CLASSES, CFG["navigation"], ProbStream(pools, rng),
                                           (1.5, 1.5, 0.0), target, (1.0, np.array([2.0, -3.0])), steps=6000)
    assert nav.state == "arrived", nav.msg
    assert zend == target
