"""A4: occupancy grid, planning, frontiers, safety layer (pure), and short simulator runs."""
import numpy as np
import pytest

import sim_env  # noqa: F401
from navigation.frontier import choose_frontier, frontier_cells
from navigation.local import SafetyConfig, SafetyLayer
from navigation.occupancy import OccupancyGrid
from navigation.planner import CostField, simplify
from sim_env.interfaces import LidarScan, RobotState, ImuReading, pose_matrix
from sim_env.runtime import SIM_ROOT

needs_sim = pytest.mark.skipif(not (SIM_ROOT / "sim/native/controller.py").exists(),
                               reason="simulator submodule not fetched")


def box_room_scan(pos, size=(4.0, 3.0), n=720, wall_xy=None, low_pts=()):
    """Planar scan from `pos` inside an axis-aligned box room [0,W]x[0,H]."""
    W, H = size
    ang = -np.pi + np.arange(n) * 2 * np.pi / n
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[:, 0] > 0, (W - pos[0]) / d[:, 0], -pos[0] / d[:, 0])
        ty = np.where(d[:, 1] > 0, (H - pos[1]) / d[:, 1], -pos[1] / d[:, 1])
    r = np.minimum(np.abs(tx), np.abs(ty)).astype(np.float32)
    T = pose_matrix([pos[0], pos[1], 0.45], np.eye(3))
    pts = np.array([[x - pos[0], y - pos[1], z - 0.45] for x, y, z in low_pts], np.float32).reshape(-1, 3)
    return LidarScan(0.0, r, -np.pi, 2 * np.pi / n, 0.1, 30.0, pts, T)


def near(mask, g, xy):
    r, c = g.to_cell(xy)
    return bool(mask[r - 1:r + 2, c - 1:c + 2].any())


def state(x, y, yaw=0.0):
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return RobotState(0, np.array([x, y, 0.29]), q, yaw, np.zeros(3), np.zeros(3), np.zeros(3),
                      ImuReading(0, np.zeros(3), np.array([0, 0, 9.81]), q), 1.0)


def test_grid_maps_walls_free_space_and_low_obstacles():
    g = OccupancyGrid(origin=(-1, -1), size=(6, 5))
    box = [(2.5, 1.5, z) for z in (0.1, 0.2, 0.3)]               # a low box the scan plane misses
    for _ in range(3):
        g.update(box_room_scan((1.0, 1.0), low_pts=box))
    occ, free = g.occupied(), g.free()
    for xy in [(4.0, 1.0), (0.0, 2.0), (2.0, 3.0)]:                # walls
        assert near(occ, g, xy)
    assert free[g.to_cell((2.0, 1.2))] and free[g.to_cell((1.0, 2.5))]
    assert near(occ, g, (2.5, 1.5))                               # low obstacle survives scan carving
    assert g.unknown()[g.to_cell((4.7, 1.0))]                     # outside the room


def test_planner_goes_around_an_obstacle():
    g = OccupancyGrid(origin=(-1, -1), size=(6, 5))
    wall = [(2.0, y, 0.2) for y in np.arange(0.0, 2.2, 0.03)]     # partition with a gap at the top
    for _ in range(3):
        g.update(box_room_scan((0.8, 1.0), low_pts=wall))
        g.update(box_room_scan((3.2, 1.0), low_pts=wall))
    f = CostField(g, g.costmap(robot_radius=0.24), (0.8, 1.0))
    path = simplify(f.path_to(g.to_cell((3.2, 1.0))))
    assert path is not None
    assert path[:, 1].max() > 2.2                                 # through the gap
    dist = g.obstacle_distance()
    r, c = g.to_cell(path[1:-1])
    assert dist[r, c].min() >= 0.24 - 0.05


def test_frontier_is_at_the_edge_of_the_known_area():
    g = OccupancyGrid(origin=(-1, -1), size=(8, 5))
    scan = box_room_scan((1.0, 1.0), size=(6.0, 3.0))
    scan.ranges = np.minimum(scan.ranges, 2.0)                    # short range: the far end stays unknown
    scan.ranges[scan.ranges >= 2.0] = np.inf
    g.update(scan, max_range=2.0)
    assert frontier_cells(g).any()
    f = CostField(g, g.costmap(robot_radius=0.24), (1.0, 1.0))
    goal, gxy, _ = choose_frontier(g, f)
    assert gxy[0] > 2.0                                           # toward the unexplored east side


def test_safety_layer_stops_before_a_wall_and_blocks_corner_sweep():
    s = SafetyLayer(SafetyConfig())
    s.observe(box_room_scan((3.0, 1.5)))                          # wall at x=4: 1.0 m ahead
    vx, _, _ = s.filter(state(3.0, 1.5), 0.3, 0, 0)
    assert vx == pytest.approx(0.3)
    s.observe(box_room_scan((3.5, 1.5)))                          # 0.5 m: front clearance 0.15 < stop
    vx, _, _ = s.filter(state(3.5, 1.5), 0.3, 0, 0)
    assert vx == 0.0
    s.observe(box_room_scan((2.0, 0.23)))                         # 0.23 m from a side wall: a turn's corners hit it
    _, _, wz = s.filter(state(2.0, 0.23), 0.0, 0, 0.5)
    assert wz == 0.0 and s.turn_blocked
    man = s.free_maneuver(state(2.0, 0.23), +1)
    assert man is not None and man[1] > 0                         # steps away from the wall (+y) first


@needs_sim
def test_escapes_a_corner_and_reaches_a_goal_without_collisions():
    from navigation.explorer import Navigator
    from sim_env.scenario_scene import load_scenario_config, open_scenario
    sim, gt = open_scenario(load_scenario_config(5), spawn=(5.25, 0.42, 0.0))
    try:
        sim.settle(1.0)
        nav = Navigator(sim, verbose=False)
        for _ in range(int(2 * np.pi / 0.5 / 0.1)):
            nav.tick(0, 0, 0.5)
        assert nav.go_to((3.5, 1.6), timeout=45)
        assert not nav.log.collisions
    finally:
        sim.close()


@needs_sim
def test_inspect_point_captures_views_of_the_stains():
    from navigation.explorer import Navigator
    from navigation.inspect import inspect_point, inspection_targets
    from sim_env.scenario_scene import load_scenario_config, open_scenario
    sim, gt = open_scenario(load_scenario_config(5), spawn=(3.3, 1.6, -1.6))
    try:
        sim.settle(1.0)
        nav = Navigator(sim, verbose=False)
        for _ in range(int(2 * np.pi / 0.5 / 0.1)):
            nav.tick(0, 0, 0.5)
        t = inspection_targets(gt)[0]                              # the impact pattern on the south wall
        res = inspect_point(nav, t["center"], t["normal"], gt, t["stain_ids"], n_views=2)
        assert len(res["views"]) >= 1
        assert max(v["visible_fraction_closeup"] for v in res["views"]) > 0.3
        assert not nav.log.collisions
    finally:
        sim.close()
