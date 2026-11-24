"""Evidence map (C3) and evidence-aware planning (C4) without the simulator."""
from types import SimpleNamespace as NS

import numpy as np

from mapping.evidence_map import EvidenceMap, floor_stains_stepped_on, score, stepped_after_found
from navigation.occupancy import OccupancyGrid
from navigation.planner import CostField


def det(xyz, normal=(0, 0, 1), prob=0.9, area=0.01):
    return NS(xyz=np.array(xyz, float), normal=np.array(normal, float), prob=prob, feats={"log_area_m2": np.log(area)})


def test_clusters_confirm_only_repeated_detections():
    ev = EvidenceMap()
    for k in range(3):                                     # a floor pool seen in 3 frames, slightly different spots
        ev.add_frame([det((1.0 + 0.05 * k, 2.0, 0.0)), det((3.0, 0.0, 1.2), normal=(1, 0, 0))],
                     zone_of=lambda xy: "lab_1" if xy[0] < 2 else "lab_2")
    ev.add_frame([det((5.0, 5.0, 0.45), prob=0.7)])        # seen once: a red object on a table?
    s = {c["id"]: c for c in ev.summary()}
    assert len(ev.clusters) == 3 and len(ev.confirmed()) == 2
    floor = [c for c in s.values() if c["surface"] == "floor"][0]
    wall = [c for c in s.values() if c["surface"] == "wall"][0]
    assert floor["views"] == 3 and floor["zone"] == "lab_1" and wall["zone"] == "lab_2"
    assert not [c for c in s.values() if c["surface"] == "raised"][0]["confirmed"]
    assert len(ev.floor_points()) == 1                     # only floor evidence is kept out of


def test_score_against_ground_truth():
    ev = EvidenceMap(min_views=1)
    ev.add_frame([det((1.0, 2.0, 0.0)), det((4.0, 4.0, 0.0))])
    gt = {"clusters": [{"id": "c0", "n_stains": 10, "center": [1.1, 2.0, 0.0], "bbox_min": [0.8, 1.8, 0], "bbox_max": [1.4, 2.2, 0]},
                       {"id": "c1", "n_stains": 10, "center": [3.0, 0.0, 1.0], "bbox_min": [3, -.2, .8], "bbox_max": [3, .2, 1.2]}],
          "distractors": [{"id": "red_mug", "center": [4.0, 4.0, 0.0]}]}
    s = score(ev, gt)
    assert s["recall"] == 0.5 and s["precision"] == 0.5 and s["truth_missed"] == ["c1"]
    assert s["distractors_flagged"] == 1


def test_keepout_makes_paths_go_around_floor_evidence():
    g = OccupancyGrid(origin=(0, 0), size=(6.0, 4.0), res=0.05)
    g.seen[:] = True
    g.logodds[:] = -2.0
    start, goal = (0.5, 2.0), (5.5, 2.0)
    straight = CostField(g, g.costmap(robot_radius=0.3), start).path_to(g.to_cell(goal))
    assert np.abs(straight[:, 1] - 2.0).max() < 0.1
    g.set_keepout([((3.0, 2.0), 0.5)])                     # a pool in the middle of the way
    around = CostField(g, g.costmap(robot_radius=0.3), start).path_to(g.to_cell(goal))
    d = np.linalg.norm(around - (3.0, 2.0), axis=1)
    assert d.min() > 0.5 + 0.25                            # pool radius + body clearance


def test_stepped_on_counts_stains_under_the_body():
    gt = {"stains": [{"id": "a", "center_world": [1.0, 0.0, 0.0]}, {"id": "b", "center_world": [1.0, 0.5, 0.0]},
                     {"id": "w", "center_world": [1.0, 0.0, 1.0]}]}
    path = [(0, x, 0.0, 0.0) for x in np.arange(0, 2, 0.1)]
    assert floor_stains_stepped_on(gt, path) == ["a"]


def test_stepped_after_found_only_counts_known_evidence():
    ev = EvidenceMap()
    ev.add_frame([det((1.0, 0.0, 0.0))], t=5.0)
    ev.add_frame([det((1.0, 0.0, 0.0))], t=6.0)              # confirmed at t = 6
    gt = {"stains": [{"id": "a", "center_world": [1.0, 0.0, 0.0]}, {"id": "b", "center_world": [3.0, 0.0, 0.0]}]}
    path = [(t, x, 0.0, 0.0) for t, x in zip(np.arange(0, 20, 1.0), np.arange(0, 4, 0.2))]
    stepped = floor_stains_stepped_on(gt, path, times=True)
    assert set(stepped) == {"a", "b"} and stepped["a"][0] < 6.0  # "a" was walked over before it was found
    assert stepped_after_found(ev, stepped) == []
    late = floor_stains_stepped_on(gt, [(t + 10, x, y, yaw) for t, x, y, yaw in path], times=True)
    assert stepped_after_found(ev, late) == ["a"]


def test_safety_does_not_freeze_on_a_point_inside_the_body():
    from navigation.local import SafetyLayer
    sl = SafetyLayer()
    inside = np.array([[-0.34, 0.0]])                        # a return grazing the rear of the body
    assert not sl.motion_hits(inside, 0.15, 0.0)            # driving forward is still allowed
    ahead = np.array([[0.45, 0.0]])
    assert sl.motion_hits(ahead, 0.3, 0.0, horizon=0.5)      # a real obstacle ahead still counts


def test_escape_from_a_corner_uses_a_move_the_filter_allows():
    """S5 wedge: wall 0.18 m ahead of the nose, wall behind-right; turning blocked. The only real escape is a
    sideways step to the free left side (forward arcs would be zeroed by the stop distance)."""
    from navigation.local import SafetyLayer
    sl = SafetyLayer()
    ys = np.linspace(-0.6, 0.6, 61)
    ahead = np.column_stack([np.full(61, 0.53), ys])                     # nose (0.35) + 0.18
    t = np.linspace(-2.9, -1.9, 40)
    behind_right = np.column_stack([0.42 * np.cos(t), 0.42 * np.sin(t)])
    pts = np.vstack([ahead, behind_right])
    sl._points_robot = lambda st, mp=None: pts
    st = NS(yaw=0.0, position=np.zeros(3))
    vx, vy, wz = sl.free_maneuver(st, +1)
    assert vx <= 0.0 and vy > 0                                         # not forward into the wall
