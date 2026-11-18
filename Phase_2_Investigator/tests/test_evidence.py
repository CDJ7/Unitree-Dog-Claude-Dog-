"""Evidence map (C3) and evidence-aware planning (C4) without the simulator."""
from types import SimpleNamespace as NS

import numpy as np

from mapping.evidence_map import EvidenceMap, floor_stains_stepped_on, score
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
