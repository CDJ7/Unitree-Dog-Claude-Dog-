import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mapping.object_map import ObjectMap, evaluate  # noqa: E402
from mapping.observations import usable_observations  # noqa: E402
from mapping.real_adapter import body_to_world  # noqa: E402

CFG = {"merge_dist_m": 0.5, "min_observations": 3, "max_obs_range_m": 4.0, "match_dist_m": 1.0,
       "use_center_correction": False, "class_half_depth_m": {"chair": 0.2}, "default_half_depth_m": 0.1}


def test_same_class_nearby_sightings_merge_and_far_ones_do_not():
    m = ObjectMap(CFG)
    for _ in range(3):
        m.add("chair", [1.0, 1.0, 0.4])
    m.add("chair", [1.2, 1.1, 0.4])
    for _ in range(3):
        m.add("chair", [3.0, -1.0, 0.4])
    assert len(m.objects()) == 2


def test_different_classes_never_merge():
    m = ObjectMap(dict(CFG, min_observations=1))
    m.add("chair", [1, 1, 0.4])
    m.add("bottle", [1, 1, 0.4])
    assert len(m.objects()) == 2


def test_min_observations_filters_one_off_outliers():
    m = ObjectMap(CFG)
    m.add("chair", [1, 1, 0.4])
    assert m.objects() == []
    for _ in range(2):
        m.add("chair", [1, 1, 0.4])
    assert len(m.objects()) == 1


def test_position_is_a_median_so_a_wild_sighting_does_not_move_it():
    m = ObjectMap(CFG)
    for x in (1.0, 1.01, 0.99, 1.0, 1.02):
        m.add("chair", [x, 0, 0.4])
    m.add("chair", [1.4, 0, 0.4])                     # within merge distance but an outlier
    assert m.objects()[0].position()[0] == pytest.approx(1.005, abs=0.02)


def test_far_sightings_are_ignored():
    m = ObjectMap(dict(CFG, min_observations=1))
    assert m.add("chair", [9, 0, 0.4], range_m=9.0) is None
    assert m.objects() == []


def test_center_correction_pushes_away_from_the_sensor():
    m = ObjectMap(dict(CFG, use_center_correction=True, min_observations=1))
    m.add("chair", [2.0, 0.0, 0.4], ray_origin=[0.0, 0.0, 0.3])
    o = m.objects()[0]
    assert o.position()[0] == pytest.approx(2.2)
    assert o.position(corrected=False)[0] == pytest.approx(2.0)
    assert o.position()[2] == pytest.approx(0.4)      # push is horizontal only


def test_evaluate_matches_per_class_and_reports_misses_and_false_positives():
    m = ObjectMap(dict(CFG, min_observations=1))
    m.add("chair", [1.1, 0.0, 0.4])                  # true chair at (1,0)
    m.add("bottle", [5.0, 5.0, 0.2])                 # phantom bottle
    truth = [("obj_chair_1", "chair", [1.0, 0.0, 0.4]), ("obj_cup_1", "cup", [0.0, 0.0, 0.1])]
    ev = evaluate(m, truth)
    rows = {r["object"]: r for r in ev["per_object"]}
    assert rows["obj_chair_1"]["detected"] and rows["obj_chair_1"]["error_3d_m"] == pytest.approx(0.1, abs=1e-3)
    assert not rows["obj_cup_1"]["detected"]
    assert ev["summary"]["n_false_positives"] == 1 and ev["summary"]["recall"] == 0.5


def test_evaluate_two_true_objects_do_not_share_one_estimate():
    m = ObjectMap(dict(CFG, min_observations=1))
    m.add("chair", [1.0, 0.0, 0.4])
    ev = evaluate(m, [("a", "chair", [1.0, 0.0, 0.4]), ("b", "chair", [1.3, 0.0, 0.4])])
    assert sum(r["detected"] for r in ev["per_object"]) == 1


def _det(box, xyz=(1, 0, 0.4), r=1.0):
    return NS(bbox=box, xyz_world=xyz, distance_m=r)


def test_usable_observations_drops_truncated_boxes():
    ok = _det((50, 50, 100, 100))
    cut = _det((0, 50, 60, 100))
    assert usable_observations([ok, cut], (180, 320)) == [ok]


def test_usable_observations_drops_the_occluded_one_not_the_occluder():
    far = _det((100, 60, 180, 100))
    near = _det((110, 70, 170, 140))                  # lower bottom edge = nearer, covers most of `far`
    kept = usable_observations([far, near], (180, 320))
    assert kept == [near]


def test_usable_observations_keeps_a_small_object_in_front_of_a_big_one():
    big = _det((50, 40, 250, 120))
    small = _det((140, 100, 160, 130))                # sits in front (bottom edge lower), inside big's box
    kept = usable_observations([big, small], (180, 320))
    assert small in kept and big in kept


def test_usable_observations_skips_detections_without_3d():
    d = NS(bbox=(50, 50, 100, 100), xyz_world=None, distance_m=None)
    assert usable_observations([d], (180, 320)) == []


def test_body_to_world_applies_pose():
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # yaw +90 deg
    p = body_to_world([2.0, 0.0, 0.3], [1.0, 1.0, 0.0], q)
    assert np.allclose(p, [1.0, 3.0, 0.3], atol=1e-9)
