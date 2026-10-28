"""Part 1: geometry, flight, stain shape and ground-truth generation."""
import json

import numpy as np
import pytest

from sim_env.spatter import generate, load_scenario
from sim_env.spatter.constants import G
from sim_env.spatter.geometry import Rect, SurfaceSet, box_surfaces, room_surfaces
from sim_env.spatter.physics import Launch, fly
from sim_env.spatter.raster import decal_patch, mask_points, stain_bounds, stain_mask
from sim_env.spatter.stains import impact_geometry, make_stain
from sim_env.scenarios import analytic_paths as scenario_paths
from sim_env.scenarios.random_scenario import random_scenario


def test_room_normals_point_inward():
    rects = room_surfaces((4, 3, 2.5))
    centre = np.array([2, 1.5, 1.25])
    for r in rects:
        assert (centre - r.center) @ r.normal > 0, r.id
        assert np.allclose(np.cross(r.u, r.v), r.normal)


def test_box_normals_point_outward():
    c = np.array([1.0, 1.0, 0.4])
    for r in box_surfaces("b", c, (1.0, 0.6, 0.8), yaw_deg=30):
        assert (r.center - c) @ r.normal > 0, r.id


def test_segment_hits_nearest_surface():
    s = SurfaceSet(room_surfaces((4, 3, 2.5)) + box_surfaces("b", (2, 1.5, 0.4), (1, 1, 0.8)))
    hit = s.first_hit(np.array([2.0, 1.5, 2.0]), np.array([2.0, 1.5, -1.0]))
    assert hit.surface.id == "b_top"
    assert np.allclose(hit.point, [2.0, 1.5, 0.8])
    assert s.first_hit(np.array([1.0, 1.0, 1.0]), np.array([1.1, 1.0, 1.0])) is None


def test_flight_without_drag_matches_projectile_formula():
    floor = SurfaceSet([Rect.from_normal("f", "floor", (0, 0, 0), (0, 0, 1), 50, 50)])
    h, vx = 1.2, 3.0
    fl = fly([Launch(np.array([0, 0, h]), np.array([vx, 0, 0]), 2e-3)], floor, drag=False)[0]
    t = np.sqrt(2 * h / G)
    assert fl.landed
    assert fl.pos[0] == pytest.approx(vx * t, abs=2e-3)
    assert fl.flight_time == pytest.approx(t, abs=2e-3)
    assert np.degrees(np.arctan2(-fl.vel[2], fl.vel[0])) == pytest.approx(np.degrees(np.arctan(G * t / vx)), abs=0.2)


def test_drag_shortens_range_more_for_small_drops():
    floor = SurfaceSet([Rect.from_normal("f", "floor", (0, 0, 0), (0, 0, 1), 50, 50)])
    ranges = [fly([Launch(np.array([0, 0, 1.0]), np.array([6.0, 0, 0]), d)], floor)[0].pos[0]
              for d in (0.5e-3, 2e-3, 5e-3)]
    nodrag = fly([Launch(np.array([0, 0, 1.0]), np.array([6.0, 0, 0]), 2e-3)], floor, drag=False)[0].pos[0]
    assert ranges[0] < ranges[1] < ranges[2] < nodrag


@pytest.mark.parametrize("alpha", [10, 30, 45, 60, 85])
def test_stain_shape_follows_impact_angle(alpha):
    rect = Rect.from_normal("w", "wall", (0, 0, 1), (1, 0, 0), 2, 1)       # wall facing +x
    a = np.radians(alpha)
    vel = 4.0 * np.array([-np.sin(a), np.cos(a), 0.0])                   # into the wall, travelling +y
    st = make_stain("s", rect, 0, "d", rect.center, vel, 2e-3, np.random.default_rng(0))
    assert st.impact_angle_deg == pytest.approx(alpha, abs=1e-6)
    assert st.width / st.length == pytest.approx(np.sin(a), rel=1e-6)
    assert np.allclose(st.dir_world, [0, 1, 0])
    # gamma is measured in the wall's own (u, v) frame
    g = np.radians(st.gamma_deg)
    assert np.allclose(np.cos(g) * rect.u + np.sin(g) * rect.v, st.dir_world)


def test_impact_geometry_straight_on_has_no_direction():
    alpha, d, vn = impact_geometry([0, 0, -3.0], [0, 0, 1.0])
    assert alpha == pytest.approx(90.0)
    assert np.allclose(d, 0)
    assert vn == pytest.approx(3.0)


def test_raster_mask_area_matches_ellipse():
    st = {"center_uv": [0.0, 0.0], "gamma_deg": 30.0, "length": 6e-3, "width": 3e-3, "tail_length": 0.0}
    m, o, res = stain_mask(st, res=5e-5)
    area = m.sum() * res ** 2
    assert area == pytest.approx(np.pi * 3e-3 * 1.5e-3, rel=0.02)
    pts = mask_points(m, o, res)
    assert np.allclose(pts.mean(0), 0, atol=res)


@pytest.mark.parametrize("path", scenario_paths(), ids=lambda p: p.stem)
def test_scenarios_generate_with_full_ground_truth(path, tmp_path):
    scn = load_scenario(path)
    gt = generate(scn)
    assert gt["scenario_id"] == scn["id"]
    assert gt["primary_mechanism"] == scn["events"][0]["mechanism"]
    surf = {s["id"]: s for s in gt["surfaces"]}
    flying = [d for d in gt["droplets"]]
    assert flying or gt["smears"]
    assert all(d["landed"] for d in flying)
    for d in flying:
        assert d["surface_id"] in surf
        assert 0 <= d["impact_angle_deg"] <= 90
    for st in gt["stains"]:
        assert st["width"] <= st["length"] + 1e-12
    # JSON round trip and reproducibility from the seed
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    again = generate(scn)
    assert json.loads(p.read_text())["stains"] == json.loads(json.dumps(again))["stains"]


def test_castoff_ground_truth_has_swing_path():
    gt = generate(load_scenario([p for p in scenario_paths() if "s4_" in p.name][0]))
    ev = gt["events"][0]
    assert ev["mechanism"] == "castoff"
    assert len(ev["swing_path"]) > 10
    pivot = np.array(ev["origin"])
    r = np.linalg.norm(np.array(ev["swing_path"]) - pivot, axis=1)
    assert np.allclose(r, ev["params"]["radius"])


def test_random_scenarios_are_seeded_and_land():
    a, b = random_scenario(11), random_scenario(11)
    assert a == b
    for seed in range(12):
        gt = generate(random_scenario(seed))
        assert all(d["landed"] for d in gt["droplets"])


def test_decal_patch_has_blood_pixels():
    gt = generate(load_scenario(scenario_paths()[0]))
    wall = [s for s in gt["stains"] if s["surface_id"] == "room_wall_x1"]
    img = decal_patch(wall[:40], [], stain_bounds(wall[:40]), res=1e-3)
    assert img.shape[2] == 4
    assert (img[..., 3] > 0).sum() > 40
