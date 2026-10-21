"""Part 4: BPA math on synthetic droplets with known angles and origins."""
import numpy as np
import pytest

from analysis.bpa.convergence import intersect_lines
from analysis.bpa.ellipse import fit_stain, usable
from analysis.bpa.measure import from_camera_mask, from_uv_points, group_observations
from analysis.bpa.origin import (MeasuredStain, area_of_convergence, gravity_corrected_origin,
                                 straight_line_origin, tangent_method)
from analysis.bpa.projection import Intrinsics, Plane, deproject, fit_plane, look_at, project, to_world
from analysis.bpa.synthetic import render_plane_view
from analysis.classifier.features import FEATURE_NAMES, group_features
from analysis.classifier.inference import explain, weapon_probabilities
from analysis.classifier.model import MechanismClassifier
from sim_env.spatter.geometry import Rect, SurfaceSet, room_surfaces
from sim_env.spatter.physics import Launch, fly
from sim_env.spatter.raster import mask_points, stain_mask
from sim_env.spatter.stains import make_stain

FLOOR = Plane(np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))


def straight_droplets(origin, n=30, seed=0, radius=(0.4, 1.5)):
    """Stains on the floor from straight (gravity-free) paths out of `origin`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(*radius)
        p = np.array([origin[0] + r * np.cos(ang), origin[1] + r * np.sin(ang), 0.0])
        t = p - origin
        t /= np.linalg.norm(t)
        alpha = np.degrees(np.arcsin(-t[2]))
        d = np.r_[t[:2], 0.0]
        d /= np.linalg.norm(d)
        out.append(MeasuredStain(p, np.array([0, 0, 1.0]), d, alpha, 3e-3, "floor", f"s{i}"))
    return out


# ------------------------------------------------------------------ ellipse

@pytest.mark.parametrize("alpha", [15, 30, 45, 55])
@pytest.mark.parametrize("gamma", [-120, 0, 75])
def test_ellipse_fit_recovers_angle_and_direction(alpha, gamma):
    W = 3e-3
    L = W / np.sin(np.radians(alpha))
    st = {"center_uv": [0.01, -0.02], "gamma_deg": gamma, "length": L, "width": W,
          "tail_length": 0.4 * L * np.cos(np.radians(alpha))}
    m, o, res = stain_mask(st, res=1e-4)
    f = fit_stain(mask_points(m, o, res), res)
    assert f.impact_angle_deg == pytest.approx(alpha, abs=2.5)
    assert abs((f.gamma_deg - gamma + 180) % 360 - 180) < 3
    assert np.allclose(f.center, st["center_uv"], atol=3e-4)        # centre of the body, not the tail
    assert usable(f)[0]


def test_round_and_tiny_stains_are_rejected():
    m, o, res = stain_mask({"center_uv": [0, 0], "gamma_deg": 0, "length": 4e-3, "width": 3.9e-3}, res=1e-4)
    assert usable(fit_stain(mask_points(m, o, res), res)) == (False, "too round")
    m, o, res = stain_mask({"center_uv": [0, 0], "gamma_deg": 0, "length": 1.2e-3, "width": 0.5e-3,
                            "tail_length": 3e-4}, res=1e-4)
    assert usable(fit_stain(mask_points(m, o, res), res)) == (False, "too small")


# ------------------------------------------------------------------ lines

def test_line_intersection_exact_and_robust():
    rng = np.random.default_rng(1)
    x = np.array([0.3, -0.2, 1.1])
    P = rng.normal(size=(20, 3))
    D = P - x                                   # travelling away from x
    fit = intersect_lines(P, D)
    assert np.allclose(fit.point, x, atol=1e-9)
    assert fit.behind.all()
    D[:3] = rng.normal(size=(3, 3))             # three wrong directions
    assert np.linalg.norm(intersect_lines(P, D).point - x) < 0.05


def test_area_of_convergence_on_floor():
    o = np.array([1.0, 2.0, 1.2])
    stains = straight_droplets(o)
    conv = area_of_convergence(stains, FLOOR, bootstrap=50)
    assert np.allclose(conv.point_world[:2], o[:2], atol=1e-6)
    assert conv.semi_axes.max() < 1e-3


# ------------------------------------------------------------------ origin

def test_straight_line_and_tangent_method_exact_without_gravity():
    o = np.array([1.0, 2.0, 1.2])
    stains = straight_droplets(o)
    est = straight_line_origin(stains, bootstrap=50)
    assert np.allclose(est.point, o, atol=1e-6)
    p, heights, _ = tangent_method(stains, FLOOR)
    assert np.allclose(p, o, atol=1e-6)
    assert np.allclose(heights, o[2], atol=1e-6)


def test_origin_across_two_surfaces():
    """Stains on a floor and a wall from one source: the 3D method uses both."""
    room = SurfaceSet(room_surfaces((4, 4, 3)))
    o = np.array([3.0, 2.0, 1.0])
    rng = np.random.default_rng(2)
    stains = []
    for i in range(40):
        d = rng.normal(size=3)
        d[0] = abs(d[0]) + 0.3                   # toward the x = 4 wall and the floor
        hit = room.first_hit(o, o + 10 * d / np.linalg.norm(d))
        r = hit.surface
        st = make_stain("s", r, 0, "d", hit.point, d / np.linalg.norm(d) * 3, 2e-3, rng)
        if st.impact_angle_deg > 80:
            continue
        stains.append(MeasuredStain(hit.point, r.normal, np.array(st.dir_world), st.impact_angle_deg, 3e-3, r.id))
    assert len({s.surface_id for s in stains}) >= 2
    assert np.allclose(straight_line_origin(stains, bootstrap=0).point, o, atol=1e-6)


def test_gravity_correction_beats_straight_lines_on_ballistic_droplets():
    room = SurfaceSet(room_surfaces((5, 5, 3)))
    o = np.array([2.5, 2.5, 1.3])
    rng = np.random.default_rng(3)
    launches = []
    for _ in range(80):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) * 0.3
        launches.append(Launch(o, d / np.linalg.norm(d) * rng.uniform(2.5, 5.5), 2e-3))
    stains = []
    for l, fl in zip(launches, fly(launches, room)):
        r = room.rects[fl.surface_index]
        st = make_stain("s", r, 0, "d", fl.pos, fl.vel, l.diameter, rng)
        if st.impact_angle_deg < 64:
            n = r.normal if fl.vel @ r.normal < 0 else -r.normal
            stains.append(MeasuredStain(fl.pos, n, np.array(st.dir_world), st.impact_angle_deg, st.width, r.id))
    sl = straight_line_origin(stains, bootstrap=0)
    gc = gravity_corrected_origin(stains, start=sl.point, speed_prior=(2.0, 4.0, 6.0), bootstrap=10)
    e_sl, e_gc = np.linalg.norm(sl.point - o), np.linalg.norm(gc.point - o)
    assert sl.point[2] > o[2] + 0.1             # the textbook bias: straight lines put it too high
    assert e_gc < 0.5 * e_sl
    assert e_gc < 0.2
    d = gc.point - o
    assert d @ np.linalg.solve(gc.cov, d) < 7.815           # inside the reported 95% ellipsoid


# ------------------------------------------------------------------ camera path

def test_project_deproject_round_trip():
    intr = Intrinsics.from_fovy(640, 480, 60)
    T = look_at([0.0, -1.0, 1.0], [0.2, 0.5, 0.3])
    pts = np.random.default_rng(4).uniform([-0.5, 0.3, 0], [0.8, 1.2, 0.8], (50, 3))
    u, v, z = project(pts, T, intr)
    back = to_world(deproject(u, v, z, intr), T)
    assert np.allclose(back, pts, atol=1e-9)


def test_fit_plane_orients_toward_camera():
    pts = np.c_[np.random.default_rng(5).uniform(-1, 1, (100, 2)), np.zeros(100)]
    pl = fit_plane(pts, toward=[0, 0, -2])
    assert np.allclose(pl.normal, [0, 0, -1])
    assert pl.rms < 1e-9


@pytest.mark.parametrize("incidence", [0, 30])
def test_camera_mask_measurement(incidence):
    wall = Rect.from_normal("w", "wall", (5, 2, 1.3), (-1, 0, 0), 2, 1.3)
    intr = Intrinsics.from_fovy(1920, 1080, 40)
    alpha, gamma, W = 35.0, 60.0, 3.5e-3
    L = W / np.sin(np.radians(alpha))
    st = {"center_uv": [0.0, 0.0], "gamma_deg": gamma, "length": L, "width": W,
          "tail_length": 0.4 * L * np.cos(np.radians(alpha))}
    target = wall.to_world([0, 0])
    inc = np.radians(incidence)
    T = look_at(target + 0.45 * (np.cos(inc) * wall.normal + np.sin(inc) * wall.u), target)
    labels, depth = render_plane_view(wall, [st], intr, T)
    obs = from_camera_mask("x", labels == 0, depth, intr, T)
    assert obs.surface_kind == "wall"
    assert np.allclose(obs.plane.normal, wall.normal, atol=1e-6)
    assert obs.alpha_deg == pytest.approx(alpha, abs=4)
    true_dir = np.cos(np.radians(gamma)) * wall.u + np.sin(np.radians(gamma)) * wall.v
    assert obs.direction @ true_dir > np.cos(np.radians(6))
    assert np.linalg.norm(obs.position - target) < 1.5e-3


# ------------------------------------------------------------------ groups, classifier, wording

def _obs_from_stains(stains, plane):
    out = []
    for i, s in enumerate(stains):
        m, o, res = stain_mask(s, res=1e-4)
        out.append(from_uv_points(f"o{i}", mask_points(m, o, res), res, plane, "floor", "floor"))
    return out


def test_grouping_and_features():
    rng = np.random.default_rng(6)
    a = [{"center_uv": [rng.uniform(0, 0.3), rng.uniform(0, 0.3)], "gamma_deg": rng.uniform(-180, 180),
          "length": 5e-3, "width": 3e-3, "tail_length": 1e-3} for _ in range(12)]
    b = [{"center_uv": [2 + rng.uniform(0, 0.3), rng.uniform(0, 0.3)], "gamma_deg": 0.0,
          "length": 5e-3, "width": 3e-3, "tail_length": 1e-3} for _ in range(8)]
    groups = group_observations(_obs_from_stains(a + b, FLOOR), link=0.3)
    assert sorted(len(g) for g in groups) == [8, 12]
    f = group_features(groups[0])
    assert set(f) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())
    assert f["frac_floor"] == 1.0


def test_inference_wording_and_weapon_probs():
    probs = {"impact": 0.85, "castoff": 0.1, "projected": 0.02, "drip": 0.02, "transfer": 0.01}
    wp = weapon_probabilities(probs)
    assert abs(sum(wp.values()) - 1) < 1e-9
    assert max(wp, key=wp.get) == "blunt"
    feats = {k: 0.5 for k in FEATURE_NAMES}
    out = explain(probs, feats)
    assert "consistent with" in out["text"]
    assert "identif" in out["text"]           # says these are not identifications
    assert out["mechanism"] == "impact"
    assert out["evidence"]


def test_classifier_fallback_without_model(tmp_path):
    clf = MechanismClassifier.load(tmp_path / "missing.pkl")
    assert not clf.is_trained
    p = clf.predict_proba_features({k: 0.0 for k in FEATURE_NAMES})
    assert abs(sum(p.values()) - 1) < 1e-9
