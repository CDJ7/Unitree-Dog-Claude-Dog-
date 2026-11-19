"""C5 pieces without the simulator: aim points, merging, measuring a rendered stain, the report text."""
from types import SimpleNamespace as NS

import numpy as np

from analysis.bpa.projection import Intrinsics, look_at
from analysis.closeup import merge_observations, stains_in_frame
from analysis.scene_report import report_text
from navigation.closeups import aim_points


def test_aim_points_cover_the_cluster():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 1.2, 300), rng.uniform(0, 0.6, 300), np.zeros(300)])
    aims = aim_points(pts, radius=0.3, max_aims=12)
    d = np.min(np.linalg.norm(pts[:, None] - np.array(aims)[None], axis=2), axis=1)
    assert 4 <= len(aims) <= 12 and np.percentile(d, 95) < 0.35


def test_merge_keeps_the_most_head_on_view():
    def ob(x, inc, px):
        return NS(position=np.array([x, 0.0, 0.0]), fit=NS(width=0.004), meta={"incidence_deg": inc, "pixels": px})
    a, b, c = ob(0.0, 50, 100), ob(0.001, 10, 80), ob(0.5, 30, 60)
    kept = merge_observations([a, b, c])
    assert len(kept) == 2 and any(k is b for k in kept) and any(k is c for k in kept)


def test_measures_a_rendered_stain():
    # one 60 deg stain on a wall at x = 0, seen head-on from 0.6 m
    from analysis.bpa.projection import Plane
    plane = Plane(origin=np.zeros(3), normal=np.array([1.0, 0, 0]), u=np.array([0, 1.0, 0]), v=np.array([0, 0, 1.0]))
    W = 0.004
    L = W / np.sin(np.radians(60))
    intr = Intrinsics.from_fovy(1920, 1080, 40)
    T = look_at(np.array([0.6, 0, 0]), np.zeros(3))
    yy, zz = np.meshgrid(np.linspace(-0.2, 0.2, 2000), np.linspace(-0.12, 0.12, 1200))
    inside = (yy / (L / 2)) ** 2 + (zz / (W / 2)) ** 2 <= 1
    # render by projecting the ellipse samples into the image
    pts = np.column_stack([np.zeros(inside.sum()), yy[inside], zz[inside]])
    Tcw = np.linalg.inv(T)
    pc = pts @ Tcw[:3, :3].T + Tcw[:3, 3]
    u = np.round(intr.fx * pc[:, 0] / pc[:, 2] + intr.cx).astype(int)
    v = np.round(intr.fy * pc[:, 1] / pc[:, 2] + intr.cy).astype(int)
    rgb = np.full((1080, 1920, 3), 230, np.uint8)
    rgb[v, u] = (110, 8, 12)
    depth = np.full((1080, 1920), 0.6, np.float32)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
    frame = NS(rgb=rgb, depth_m=depth, K=K, T_world_cam=T)
    obs = stains_in_frame(frame, frame_id="t")
    assert len(obs) == 1
    o = obs[0]
    assert abs(o.fit.width - W) < 0.0008 and abs(o.alpha_deg - 60) < 8 and np.linalg.norm(o.position) < 0.003


def test_report_text_groups_by_zone():
    g = [{"id": "g0", "n_stains": 40, "surface": "wall", "zone": "victim_area", "centre": [0.05, 2.5, 1.2],
          "alpha_median_deg": 45.0, "text": "This stain group is consistent with an impact.",
          "origin": {"gravity_corrected": [0.8, 2.4, 1.4], "height_m": 1.4, "ellipsoid_95_semi_axes_m": [0.3, 0.1, 0.1]}}]
    t = report_text(g)
    assert "## victim_area" in t and "area of origin" in t and "1.40 m above the floor" in t
