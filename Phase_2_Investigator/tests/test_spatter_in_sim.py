"""A2: droplets land on MuJoCo geometry exactly as on the analytic room, and the
rendered stains have the ellipse shape their impact angle predicts.

Uses small self-contained MuJoCo scenes (no simulator submodule needed).
"""
import numpy as np
import pytest

import sim_env  # noqa: F401  (offscreen GL backend)
import mujoco

from sim_env.scenario_scene import stain_geoms
from sim_env.spatter import generate
from sim_env.spatter.geometry import SurfaceSet, box_surfaces, room_surfaces
from sim_env.spatter.mj_surfaces import MjSurfaces

ROOM = (4.0, 3.0, 2.5)
TABLE = {"id": "table", "center": [2.6, 1.2, 0.375], "size": [0.9, 0.6, 0.75]}


def room_xml(extra="", light=True):
    """The analytic room (inner faces at x=0..4, y=0..3, z=0..2.5) built from visible boxes."""
    w, d, h = ROOM
    t = 0.05
    boxes = [("floor", (w / 2, d / 2, -t), (w / 2 + t, d / 2 + t, t)),
             ("ceiling", (w / 2, d / 2, h + t), (w / 2 + t, d / 2 + t, t)),
             ("wall_x0", (-t, d / 2, h / 2), (t, d / 2, h / 2)),
             ("wall_x1", (w + t, d / 2, h / 2), (t, d / 2, h / 2)),
             ("wall_y0", (w / 2, -t, h / 2), (w / 2, t, h / 2)),
             ("wall_y1", (w / 2, d + t, h / 2), (w / 2, t, h / 2)),
             ("table", TABLE["center"], [s / 2 for s in TABLE["size"]])]
    geoms = "".join(f'<geom name="{"" if n == "table" else "room_geometry_"}{n}" type="box" pos="{" ".join(map(str, c))}" '
                    f'size="{" ".join(map(str, s))}" rgba=".85 .85 .8 1" group="1"/>' for n, c, s in boxes)
    lights = '<light pos="2 1.5 2.3" dir="0 0 -1"/>' if light else ""
    return (f'<mujoco><visual><global offwidth="1600" offheight="1600"/></visual>'
            f'<asset><material name="csi_blood" rgba=".42 .02 .03 1"/></asset>'
            f'<worldbody>{lights}{geoms}{extra}</worldbody></mujoco>')


SCN = {
    "id": "test", "seed": 7,
    "room": {"size": list(ROOM), "materials": {}},
    "furniture": [TABLE],
    "events": [
        {"mechanism": "impact", "origin": [1.2, 1.5, 1.4], "params": {"n": 60, "axis": [-1, 0, 0.1], "cone_deg": 50}},
        {"mechanism": "impact", "origin": [2.0, 1.4, 1.1], "params": {"n": 40, "axis": [0.4, -0.3, -1], "cone_deg": 40}},
        {"mechanism": "castoff", "origin": [2.0, 1.5, 1.4], "params": {"n": 25, "heading_deg": 0}},
        {"mechanism": "drip", "origin": [1.0, 1.0, 1.0], "params": {"n": 8, "end": [3.0, 2.5, 1.0]}},
    ],
}


def kind_of(analytic_id):
    if "floor" in analytic_id:
        return "floor"
    if "ceiling" in analytic_id:
        return "ceiling"
    return "wall" if "wall" in analytic_id else "furniture"


@pytest.fixture(scope="module")
def both():
    analytic = generate(SCN, satellites=False)
    m = mujoco.MjModel.from_xml_string(room_xml())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mj = generate(SCN, satellites=False, surfaces=MjSurfaces(m, d))
    return analytic, mj


def test_droplets_land_on_the_same_surfaces_as_the_analytic_room(both):
    analytic, mj = both
    a = {x["id"]: x for x in analytic["droplets"]}
    surf_kind = {s["id"]: s["kind"] for s in mj["surfaces"]}
    n = 0
    for dm in mj["droplets"]:
        da = a[dm["id"]]
        assert dm["landed"] == da["landed"]
        if not dm["landed"]:
            continue
        n += 1
        assert np.linalg.norm(np.subtract(dm["impact_pos"], da["impact_pos"])) < 1e-3, dm["id"]
        ka = kind_of(da["surface_id"]) if not da["surface_id"].startswith("table") else "furniture"
        km = surf_kind[dm["surface_id"]]
        if ka == "furniture":     # the table top / sides are named after their geom
            assert "table" in dm["surface_id"]
        else:
            assert km == ka, (dm["id"], km, ka)
        assert abs(dm["impact_angle_deg"] - da["impact_angle_deg"]) < 0.5
        assert np.dot(dm["dir_world"], da["dir_world"]) > 0.999
    assert n > 100


def test_each_mechanism_lands_where_expected(both):
    _, mj = both
    kinds = {s["id"]: s["kind"] for s in mj["surfaces"]}
    by_event = {}
    for dm in mj["droplets"]:
        if dm["landed"]:
            by_event.setdefault(dm["event"], []).append(kinds[dm["surface_id"]])
    assert by_event[0].count("wall") > 0.6 * len(by_event[0])        # fired at the x0 wall (cone reaches the floor)
    assert "furniture" in by_event[1] or "floor" in by_event[1]      # fired down at the table
    assert set(by_event[3]) == {"floor"}                              # drips fall on the floor


def test_surface_normals_face_the_droplet(both):
    _, mj = both
    surf = {s["id"]: s for s in mj["surfaces"]}
    for dm in mj["droplets"]:
        if dm["landed"]:
            assert np.dot(surf[dm["surface_id"]]["normal"], dm["impact_vel"]) < 0


@pytest.mark.parametrize("alpha_deg", [20, 35, 50, 70])
def test_rendered_stain_matches_its_impact_angle(alpha_deg):
    """One droplet hits the x0 wall at a known angle; the rendered ellipse has W/L = sin(alpha)."""
    import cv2

    a = np.radians(alpha_deg)
    gam = np.radians(30)                         # travel direction on the wall, from +y toward up
    v_t = np.array([0, np.cos(gam), np.sin(gam)])
    vel = 5.0 * (np.sin(a) * np.array([-1, 0, 0]) + np.cos(a) * v_t)
    hit = np.array([0.0, 1.5, 1.2])
    scn = {"id": "one", "seed": 1, "room": {"size": list(ROOM)},
           "events": [{"mechanism": "impact", "origin": (hit - vel / 5.0 * 0.05).tolist(),
                       "params": {"n": 1, "cone_deg": 0.01, "axis": vel.tolist(), "speed": [5.0, 5.0],
                                  "d_median_mm": 3.0, "d_sigma": 0.0, "d_range_mm": [3.0, 3.0], "jitter_m": 0.0}}]}
    m0 = mujoco.MjModel.from_xml_string(room_xml())
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    gt = generate(scn, satellites=False, surfaces=MjSurfaces(m0, d0))
    st = gt["stains"][0]
    assert abs(st["impact_angle_deg"] - alpha_deg) < 1.0
    xml = "".join(stain_geoms(gt, tails=False))
    cam = '<camera name="c" pos="0.15 1.5 1.2" xyaxes="0 1 0 0 0 1" fovy="20"/>'
    m = mujoco.MjModel.from_xml_string(room_xml(xml + cam))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 800, 800)
    r.update_scene(d, "c")
    img = r.render()
    r.close()
    red = (img[..., 0] > 60) & (img[..., 1] < 40) & (img[..., 2] < 40)
    cnts, _ = cv2.findContours(red.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w1, w2), ang = cv2.fitEllipse(c)
    W, L = sorted((w1, w2))
    assert abs(W / L - np.sin(a)) < 0.03, (W / L, np.sin(a))
    # pixel scale: the wall is 0.15 m away, fovy 20 deg over 800 px
    mm_per_px = 2 * 0.15 * np.tan(np.radians(10)) / 800 * 1000
    assert abs(L * mm_per_px - st["length"] * 1000) < 0.08 * st["length"] * 1000 + 0.1
    # long axis direction in the image: image right = world +y, image up = world +z
    long_axis = np.radians(ang + 90 if w1 < w2 else ang)
    img_dir = np.array([np.cos(long_axis), np.sin(long_axis)])          # (u right, v down)
    expected = np.array([np.cos(gam), -np.sin(gam)])                     # world (+y,+z) -> image (u, -v)
    assert abs(img_dir @ expected) > np.cos(np.radians(4))


def test_all_five_mechanisms_generate_and_render():
    scn = dict(SCN, events=SCN["events"] + [
        {"mechanism": "projected", "origin": [2.0, 1.5, 1.3], "params": {"azimuth_deg": 180, "duration": 1.5}},
        {"mechanism": "transfer", "start": [0.0, 0.8, 0.9], "end": [0.0, 1.3, 1.0], "normal": [1, 0, 0]},
    ])
    m0 = mujoco.MjModel.from_xml_string(room_xml())
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    surf = MjSurfaces(m0, d0)
    gt = generate(scn, surfaces=surf)
    gt["surfaces"] = surf.to_dict()
    assert {e["mechanism"] for e in gt["events"]} == {"impact", "castoff", "drip", "projected", "transfer"}
    smear = gt["smears"][0]
    assert {s["id"]: s["kind"] for s in gt["surfaces"]}[smear["surface_id"]] == "wall"
    geoms = stain_geoms(gt)
    m = mujoco.MjModel.from_xml_string(room_xml("".join(geoms)))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    assert sum(n.startswith("stain_") for n in names) == len(geoms) > len(gt["stains"])
    stain_ids = {n for n in names if n.startswith("stain_e")}
    for e in range(len(scn["events"])):
        assert any(n.startswith(f"stain_e{e}_") for n in stain_ids), f"event {e} has no rendered stains"
