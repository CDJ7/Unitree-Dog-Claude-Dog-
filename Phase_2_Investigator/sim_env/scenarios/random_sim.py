"""Seeded random crime-scene scenarios for the simulator (houses and the simple room).

    python -m sim_env.scenarios.random_sim --seed 7            # writes sim_env/scenarios/generated/r7.yaml
    ./scripts/run_sim.sh --random 7                            # build + open it

The environment is mapped once by casting rays straight down on a 5 cm grid
(cached in sim_env/build/maps/): free floor, furniture tops (for distractors)
and wall distance. A scenario then picks, from the seed:
- the main mechanism (impact, castoff, projected, drip or transfer as primary)
  and a matching weapon category, plus a secondary event;
- a victim spot on free floor 0.6-1.3 m from a wall, the spatter aimed at it;
- attacker position for swings, the victim's final posture and position;
- 2 obstacles on free floor between the robot spawn and the scene;
- 3 red distractors on furniture tops.
The YAML is in the same schema as configs/*.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
MAPS = HERE.parent / "build" / "maps"
ENVIRONMENTS = ["scenesmith_house_185", "scenesmith_house_186", "csi_simple_room"]
WEAPONS = {"blunt": ["bat", "pipe", "hammer"], "edged": ["knife", "machete"], "none": [None]}
RES = 0.05


def floor_map(env):
    """Grid maps of an environment (5 cm cells), cached in sim_env/build/maps/<env>.npz.

    top:       top visible surface z per cell (furniture tops for distractors), nan = nothing
    blocked:   a collision box (wall, furniture) occupies the cell between 0.02 and 1.8 m
    tall:      a collision box taller than 1.2 m (walls, wardrobes): what a victim can sit against
    free:      floor present and not blocked
    wall_dist: distance (m) from each cell to the nearest non-free cell
    """
    path = MAPS / f"{env}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import mujoco
    import sim_env  # noqa: F401
    from sim_env.runtime import SimSession
    from sim_env.spatter.mj_surfaces import MjSurfaces
    xs = np.arange(-1.0, 10.0, RES)
    ys = np.arange(-1.0, 7.0, RES)
    X, Y = np.meshgrid(xs, ys)
    sim = SimSession(env)
    try:
        m, d = sim.model, sim.data
        surf = MjSurfaces(m, d, exclude_geoms=sim.robot_geoms)
        top = np.full(X.shape, np.nan)
        for i in range(len(ys)):
            for j in range(len(xs)):
                t, _ = surf.cast([xs[j], ys[i], 1.95], [0, 0, -2.1])
                if t is not None:
                    top[i, j] = 1.95 - 2.1 * t
        blocked = np.zeros(X.shape, bool)
        tall = np.zeros(X.shape, bool)
        for g in range(m.ngeom):
            if m.geom_group[g] != 3 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or g in sim.robot_geoms:
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            c, h = d.geom_xpos[g], m.geom_size[g]
            zext = np.abs(R[2]) @ h
            lo, hi = c[2] - zext, c[2] + zext
            if hi < 0.02 or lo > 1.8:
                continue
            # cell centres inside the box footprint (box local frame)
            P = np.stack([X - c[0], Y - c[1], np.zeros_like(X)], -1) @ R
            inside = (np.abs(P[..., 0]) <= h[0] + 0.02) & (np.abs(P[..., 1]) <= h[1] + 0.02)
            if abs(R[2, 2]) < 0.9:            # tilted box: use its xy bounding box
                ext = np.abs(R[:2]) @ h
                inside = (np.abs(X - c[0]) <= ext[0]) & (np.abs(Y - c[1]) <= ext[1])
            blocked |= inside
            if hi > 1.2:
                tall |= inside
    finally:
        sim.close()
    free = (np.abs(np.nan_to_num(top, nan=9.0)) < 0.03) & ~blocked
    from scipy.ndimage import distance_transform_edt
    wall_dist = distance_transform_edt(free) * RES
    out = {"origin": np.array([xs[0], ys[0]]), "res": np.array(RES), "top": top, "free": free,
           "blocked": blocked, "tall": tall, "wall_dist": wall_dist}
    MAPS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def _cell_to_xy(fm, i, j):
    return float(fm["origin"][0] + j * RES), float(fm["origin"][1] + i * RES)


def _nearest_wall_dir(fm, i, j):
    """(unit xy direction, distance m) toward the nearest tall collision cell (a wall)."""
    idx = np.argwhere(fm["tall"])
    if len(idx) == 0:
        return np.array([1.0, 0.0]), 9.0
    dd = np.hypot(idx[:, 0] - i, idx[:, 1] - j)
    k = int(np.argmin(dd))
    v = np.array([idx[k, 1] - j, idx[k, 0] - i], float)
    return v / max(np.linalg.norm(v), 1e-9), float(dd[k] * RES)


def _segment_free(fm, a, b):
    n = int(np.hypot(b[0] - a[0], b[1] - a[1]) / RES) + 2
    for s_ in np.linspace(0, 1, n):
        x, y = a[0] + s_ * (b[0] - a[0]), a[1] + s_ * (b[1] - a[1])
        jj = int(round((x - fm["origin"][0]) / RES))
        ii = int(round((y - fm["origin"][1]) / RES))
        if not (0 <= ii < fm["free"].shape[0] and 0 <= jj < fm["free"].shape[1]) or not fm["free"][ii, jj]:
            return False
    return True


def _pick(rng, mask):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise ValueError("no cell satisfies the constraints")
    return idx[rng.integers(len(idx))]


def random_sim_scenario(seed, env=None, mechanism=None):
    rng = np.random.default_rng(seed)
    env = env or ENVIRONMENTS[rng.integers(len(ENVIRONMENTS))]
    fm = floor_map(env)
    free, wd = fm["free"], fm["wall_dist"]
    spawn_json = json.loads((HERE.parent / "build/overlay/assets/environments" / env / "spawn.json").read_text())
    sx, sy = spawn_json["position"]
    syaw = float(spawn_json.get("yaw", 0.0))
    ii, jj = np.indices(free.shape)
    X = fm["origin"][0] + jj * RES
    Y = fm["origin"][1] + ii * RES
    far_from_spawn = np.hypot(X - sx, Y - sy) > 1.6
    mech = mechanism or rng.choice(["impact", "castoff", "projected", "impact", "drip"])
    cat = {"impact": rng.choice(["blunt", "blunt", "none"]), "castoff": rng.choice(["blunt", "edged"]),
           "projected": "edged", "drip": rng.choice(["edged", "none"])}[mech]
    weapon = WEAPONS[cat][rng.integers(len(WEAPONS[cat]))]

    for _ in range(200):
        i, j = _pick(rng, free & (wd > 0.6) & (wd < 1.6) & far_from_spawn)
        to_wall, d_wall = _nearest_wall_dir(fm, i, j)
        if 0.6 < d_wall < 1.6:
            break
    vx, vy = _cell_to_xy(fm, i, j)
    yaw_wall = float(np.degrees(np.arctan2(to_wall[1], to_wall[0])))
    away = -to_wall
    events, mannequins = [], []
    head = 1.55 if cat != "none" else 0.6
    victim = {"id": "victim", "role": "victim", "height": round(float(rng.uniform(1.6, 1.85)), 2),
              "event_position": [round(vx, 2), round(vy, 2)], "event_yaw_deg": round(yaw_wall + 180, 1),
              "event_posture": "standing"}
    if mech == "impact":
        events.append({"mechanism": "impact", "description": "Blow at the victim; spatter toward the nearest wall",
                       "origin": [round(vx, 2), round(vy, 2), head], "weapon_category": cat, "weapon": weapon,
                       "params": {"n": int(rng.integers(150, 260)), "cone_deg": float(rng.uniform(35, 55)),
                                  "axis": [round(float(to_wall[0]), 3), round(float(to_wall[1]), 3), 0.12],
                                  "speed": [2.0, float(rng.uniform(5, 7.5))], "d_median_mm": 1.5}})
    elif mech == "projected":
        events.append({"mechanism": "projected", "description": "Arterial-style projected pattern toward the wall",
                       "origin": [round(vx, 2), round(vy, 2), 1.35], "weapon_category": cat, "weapon": weapon,
                       "params": {"azimuth_deg": round(yaw_wall, 1), "elevation_deg": float(rng.uniform(5, 20)),
                                  "sweep_deg": float(rng.uniform(20, 45)), "duration": float(rng.uniform(1.5, 3.0))}})
    elif mech == "drip":
        for _ in range(500):
            ei, ej = _pick(rng, free & (wd > 0.4))
            ex, ey = _cell_to_xy(fm, ei, ej)
            if 1.5 < np.hypot(ex - vx, ey - vy) < 3.5 and _segment_free(fm, (vx, vy), (ex, ey)):
                break
        events.append({"mechanism": "drip", "description": "Bleeding person walks across the room",
                       "origin": [round(vx, 2), round(vy, 2), 1.0], "weapon_category": cat, "weapon": weapon,
                       "params": {"n": int(rng.integers(15, 30)), "end": [round(ex, 2), round(ey, 2), 1.0]}})
    if mech == "castoff" or (mech == "impact" and cat == "blunt"):
        ax_, ay_ = vx + away[0] * 0.8, vy + away[1] * 0.8
        heading = float(np.degrees(np.arctan2(to_wall[1], to_wall[0])))
        if mech == "castoff":
            events.append({"mechanism": "impact", "description": "Blow at the victim",
                           "origin": [round(vx, 2), round(vy, 2), head], "weapon_category": cat, "weapon": weapon,
                           "params": {"n": 90, "cone_deg": 45, "axis": [round(float(to_wall[0]), 3),
                                                                       round(float(to_wall[1]), 3), 0.1]}})
        events.append({"mechanism": "castoff", "description": "Back-swing of the bloodied weapon",
                       "origin": [round(ax_, 2), round(ay_, 2), 1.45], "weapon_category": cat, "weapon": weapon,
                       "params": {"n": int(rng.integers(30, 50)), "heading_deg": round(heading, 1),
                                  "radius": 0.95 if cat == "blunt" else 0.75, "omega": float(rng.uniform(9, 12)),
                                  "release_deg": [40, 150]}})
        mannequins.append({"id": "attacker", "role": "attacker", "present": False,
                           "position": [round(ax_, 2), round(ay_, 2)], "yaw_deg": round(heading, 1),
                           "posture": "standing", "height": 1.8, "event_position": [round(ax_, 2), round(ay_, 2)],
                           "event_yaw_deg": round(heading, 1), "event_posture": "standing"})
    if rng.random() < 0.5 and mech != "drip":
        events.append({"mechanism": "drip", "description": "Wound drips where the victim stood",
                       "origin": [round(vx + away[0] * 0.2, 2), round(vy + away[1] * 0.2, 2), 0.9],
                       "weapon_category": cat, "params": {"n": int(rng.integers(8, 16))}})
    # final posture: sitting against the wall if the way there is clear, else lying where they fell
    fx, fy = vx + to_wall[0] * (d_wall - 0.22), vy + to_wall[1] * (d_wall - 0.22)
    def tall_at(x, y):
        jj = int(round((x - fm["origin"][0]) / RES))
        ii = int(round((y - fm["origin"][1]) / RES))
        return 0 <= ii < fm["tall"].shape[0] and 0 <= jj < fm["tall"].shape[1] and bool(fm["tall"][ii, jj])

    side = np.array([-to_wall[1], to_wall[0]])
    wall_behind = all(tall_at(fx + to_wall[0] * 0.3 + side[0] * o, fy + to_wall[1] * 0.3 + side[1] * o)
                      for o in (-0.25, 0.0, 0.25))
    if wall_behind and _segment_free(fm, (vx, vy), (vx + to_wall[0] * (d_wall - 0.35),
                                                     vy + to_wall[1] * (d_wall - 0.35))):
        victim.update(position=[round(fx, 2), round(fy, 2)], yaw_deg=round(yaw_wall + 180, 1), posture="sitting")
    else:
        lx, ly = vx, vy
        ang = float(rng.uniform(0, 360))
        for _ in range(36):
            hx, hy = lx + 1.63 * np.cos(np.radians(ang)), ly + 1.63 * np.sin(np.radians(ang))
            if _segment_free(fm, (lx, ly), (hx, hy)):
                break
            ang += 10
        victim.update(position=[round(lx, 2), round(ly, 2)], yaw_deg=round(ang, 1), posture="lying")
    mannequins.insert(0, victim)
    # obstacles: on free floor between the spawn and the scene
    obstacles = []
    for k in range(2):
        for _ in range(500):
            oi, oj = _pick(rng, free & (wd > 0.5))
            ox, oy = _cell_to_xy(fm, oi, oj)
            if np.hypot(ox - sx, oy - sy) > 1.0 and np.hypot(ox - vx, oy - vy) > 1.2 and \
                    all(np.hypot(ox - o["center"][0], oy - o["center"][1]) > 1.0 for o in obstacles):
                break
        if k == 0:
            obstacles.append({"id": "storage_box", "kind": "box", "center": [round(ox, 2), round(oy, 2)],
                              "size": [0.45, 0.35, 0.35], "yaw_deg": round(float(rng.uniform(0, 90)), 1),
                              "color": [0.55, 0.45, 0.32]})
        else:
            obstacles.append({"id": "laundry_basket", "kind": "basket", "center": [round(ox, 2), round(oy, 2)],
                              "radius": 0.21, "height": 0.44, "color": [0.75, 0.75, 0.70]})
    # distractors on furniture tops 0.3-0.9 m high
    tops = np.argwhere((np.nan_to_num(fm["top"]) > 0.3) & (np.nan_to_num(fm["top"]) < 0.9))
    kinds = [("red_cushion", "cushion", [0.38, 0.14, 0.34]), ("red_book", "book", [0.22, 0.16, 0.04]),
             ("red_towel", "towel", [0.40, 0.28, 0.04]), ("red_mug", "mug", None)]
    distractors = []
    for name, kind, size in [kinds[k] for k in rng.choice(len(kinds), 3, replace=False)]:
        ti, tj = tops[rng.integers(len(tops))]
        tx, ty = _cell_to_xy(fm, ti, tj)
        red = [round(float(rng.uniform(0.6, 0.85)), 2), round(float(rng.uniform(0.03, 0.09)), 2),
               round(float(rng.uniform(0.04, 0.10)), 2)]
        item = {"id": name, "kind": kind, "center": [round(tx, 2), round(ty, 2)], "z": "auto", "color": red}
        item.update({"radius": 0.04, "height": 0.10} if size is None else {"size": size})
        distractors.append(item)
    primary = next(k for k, e in enumerate(events) if e["mechanism"] == mech)
    return {
        "id": f"r{seed}", "title": f"Random scenario {seed}: {mech} ({cat})", "seed": int(seed),
        "environment": env, "spawn": [round(sx, 2), round(sy, 2), round(syaw, 4)],
        "mannequins": mannequins, "obstacles": obstacles, "distractors": distractors, "events": events,
        "summary": {"primary_event": primary, "sequence": [e["description"] for e in events]},
        "generated_by": "sim_env.scenarios.random_sim",
    }


def write_random(seed, env=None, mechanism=None):
    scn = random_sim_scenario(seed, env, mechanism)
    scn = json.loads(json.dumps(scn, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"r{seed}.yaml"
    path.write_text(yaml.safe_dump(scn, sort_keys=False))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--env", choices=ENVIRONMENTS)
    ap.add_argument("--mechanism", choices=["impact", "castoff", "projected", "drip"])
    a = ap.parse_args()
    print(write_random(a.seed, a.env, a.mechanism))


if __name__ == "__main__":
    main()
