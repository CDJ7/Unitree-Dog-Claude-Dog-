"""Seeded random crime-scene scenarios in the same schema as configs/*.yaml.

    python -m sim_env.scenarios.random_scenario --seed 7 --out /tmp/r7.yaml

Each scenario has a box room, 1-3 pieces of furniture against the walls, a
primary event (any mechanism, chosen uniformly unless `mechanism` is given),
up to two secondary events, mannequins placed consistently with the event,
and 1-3 red distractor objects that are not blood.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

FURNITURE = [  # name, size range (sx, sy, sz), material
    ("sofa", ((1.6, 2.2), (0.8, 0.95), (0.7, 0.9)), "fabric"),
    ("table", ((0.8, 1.4), (0.6, 0.9), (0.7, 0.78)), "wood"),
    ("cabinet", ((0.8, 1.6), (0.4, 0.6), (0.8, 1.9)), "wood"),
    ("bed", ((1.4, 1.8), (1.9, 2.1), (0.5, 0.65)), "fabric"),
]
DISTRACTORS = ["cushion", "book", "towel", "bag", "mug", "rug"]
MECHS = ["impact", "castoff", "projected", "drip", "transfer"]
WEAPONS = {"impact": [("blunt", "pipe"), ("blunt", "bat"), ("blunt", "hammer"), ("none", None)],
           "castoff": [("blunt", "bat"), ("blunt", "pipe"), ("edged", "knife")],
           "projected": [("edged", "knife"), ("none", None)],
           "drip": [("none", None)],
           "transfer": [("none", None)]}


def _r(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _round(x, nd=3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


def random_scenario(seed, mechanism=None, secondary=True):
    rng = np.random.default_rng(seed)
    W, D, H = _r(rng, 3.5, 6.0), _r(rng, 3.0, 5.0), _r(rng, 2.4, 2.9)
    furniture = []
    for i in range(int(rng.integers(1, 4))):
        name, (rx, ry, rz), mat = FURNITURE[int(rng.integers(len(FURNITURE)))]
        sx, sy, sz = _r(rng, *rx), _r(rng, *ry), _r(rng, *rz)
        side = int(rng.integers(4))            # which wall it stands against
        if side in (0, 1):
            x = 0.05 + sy / 2 if side == 0 else W - 0.05 - sy / 2
            y = _r(rng, sx / 2 + 0.1, D - sx / 2 - 0.1)
            sx, sy = sy, sx
        else:
            y = 0.05 + sy / 2 if side == 2 else D - 0.05 - sy / 2
            x = _r(rng, sx / 2 + 0.1, W - sx / 2 - 0.1)
        furniture.append({"id": f"{name}_{i}", "center": [x, y, sz / 2], "size": [sx, sy, sz], "material": mat})

    def free_xy(margin=0.7):
        for _ in range(200):
            x, y = _r(rng, margin, W - margin), _r(rng, margin, D - margin)
            if all(abs(x - f["center"][0]) > f["size"][0] / 2 + 0.3 or abs(y - f["center"][1]) > f["size"][1] / 2 + 0.3
                   for f in furniture):
                return x, y
        return W / 2, D / 2

    def clamp(c, size, margin=0.3):
        return float(np.clip(c, margin, size - margin))

    mech = mechanism or MECHS[int(rng.integers(len(MECHS)))]
    events, mannequins, sequence = [], [], []
    vx, vy = free_xy()
    heading = _r(rng, -180, 180)
    hx, hy = np.cos(np.radians(heading)), np.sin(np.radians(heading))
    cat, weapon = WEAPONS[mech][int(rng.integers(len(WEAPONS[mech])))]
    if mech == "impact":
        z = _r(rng, 0.4, 1.7)
        events.append({"mechanism": "impact", "origin": [vx, vy, z], "weapon_category": cat, "weapon": weapon,
                       "params": {"n": int(rng.integers(80, 180)), "axis": [hx, hy, _r(rng, -0.1, 0.4)],
                                  "cone_deg": _r(rng, 45, 75), "speed": [_r(rng, 1.0, 2.0), _r(rng, 4.0, 7.5)]}})
        mannequins.append({"id": "victim", "role": "victim", "position": [vx, vy], "yaw_deg": heading + 180,
                           "posture": "standing" if z > 1.2 else ("kneeling" if z > 0.6 else "lying"), "height": 1.75})
        if cat != "none":
            mannequins.append({"id": "attacker", "role": "attacker",
                               "position": [clamp(vx - 0.8 * hx, W), clamp(vy - 0.8 * hy, D)],
                               "yaw_deg": heading, "posture": "standing", "height": 1.8})
        sequence.append("A blow to the victim produces impact spatter")
    elif mech == "castoff":
        # the swing arc (radius up to 1 m) must stay inside the room
        ax, ay = free_xy(1.1)
        vx, vy = clamp(ax + 0.8 * hx, W), clamp(ay + 0.8 * hy, D)
        events.append({"mechanism": "castoff", "origin": [ax, ay, _r(rng, 1.35, 1.55)], "weapon_category": cat,
                       "weapon": weapon, "params": {"n": int(rng.integers(25, 55)), "heading_deg": heading,
                                                    "radius": _r(rng, 0.7, 1.0), "omega": _r(rng, 8.0, 13.0),
                                                    "release_deg": [_r(rng, 30, 60), _r(rng, 130, 165)]}})
        mannequins += [{"id": "attacker", "role": "attacker", "position": [ax, ay], "yaw_deg": heading,
                        "posture": "standing", "height": 1.8},
                       {"id": "victim", "role": "victim", "position": [vx, vy], "yaw_deg": heading + 180,
                        "posture": "kneeling", "height": 1.7}]
        sequence.append("A bloodied object is swung overhead, throwing cast-off")
    elif mech == "projected":
        events.append({"mechanism": "projected", "origin": [vx, vy, _r(rng, 1.0, 1.5)], "weapon_category": cat,
                       "weapon": weapon, "params": {"azimuth_deg": heading, "elevation_deg": _r(rng, 0, 30),
                                                    "sweep_deg": _r(rng, 20, 50), "duration": _r(rng, 1.5, 3.0),
                                                    "source_vel": [_r(rng, -0.2, 0.2), _r(rng, -0.2, 0.2), -0.1]}})
        mannequins.append({"id": "victim", "role": "victim", "position": [vx, vy], "yaw_deg": heading,
                           "posture": "standing", "height": 1.75})
        sequence.append("An injury produces pulsing projected spurts")
    elif mech == "drip":
        ex, ey = free_xy()
        events.append({"mechanism": "drip", "origin": [vx, vy, _r(rng, 0.8, 1.2)], "weapon_category": cat,
                       "weapon": weapon, "params": {"n": int(rng.integers(10, 30)),
                                                    "end": [ex, ey, 1.0] if rng.uniform() < 0.7 else None}})
        mannequins.append({"id": "victim", "role": "victim", "position": [ex, ey], "yaw_deg": heading,
                           "posture": "standing", "height": 1.75})
        sequence.append("A bleeding person moves through the room, dripping")
    else:  # transfer on the nearest wall
        walls = {"room_wall_x0": (0.0, vy), "room_wall_x1": (W, vy), "room_wall_y0": (vx, 0.0), "room_wall_y1": (vx, D)}
        wid = min(walls, key=lambda k: np.hypot(walls[k][0] - vx, walls[k][1] - vy))
        wx, wy = walls[wid]
        along = np.array([0.0, 1.0]) if wid in ("room_wall_x0", "room_wall_x1") else np.array([1.0, 0.0])
        sgn = 1 if rng.uniform() < 0.5 else -1
        z0 = _r(rng, 0.7, 1.4)
        L = _r(rng, 0.3, 0.8)
        start = [wx, wy, z0]
        end = [float(np.clip(wx + sgn * along[0] * L, 0.1, W - 0.1)), float(np.clip(wy + sgn * along[1] * L, 0.1, D - 0.1)),
               z0 + _r(rng, -0.3, 0.1)]
        events.append({"mechanism": "transfer", "surface": wid, "start": start, "end": end,
                       "weapon_category": cat, "params": {"width": _r(rng, 0.04, 0.09)}})
        mannequins.append({"id": "victim", "role": "victim", "position": [vx, vy], "yaw_deg": heading,
                           "posture": "standing", "height": 1.75})
        sequence.append("A bloodied hand or object is wiped along the wall")
    if secondary and mech != "drip" and rng.uniform() < 0.6:
        ex, ey = free_xy()
        events.append({"mechanism": "drip", "origin": [vx, vy, _r(rng, 0.8, 1.1)], "weapon_category": "none",
                       "params": {"n": int(rng.integers(8, 20)), "end": [ex, ey, 1.0]}})
        sequence.append("The victim moves away, dripping")
    distractors = []
    for i in range(int(rng.integers(1, 4))):
        kind = DISTRACTORS[int(rng.integers(len(DISTRACTORS)))]
        x, y = free_xy(0.3)
        distractors.append({"id": f"red_{kind}_{i}", "kind": kind, "center": [x, y, 0.05],
                            "size": [0.3, 0.2, 0.1], "color": [_r(rng, 0.55, 0.85), _r(rng, 0.02, 0.1), _r(rng, 0.03, 0.1)]})
    scn = {"id": f"random_{seed:05d}", "title": f"Random scenario {seed} ({mech})", "seed": int(seed),
           "room": {"size": [W, D, H]}, "furniture": furniture, "distractors": distractors,
           "mannequins": mannequins, "events": events,
           "summary": {"primary_event": 0, "sequence": sequence}, "binding": {"environment": None}}
    return _round(scn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mechanism", choices=MECHS)
    ap.add_argument("--out", help="write YAML here (default: print)")
    a = ap.parse_args()
    text = yaml.safe_dump(random_scenario(a.seed, a.mechanism), sort_keys=False)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
