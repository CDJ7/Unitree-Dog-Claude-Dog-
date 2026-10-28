"""Scenario -> droplets -> stains, with full ground truth.

    gt = generate(load_scenario("sim_env/scenarios/analytic/s1_living_room_blunt.yaml"))
    save_ground_truth(gt, "out/s1_gt.json")

The scenario is a plain dict (see sim_env/scenarios/analytic/*.yaml and
docs/plan.md for the schema). The output is a JSON-ready dict; nothing here
needs MuJoCo, so it runs anywhere and is fully reproducible from the seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from . import mechanisms as M
from .geometry import SurfaceSet, box_surfaces, room_surfaces
from .physics import fly
from .stains import Smear, make_stain, satellites_for

GENERATOR_VERSION = "0.1.0"


def load_scenario(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_surfaces(scn):
    room = scn["room"]
    rects = room_surfaces(room["size"], room.get("origin", (0, 0, 0)), room.get("ceiling", True),
                          room.get("prefix", "room"), room.get("materials"))
    for f in scn.get("furniture", []) or []:
        rects += box_surfaces(f["id"], f["center"], f["size"], f.get("material", "wood"), f.get("yaw_deg", 0.0))
    return SurfaceSet(rects)


def _transfer(ev_idx, ev, surfaces, rng):
    """A smear from ev['start'] to ev['end'] (world points) on surface ev['surface']."""
    p = M.merged("transfer", ev.get("params"))
    if ev.get("surface") in surfaces.by_id:
        rect = surfaces.by_id[ev["surface"]]
    else:                                   # MuJoCo scene: find the surface by ray casting
        rect, _ = surfaces.surface_at(ev["start"], ev.get("normal", (0, 0, 1)))
    a = rect.to_local(ev["start"])
    b = rect.to_local(ev["end"])
    s = np.linspace(0, 1, 25)
    line = a[None] + s[:, None] * (b - a)[None]
    side = np.array([-(b - a)[1], (b - a)[0]])
    side /= max(np.linalg.norm(side), 1e-9)
    line += side[None] * (0.01 * np.sin(s * np.pi * rng.uniform(1.5, 3.0)))[:, None]
    line[:, 0] = np.clip(line[:, 0], -rect.half_u, rect.half_u)
    line[:, 1] = np.clip(line[:, 1], -rect.half_v, rect.half_v)
    sm = Smear(f"e{ev_idx}_smear0", rect.id, ev_idx, line.tolist(), float(p["width"]), float(p["feather_frac"]),
               rect.to_world(line).tolist())
    meta = {"origin": list(map(float, ev["start"])), "end": list(map(float, ev["end"])), "params": p,
            "surface": rect.id}
    return sm, meta


def generate(scn, seed=None, satellites=True, surfaces=None):
    """Spatter + ground truth. `surfaces` defaults to the analytic room in the
    scenario; pass an `mj_surfaces.MjSurfaces` to land droplets in a MuJoCo scene."""
    seed = scn.get("seed", 0) if seed is None else seed
    rng = np.random.default_rng(seed)
    if surfaces is None:
        surfaces = build_surfaces(scn)
    droplets, stains, smears, events = [], [], [], []
    for ei, ev in enumerate(scn["events"]):
        mech = ev["mechanism"]
        record = {"index": ei, "mechanism": mech, "weapon_category": ev.get("weapon_category"),
                  "weapon": ev.get("weapon"), "description": ev.get("description", "")}
        if mech == "transfer":
            sm, meta = _transfer(ei, ev, surfaces, rng)
            smears.append(sm.to_dict())
            record.update(meta, droplet_ids=[], smear_ids=[sm.id])
            events.append(record)
            continue
        launches, meta = M.LAUNCHERS[mech](ev["origin"], ev.get("params"), rng)
        flights = fly(launches, surfaces)
        ids = []
        for i, (l, fl) in enumerate(zip(launches, flights)):
            did = f"e{ei}_d{i}"
            ids.append(did)
            drop = {"id": did, "event": ei, "release_pos": l.pos.tolist(), "release_vel": l.vel.tolist(),
                    "release_time": l.t_release, "diameter": l.diameter, "landed": fl.landed,
                    "flight_time": fl.flight_time,
                    "path": fl.path[:: max(1, len(fl.path) // 12)].tolist() + [fl.pos.tolist()]}
            if fl.landed:
                rect = surfaces.rects[fl.surface_index]
                st = make_stain(f"{did}_st", rect, ei, did, fl.pos, fl.vel, l.diameter, rng)
                drop.update(surface_id=rect.id, impact_pos=fl.pos.tolist(), impact_vel=fl.vel.tolist(),
                            impact_angle_deg=st.impact_angle_deg, gamma_deg=st.gamma_deg,
                            dir_world=st.dir_world, stain_id=st.id)
                stains.append(st.to_dict())
                if satellites:
                    for s in satellites_for(st, rect, float(np.linalg.norm(fl.vel)), rng):
                        stains.append(s.to_dict())
            droplets.append(drop)
        record.update(meta, droplet_ids=ids)
        events.append(record)
    primary = scn.get("summary", {}).get("primary_event", 0)
    pe = events[primary]
    return {
        "scenario_id": scn["id"], "title": scn.get("title", ""), "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "primary_mechanism": pe["mechanism"], "weapon_category": pe.get("weapon_category"),
        "weapon": pe.get("weapon"), "origin": pe.get("origin"),
        "sequence": scn.get("summary", {}).get("sequence", []),
        "events": events, "droplets": droplets, "stains": stains, "smears": smears,
        "surfaces": surfaces.to_dict(),
        "room": scn.get("room"), "furniture": scn.get("furniture", []) or [],
        "mannequins": scn.get("mannequins", []) or [], "distractors": scn.get("distractors", []) or [],
        "binding": scn.get("binding"),
    }


def save_ground_truth(gt, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=1)
    return path


def load_ground_truth(path):
    with open(path) as f:
        return json.load(f)


def summary(gt):
    """One line per event: what landed where (handy for quick checks)."""
    lines = [f"{gt['scenario_id']}: primary {gt['primary_mechanism']} ({gt['weapon_category']})"]
    for ev in gt["events"]:
        ds = [d for d in gt["droplets"] if d["event"] == ev["index"]]
        landed = [d for d in ds if d["landed"]]
        by_surf = {}
        for d in landed:
            by_surf[d["surface_id"]] = by_surf.get(d["surface_id"], 0) + 1
        lines.append(f"  event {ev['index']} {ev['mechanism']}: {len(landed)}/{len(ds)} landed {by_surf}"
                     + (f" smears {ev.get('smear_ids')}" if ev["mechanism"] == "transfer" else ""))
    return "\n".join(lines)
