#!/usr/bin/env python3
"""Go to a zone on a zone map the dog built itself (scripts/map_zones.py).

    python scripts/goto_zone.py --map outputs/zone_map/r1854 --list
    python scripts/goto_zone.py --map outputs/zone_map/r1854 --zone bathroom lounge_area      # simulation
    python scripts/goto_zone.py --map outputs/zone_map/<real map> --backend real --zone lab_2  # the real Go2

How it gets there:
1. Relocalise (real robot, or --relocalize): turn a full circle, collect LiDAR points, and match them to
   the saved map (mapping/localize.py), because the dog's odometry restarts from zero at every boot.
   If the match is poor, it stops and says so instead of driving on a wrong map.
2. Plan a path on the saved occupancy map to the zone's goal point (the free point deepest inside the
   zone, zones.json) and drive it with Phase 2's planner, pure pursuit and LiDAR safety layer; the map
   keeps updating from the live LiDAR, so new obstacles are avoided and the path is replanned.
3. On arrival, look around once: the camera classifier's majority vote is compared with the map's zone
   there, and the new evidence is added to the map (saved with --update-map).
Results: outputs/goto/<time>.json (+ a PNG of the map with the path driven).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401

import numpy as np  # noqa: E402


def relocalize(sim, zm, log=print, rate=0.5):
    """Look around with a throwaway map, then align the LiDAR points to the saved map."""
    from mapping.localize import align
    from navigation.explorer import Navigator
    tmp = Navigator(sim, verbose=False)
    tmp.safety.cfg = sim.safety_config
    pts = []
    tmp.on_tick.append(lambda nav: pts.append(sim.odom_points()[::4, :2]))
    st0 = sim.state()
    turned, prev = 0.0, st0.yaw
    t0 = sim.time
    while turned < 2 * np.pi and sim.time - t0 < 40:
        st = tmp.tick(0, 0, rate)
        turned += abs((st.yaw - prev + np.pi) % (2 * np.pi) - np.pi)
        prev = st.yaw
    tmp.tick(0, 0, 0)
    p = np.vstack(pts) if pts else np.zeros((0, 2))
    x, y, _, yaw, _ = sim.link.pose()
    res = align(zm.grid, p, (x, y), yaw)
    log(f"[relocalise] {res['points']} points, score {res['score']:.2f} -> on the map at "
        f"({res['x']:.2f}, {res['y']:.2f}), heading {np.degrees(res['yaw']):.0f} deg: {'OK' if res['ok'] else 'NOT TRUSTED'}")
    if res["ok"]:
        sim.set_map_transform(res["dx"], res["dy"], res["dyaw"])
    return res


def main():
    from real_robot.session import add_real_args
    from sim_env.launch import add_sim_args
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--map", required=True, help="a zone map directory from map_zones.py")
    ap.add_argument("--zone", nargs="*", default=[], help="zone name(s) or 1-based number(s), visited in order")
    ap.add_argument("--list", action="store_true", help="list the map's zones and exit")
    ap.add_argument("--backend", choices=["sim", "real"], default="sim")
    ap.add_argument("--relocalize", action="store_true", help="also relocalise in simulation (always on real)")
    ap.add_argument("--odom-offset", nargs=3, type=float, metavar=("DX", "DY", "DYAW_DEG"),
                    help="--backend real --link sim: pretend the dog booted elsewhere (tests relocalisation)")
    ap.add_argument("--update-map", action="store_true", help="save the map with the new evidence afterwards")
    ap.add_argument("--model", help="zone classifier checkpoint (default: the one the map was built with)")
    ap.add_argument("--house", default="scenesmith_house_185")
    add_real_args(ap)
    a = ap.parse_args()

    from mapping.mapper import ZoneMapper
    from mapping.zone_map import ZoneMap
    from navigation.explorer import Navigator
    from perception.zone_classifier import ZoneClassifier

    zm, _ = ZoneMap.load(a.map)
    meta = json.loads((Path(a.map) / "zones.json").read_text())
    regions = zm.regions()
    if a.list or not a.zone:
        print(f"zones on {a.map}:")
        for i, n in enumerate(zm.classes):
            r = regions.get(n)
            print(f"  {i + 1} {n:16s} " + (f"{r['area_m2']:6.1f} m2  goal {r['goal']}  confidence {r['confidence']}"
                                             if r else "not on this map"))
        return
    targets = [zm.classes[int(z) - 1] if z.isdigit() else z for z in a.zone]
    for t in targets:
        if t not in regions:
            raise SystemExit(f"zone '{t}' is not on this map (zones: {', '.join(regions)})")

    if a.backend == "sim":
        from sim_env.launch import session_from_args
        if not (a.scenario or a.random is not None) and meta.get("scenario"):
            a.scenario = meta["scenario"]
        sim, _ = session_from_args(a)
    else:
        from real_robot.session import RealGo2Session, SimLink
        if a.link == "sim":
            from sim_env.launch import session_from_args
            if not (a.scenario or a.random is not None) and meta.get("scenario"):
                a.scenario = meta["scenario"]
            base, _ = session_from_args(a)
            off = a.odom_offset or (0, 0, 0)
            sim = RealGo2Session(SimLink(base, odom_offset=(off[0], off[1], np.radians(off[2]))))
        else:
            sim = RealGo2Session.from_args(a)
    out = ROOT / "outputs" / "goto"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    result = {"map": str(a.map), "backend": a.backend, "targets": targets, "visits": []}
    nav = None
    try:
        if a.backend == "real" or a.relocalize:
            if not hasattr(sim, "set_map_transform"):
                raise SystemExit("--relocalize needs --backend real (the sim session is always in the map frame)")
            res = relocalize(sim, zm)
            result["relocalisation"] = res
            if not res["ok"]:
                raise SystemExit("Could not find the dog on this map (score too low). Start it somewhere the map "
                                 "covers well, away from people and moved furniture, and try again.")
        clf = ZoneClassifier(a.model or meta.get("model"))
        nav = Navigator(sim, grid=zm.grid, verbose=True)
        if hasattr(sim, "safety_config"):                     # real dog: larger stop / slow distances
            nav.safety.cfg = sim.safety_config
        mapper = ZoneMapper(nav, clf, zone_map=zm)
        for t in targets:
            goal = np.array(zm.regions()[t]["goal"])
            print(f"[goto] {t}: path to {goal.round(2).tolist()}")
            t0 = sim.time
            ok = nav.go_to(goal, timeout=240.0, tol=0.6, final_tol=0.25)
            st = sim.state()
            mapper.look_around()
            votes = Counter(int(np.argmax(p)) for p in mapper.recent)
            seen = zm.classes[votes.most_common(1)[0][0]] if votes else None
            on_map = zm.zone_at(st.position[:2])[0]
            v = {"target": t, "reached_goal": bool(ok), "seconds": round(sim.time - t0, 1),
                 "end_xy": np.round(st.position[:2], 2).tolist(),
                 "map_zone_at_end": zm.classes[on_map] if on_map >= 0 else None,
                 "camera_majority": seen, "camera_share": round(votes.most_common(1)[0][1] / sum(votes.values()), 2)
                 if votes else None}
            v["arrived"] = bool(ok and v["map_zone_at_end"] == t)
            result["visits"].append(v)
            print(f"[goto] {t}: {'ARRIVED' if v['arrived'] else 'NOT ARRIVED'} in {v['seconds']} s; map says "
                  f"{v['map_zone_at_end']}, camera look-around says {seen} ({v['camera_share']})")
    except KeyboardInterrupt as e:
        print(f"stopped: {e}")
        result["stopped"] = str(e)
    finally:
        if nav is not None:
            nav.vel.stop()
            result["path"] = [(round(x, 2), round(y, 2)) for _, x, y, _ in nav.log.path[::5]]
            result["n_collisions"] = len(nav.log.collisions)
        if hasattr(sim, "close"):
            sim.close()
    (out / f"goto_{stamp}.json").write_text(json.dumps(result, indent=1, default=float))
    if nav is not None:
        import cv2
        path = [(x, y) for _, x, y, _ in nav.log.path]
        cv2.imwrite(str(out / f"goto_{stamp}.png"), cv2.cvtColor(zm.image(path), cv2.COLOR_RGB2BGR))
        if a.update_map:
            zm.save(a.map, extra={k: meta[k] for k in ("backend", "model", "scenario", "link") if k in meta})
    n = sum(v["arrived"] for v in result["visits"])
    print(f"{n}/{len(targets)} zones reached -> {out / f'goto_{stamp}.json'}")


if __name__ == "__main__":
    main()
