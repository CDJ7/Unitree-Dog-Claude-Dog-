#!/usr/bin/env python3
"""A small 2-D test world for the go-to-zone navigator (used by tests/test_goto_zone.py and the sweep below).

    python real_go2/nav_sim.py            # 2 layouts x 4 starts x 4 headings -> results/real_q1/nav_sweep.json
                                          #   and nav_sim_paths.png (every path drawn)
    python real_go2/nav_sim.py --map      # the lab's zone map: every zone -> every other zone, from cold start
                                          #   -> results/real_q1/nav_map_sweep.json + nav_map_paths.png

--map world: the walkable area of results/real_q1/zone_map.npz is free space, everything else is an obstacle
for the simulated LiDAR. The camera is simulated with REAL classifier outputs: the validation session's
probability vectors (resnet18/val_predictions.csv) replayed in time order for the zone the dog is really in,
so the classifier's real confusions and their streaks are kept. The odometry frame is rotated and shifted
at random (unknown to the navigator), as on the dog.

Room 6 x 4 m; the target zone is x > 4.5. The robot follows the navigator's (vx, wz) commands exactly
(no slip); its LiDAR free-space profile is ray-cast with a body-wide corridor (3 parallel rays 0.25 m apart);
the zone classifier is simulated from geometry: it 'sees' the target when facing it within +-26 deg with no
view blocker in the way, and says 'target' everywhere inside the zone. It tests the decision logic and the
obstacle avoidance, not the real camera, LiDAR or legs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.goto_zone import ZoneNavigator, wrap  # noqa: E402
from real_go2.zone_map import ZoneMap, apply, map_cfg, rot  # noqa: E402

WALLS = [((0, 0), (6, 0)), ((6, 0), (6, 4)), ((6, 4), (0, 4)), ((0, 4), (0, 0))]


def box(x0, x1, y0, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def raycast(x, y, a, segs, look):
    """Distance from (x, y) along direction a to the nearest segment, or `look`."""
    dx, dy, best = math.cos(a), math.sin(a), look
    for (x1, y1), (x2, y2) in segs:
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x1 - x) * ey - (y1 - y) * ex) / den
        u = ((x1 - x) * dy - (y1 - y) * dx) / den
        if 0 < t < best and 0 <= u <= 1:
            best = t
    return best


def prob_for(k, conf, n):
    p = np.full(n, (1 - conf) / (n - 1))
    p[k] = conf
    return p


def run(classes, nav_cfg, obstacles, view_blockers, start=(1.0, 2.0, 0.0), target=(5.2, 2.0), zone_x=4.5,
        target_k=None, other_k=None, steps=12000, trace=None, turn_deadband=0.0):
    """Closed loop until arrived/failed. Returns (navigator, x, y, closest approach to any obstacle in m).
    If `trace` is a list, (x, y) of every step is appended to it. turn_deadband (rad/s): turning on the spot
    slower than this does nothing, like a legged robot that needs a minimum rate to step round."""
    n = len(classes)
    target_k = classes.index("shelves") if target_k is None else target_k
    other_k = classes.index("open_area") if other_k is None else other_k
    nb, look = int(nav_cfg["heading_bins"]), float(nav_cfg["lookahead_m"])
    segs = WALLS + list(obstacles)
    (x, y, h), (tx, ty) = start, target
    nav = ZoneNavigator(classes, nav_cfg, have_lidar=True)
    t, dt = 0.0, 0.1
    nav.start(target_k, t, h)
    closest = 9.0
    for _ in range(steps):
        if x > zone_x:
            prob = prob_for(target_k, 0.85, n)
        else:
            bearing, dist = math.atan2(ty - y, tx - x), math.hypot(tx - x, ty - y)
            visible = raycast(x, y, bearing, view_blockers, dist) >= dist - 1e-6
            prob = prob_for(target_k, 0.9, n) if (visible and abs(wrap(h - bearing)) < 0.45) else prob_for(other_k, 0.8, n)
        free = np.array([min(raycast(x + 0.25 * math.cos(a + s * math.pi / 2), y + 0.25 * math.sin(a + s * math.pi / 2),
                                     a, segs, look) for s in (-1, 0, 1)) for a in h + np.arange(nb) * 2 * math.pi / nb])
        vx, wz = nav.update(t, h, prob, free)
        if nav.state in ("arrived", "failed"):
            break
        x, y = x + vx * math.cos(h) * dt, y + vx * math.sin(h) * dt
        if trace is not None:
            trace.append((x, y))
        if abs(vx) < 0.05 and abs(wz) < turn_deadband:
            wz = 0.0
        h, t = wrap(h + wz * dt), t + dt
        closest = min(closest, min(raycast(x, y, a, segs, 9.0) for a in np.linspace(0, 2 * math.pi, 36, endpoint=False)))
    return nav, x, y, closest


class ProbStream:
    """Classifier outputs for the simulated camera: per zone, real probability vectors in recording order.
    While the dog stays in a zone the vectors are replayed in order (keeping streaks of mistakes); entering a
    zone starts at a random place in that zone's sequence."""

    def __init__(self, pools, rng):
        self.pools, self.rng, self.z, self.i = pools, rng, None, 0

    def __call__(self, z):
        if z != self.z:
            self.z, self.i = z, int(self.rng.integers(len(self.pools[z])))
        self.i = (self.i + 1) % len(self.pools[z])
        return self.pools[z][self.i]

    @classmethod
    def from_csv(cls, path, classes, rng):
        import csv
        rows = list(csv.DictReader(Path(path).open()))
        pools = {z: np.array([[float(r["p_" + c]) for c in classes] for r in rows if int(r["true"]) == z])
                 for z in range(len(classes))}
        return cls(pools, rng)


def map_free(zm, x, y, h, nb, look, step=0.1):
    """Simulated LiDAR free distance in nb body directions: 3 parallel rays (0.25 m apart) per direction,
    stopped by the first non-walkable cell."""
    out = np.full(nb, float(look))
    d = np.arange(step, look + 1e-9, step)
    for i in range(nb):
        a = h + i * 2 * math.pi / nb
        c, s_ = math.cos(a), math.sin(a)
        for o in (-0.25, 0.0, 0.25):
            px, py = x - o * s_ + d * c, y + o * c + d * s_
            ij = np.floor((np.stack([px, py], 1) - zm.lo) / zm.res).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < zm.shape[0]) & (ij[:, 1] < zm.shape[1])
            hit = ~ok
            hit[ok] = ~zm.walk[ij[ok, 0], ij[ok, 1]]
            if hit.any():
                out[i] = min(out[i], float(d[int(np.argmax(hit))]))
    return out


def true_zone(zm, xy):
    return int(np.argmax(zm.zone_probs(xy)))


def run_map(zm, classes, nav_cfg, stream, start, target_k, odom_tf, steps=9000, dt=0.1, frame_hz=5.0, trace=None,
            use_map=True):
    """Closed loop on the zone map from a cold start (not localised). start: map (x, y, heading). odom_tf:
    (rotation, shift) from the map frame to this run's odometry frame. Returns (navigator, final map xy,
    seconds, true zone at the end, closest approach to a non-walkable cell)."""
    nb, look = int(nav_cfg["heading_bins"]), float(nav_cfg["lookahead_m"])
    th_o, t_o = odom_tf
    nav = ZoneNavigator(classes, nav_cfg, have_lidar=True, zmap=zm if use_map else None)
    x, y, h = start
    t, next_frame, closest = 0.0, 0.0, 9.0

    def odom(x, y, h):
        p = apply(np.array([[x, y]]), th_o, t_o)[0]
        return p, wrap(h + th_o)

    nav.start(target_k, t, odom(x, y, h)[1])
    for _ in range(steps):
        p, ho = odom(x, y, h)
        prob = None
        if t >= next_frame:
            next_frame += 1.0 / frame_hz
            prob = stream(true_zone(zm, (x, y)))
        free = map_free(zm, x, y, h, nb, look)
        closest = min(closest, float(free.min()))
        vx, wz = nav.update(t, ho, prob, free, pos=p)
        if nav.state in ("arrived", "failed"):
            break
        nx_, ny_ = x + vx * math.cos(h) * dt, y + vx * math.sin(h) * dt
        ij = zm.cell((nx_, ny_))
        if zm.inside(ij) and zm.walk[ij]:          # the walkable edge is a wall
            x, y = nx_, ny_
        h, t = wrap(h + wz * dt), t + dt
        if trace is not None:
            trace.append((x, y))
    return nav, (x, y), t, true_zone(zm, (x, y)), closest


def map_sweep(cfg, starts_per_zone=3, seed=6010):
    """Every zone -> every other zone, starts_per_zone random walkable starts each, random odometry frames."""
    out_dir = TEST2 / cfg["results_dir"]
    nav_cfg, classes = cfg["navigation"], cfg["classes"]
    zm = ZoneMap.load(TEST2 / map_cfg(nav_cfg)["file"])
    rng = np.random.default_rng(seed)
    cells = np.argwhere(zm.walk & (zm.known >= 0.5))
    zone_of = np.array([true_zone(zm, zm.centre(c)) for c in cells])
    cases = []
    for s in range(len(classes)):
        for g in range(len(classes)):
            if g != s:
                for _ in range(starts_per_zone):
                    sx, sy = zm.centre(cells[rng.choice(np.flatnonzero(zone_of == s))])
                    cases.append((s, g, (float(sx), float(sy), float(rng.uniform(-math.pi, math.pi))),
                                  (float(rng.uniform(-math.pi, math.pi)), rng.uniform(-5, 5, 2))))
    result, paths = {}, []
    for mode in ("zone_map", "classifier_only"):        # the same cases and camera sequence for both
        stream = ProbStream.from_csv(out_dir / "resnet18/val_predictions.csv", classes, np.random.default_rng(seed))
        rows = []
        print(mode)
        for s, g, start, tf in cases:
            tr = [start[:2]]
            nav, (x, y), secs, zend, closest = run_map(zm, classes, nav_cfg, stream, start, g, tf, trace=tr,
                                                       use_map=mode == "zone_map")
            ok = nav.state == "arrived" and zend == g
            rows.append({"from": classes[s], "to": classes[g], "start": [round(v, 2) for v in start],
                         "result": nav.state, "really_in": classes[zend], "correct": ok,
                         "seconds": round(secs, 1), "legs": nav.legs, "explored": nav.explored,
                         "closest_edge_m": round(closest, 2), "msg": nav.msg})
            if mode == "zone_map":
                paths.append((ok, nav.state, tr, classes[g]))
            print(f"  {classes[s]:>9} -> {classes[g]:<9} {nav.state:8} really in {classes[zend]:<9} "
                  f"{secs:5.0f} s  {nav.msg[:70]}")
        result[mode] = {"summary": {
            "runs": len(rows), "arrived": sum(r["result"] == "arrived" for r in rows),
            "arrived_in_the_right_zone": sum(r["correct"] for r in rows),
            "wrong_arrivals": sum(r["result"] == "arrived" and not r["correct"] for r in rows),
            "median_seconds_when_correct": float(np.median([r["seconds"] for r in rows if r["correct"]]))
            if any(r["correct"] for r in rows) else None}, "runs_detail": rows}
    path = out_dir / "nav_map_sweep.json"
    path.write_text(json.dumps({"navigation": nav_cfg, "seed": seed, "steps_limit_s": 900,
                                "camera": "resnet18/val_predictions.csv replayed per zone (same for both modes)",
                                **result}, indent=1))
    print(json.dumps({m: r["summary"] for m, r in result.items()}, indent=1))
    plot_map_paths(zm, classes, paths, path.with_name("nav_map_paths.png"))
    print(f"-> {path.relative_to(TEST2)}, {path.with_name('nav_map_paths.png').relative_to(TEST2)}")


def plot_map_paths(zm, classes, paths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a7de1", "#e8602c", "#1bb27f", "#f0a30a"]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 5))
    ext = [zm.lo[0], zm.lo[0] + zm.shape[0] * zm.res, zm.lo[1], zm.lo[1] + zm.shape[1] * zm.res]
    k = zm.prob.argmax(2)
    for ax, name in zip(axes, classes):
        rgb = np.ones(zm.shape + (3,))
        for z in range(len(classes)):
            rgb[zm.walk & (k == z)] = 0.35 * np.array(matplotlib.colors.to_rgb(cols[z])) + 0.65
        ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=ext)
        mine = [p for p in paths if p[3] == name]
        for ok, state, tr, _ in mine:
            a = np.array(tr)
            ax.plot(a[:, 0], a[:, 1], color="#2e8b57" if ok else "#d1495b", lw=1)
            ax.plot(a[0, 0], a[0, 1], "o", color="black", ms=3)
            ax.plot(a[-1, 0], a[-1, 1], "x" if not ok else "s", color="black", ms=4)
        ax.set_title(f"to {name}: correct {sum(p[0] for p in mine)}/{len(mine)}", fontsize=10)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Zone-map navigation in simulation (real map, real classifier outputs); green = arrived in "
                 "the right zone", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


LAYOUTS = {"low desk blocks the walk, not the view": (box(2.3, 2.7, 1.0, 3.0), False),
           "tall shelf hides the zone": (box(2.3, 2.7, 0.8, 3.2), True)}
STARTS = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (0.6, 2.0)]
HEADINGS = [0.0, math.pi / 2, math.pi, -math.pi / 2]


def plot(paths, path):
    """One panel per layout: room, obstacle, target zone, and every run's path (green arrived, red not)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(paths), figsize=(6 * len(paths), 4.3))
    for ax, (name, runs) in zip(np.atleast_1d(axes), paths.items()):
        obs, _ = LAYOUTS[name]
        ax.add_patch(plt.Rectangle((4.5, 0), 1.5, 4, color="#f2c14e", alpha=0.35, lw=0))
        ax.text(5.25, 3.75, "target zone", ha="center", fontsize=9)
        xs = [p[0][0] for p in obs] + [obs[0][0][0]]
        ys = [p[0][1] for p in obs] + [obs[0][0][1]]
        ax.fill(xs, ys, color="#555555")
        for ok, tr, (sx, sy, _) in runs:
            a = np.array(tr)
            ax.plot(a[:, 0], a[:, 1], color="#2e8b57" if ok else "#d1495b", lw=1.1, alpha=0.75)
            ax.plot(sx, sy, "o", color="black", ms=3)
        n_ok = sum(r[0] for r in runs)
        ax.set_title(f"{name}: arrived {n_ok}/{len(runs)}", fontsize=10)
        ax.set_xlim(0, 6), ax.set_ylim(0, 4), ax.set_aspect("equal")
        ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cfg = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
    if "--map" in sys.argv[1:]:
        map_sweep(cfg)
        return
    out = {"navigation": cfg["navigation"], "layouts": {}}
    paths = {}
    for name, (obs, hides) in LAYOUTS.items():
        rows, paths[name] = [], []
        for sx, sy in STARTS:
            for sh in HEADINGS:
                tr = [(sx, sy)]
                nav, x, y, c = run(cfg["classes"], cfg["navigation"], obs, obs if hides else [], start=(sx, sy, sh),
                                   trace=tr)
                paths[name].append((nav.state == "arrived", tr, (sx, sy, sh)))
                rows.append({"start": [sx, sy, round(math.degrees(sh))], "result": nav.state, "legs": nav.legs,
                             "explored": nav.explored, "end_xy": [round(x, 2), round(y, 2)],
                             "closest_obstacle_m": round(c, 3), "msg": nav.msg})
        ok = sum(r["result"] == "arrived" for r in rows)
        out["layouts"][name] = {"arrived": ok, "runs": len(rows),
                                "closest_obstacle_m": min(r["closest_obstacle_m"] for r in rows), "runs_detail": rows}
        print(f"{name}: arrived {ok}/{len(rows)}, closest approach to an obstacle "
              f"{out['layouts'][name]['closest_obstacle_m']:.2f} m (from the dog's centre)")
    path = TEST2 / cfg["results_dir"] / "nav_sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    plot(paths, path.with_name("nav_sim_paths.png"))
    print(f"-> {path.relative_to(TEST2)}, {path.with_name('nav_sim_paths.png').relative_to(TEST2)}")


if __name__ == "__main__":
    main()
