#!/usr/bin/env python3
"""A small 2-D test world for the go-to-zone navigator (used by tests/test_goto_zone.py and the sweep below).

    python real_go2/nav_sim.py            # 2 layouts x 4 starts x 4 headings -> results/real_q1/nav_sweep.json
                                          #   and nav_sim_paths.png (every path drawn)

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
