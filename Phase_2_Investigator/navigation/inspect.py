"""Inspect point (A4): approach a stain region and capture it from 2-3 angles.

    inspect_clusters(nav, gt)            # all ground-truth clusters (Stage B: detections instead)
    inspect_point(nav, point, normal)    # one target

For a target point with surface normal n:
- walls / furniture faces: stand in front of it along the horizontal normal;
  floors and table tops: stand back from it on the robot's side;
- standoff distance grows with the height difference to the camera, so the
  emulated pitch stays within +-40 deg (the real Go2 can pitch its body; the
  sim gait cannot, so the close-up camera is pitched instead: sim/real note);
- views at 0 and +-`spread` degrees around the target, each moved if its spot
  is not free on the map;
- at each view: drive there (cmd_vel, safety layer on), face the target, hold
  still 0.5 s, then capture the 320x240 sensor frame and a 1920x1080 evidence
  close-up (40 deg FOV) with depth, intrinsics and pose.
Each view records what fraction of the target's ground-truth stains are in the
close-up and not occluded (a check of the behaviour, not used by it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAM_H = 0.41          # Go2 front camera height when standing (m)


def inspection_targets(gt, merge=0.9, min_stains=4):
    """Merge nearby ground-truth clusters on similar surfaces into inspection targets."""
    cl = [c for c in gt.get("clusters", []) if c["n_stains"] >= min_stains or "transfer" in c["mechanisms"]]
    targets = []
    for c in sorted(cl, key=lambda c: -c["n_stains"]):
        cen, n = np.asarray(c["center"]), np.asarray(c["normal"])
        for t in targets:
            if np.linalg.norm(t["center"] - cen) < merge and float(t["normal"] @ n) > 0.8:
                w = t["n_stains"]
                t["center"] = (t["center"] * w + cen * c["n_stains"]) / (w + c["n_stains"])
                t["n_stains"] += c["n_stains"]
                t["clusters"].append(c["id"])
                t["stain_ids"] += c["stain_ids"]
                break
        else:
            targets.append({"center": cen.astype(float), "normal": n.astype(float), "n_stains": c["n_stains"],
                            "clusters": [c["id"]], "surface_kind": c["surface_kind"],
                            "stain_ids": list(c["stain_ids"]), "mechanisms": c["mechanisms"]})
    by_id = {s["id"]: s for s in gt["stains"]}
    for i, t in enumerate(targets):
        t["id"] = f"t{i}"
        pts = np.array([by_id[s]["center_world"] for s in t["stain_ids"] if s in by_id] or [t["center"]])
        t["radius"] = float(np.percentile(np.linalg.norm(pts - t["center"], axis=1), 90))
    return targets


def line_of_sight(grid, a, b, stop_short=0.15):
    """No tall obstacle on the segment a -> b (ignoring the last `stop_short` metres).

    Only obstacles seen by the planar scan (~0.45 m up) block the view: the camera
    (0.41 m, looking down at the target) sees over low things like a lying mannequin."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    if L <= stop_short:
        return True
    n = int((L - stop_short) / (grid.res * 0.5)) + 1
    pts = a[None] + (b - a)[None] * (np.linspace(0, (L - stop_short) / L, n)[:, None])
    r, c = grid.to_cell(pts)
    ok = grid.inside(r, c)
    return not np.any((grid.logodds > 0.6)[r[ok], c[ok]])


EVIDENCE_FOVY = 55.0


def standoff_views(nav, center, normal, n_views=3, spread_deg=25.0, radius=0.2):
    """Robot positions (xy, distance, angle) for up to n_views views of a target.

    Each must be on free map cells with room around it and a clear line of sight
    to the target. Walls are viewed along their normal +- spread; floors and
    tops from any side, starting from the robot's side.
    """
    center = np.asarray(center, float)
    n = np.asarray(normal, float)
    h = center[2]
    fit = radius / np.tan(np.radians(EVIDENCE_FOVY / 2)) * 1.1        # the whole region in the close-up
    dist = float(np.clip(max(abs(h - CAM_H) / np.tan(np.radians(35)), fit), 0.85, 2.2))
    if abs(n[2]) < 0.7:                                   # wall / vertical face
        base = n[:2] / np.linalg.norm(n[:2])
        wanted = [0.0, spread_deg, -spread_deg]
        search = [0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60]
    else:                                                 # floor or top surface
        to_robot = nav.sim.state().position[:2] - center[:2]
        base = to_robot / max(np.linalg.norm(to_robot), 1e-6)
        wanted = [0.0, spread_deg, -spread_deg]
        search = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]
    clear = nav.grid.obstacle_distance()
    free = nav.grid.free()
    for min_clear in (0.40, 0.32):
        views = _pick_views(nav, center, base, dist, wanted, search, clear, free, min_clear, n_views)
        if views:
            return views
    return []


def _pick_views(nav, center, base, dist, wanted, search, clear, free, min_clear, n_views):
    views = []
    used = []
    for want in wanted:
        for extra in search:
            ang = want + extra
            if any(abs(((ang - u) + 180) % 360 - 180) < 15 for u in used):
                continue
            a = np.radians(ang)
            d2 = np.array([np.cos(a) * base[0] - np.sin(a) * base[1], np.sin(a) * base[0] + np.cos(a) * base[1]])
            found = None
            for dd in (dist, dist + 0.25, dist - 0.15, dist + 0.5, dist + 0.8):
                xy = center[:2] + d2 * dd
                r, c = nav.grid.to_cell(xy)
                if nav.grid.inside(r, c) and free[r, c] and clear[r, c] >= min_clear and \
                        line_of_sight(nav.grid, xy, center[:2]):
                    found = (xy, dd, ang)
                    break
            if found:
                views.append(found)
                used.append(ang)
                break
        if len(views) >= n_views:
            break
    return views


def stain_visibility(frame, gt, stain_ids, tol=0.03):
    """Fraction of the given stains inside the frame and not occluded (depth test)."""
    by_id = {s["id"]: s for s in gt["stains"]}
    pts = np.array([by_id[i]["center_world"] for i in stain_ids if i in by_id])
    if not len(pts):
        return 0.0
    uv, z = frame.project(pts)
    H, W = frame.depth_m.shape
    inside = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    vis = np.zeros(len(pts), bool)
    ui = np.clip(uv[inside, 0].astype(int), 0, W - 1)
    vi = np.clip(uv[inside, 1].astype(int), 0, H - 1)
    vis[inside] = np.abs(frame.depth_m[vi, ui] - z[inside]) < tol + 0.01 * z[inside]
    return float(vis.mean())


def save_frame(frame, path_stem, jpeg=True):
    from PIL import Image
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame.rgb)
    rgb_path = path_stem.with_suffix(".jpg" if jpeg else ".png")
    img.save(rgb_path, quality=92) if jpeg else img.save(rgb_path)
    d = frame.depth_m
    mm = np.where(np.isfinite(d), np.clip(d * 1000, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(mm).save(path_stem.with_name(path_stem.name + "_depth.png"))
    meta = {"t": frame.t, "K": frame.K.tolist(), "T_world_cam": frame.T_world_cam.tolist(),
            "width": int(frame.rgb.shape[1]), "height": int(frame.rgb.shape[0]), "frame_id": frame.frame_id,
            "depth": "16-bit PNG, millimetres along the optical axis, 0 = no return"}
    path_stem.with_name(path_stem.name + "_meta.json").write_text(json.dumps(meta, indent=1))
    return str(rgb_path)


def stains_seen(frame, gt, stain_ids, tol=0.03):
    """Set of stain ids inside the frame and not occluded."""
    by_id = {s["id"]: s for s in gt["stains"]}
    ids = [i for i in stain_ids if i in by_id]
    if not ids:
        return set()
    pts = np.array([by_id[i]["center_world"] for i in ids])
    uv, z = frame.project(pts)
    H, W = frame.depth_m.shape
    seen = set()
    for i, (u, v), zz in zip(ids, uv, z):
        if zz > 0.05 and 0 <= u < W and 0 <= v < H and abs(frame.depth_m[int(v), int(u)] - zz) < tol + 0.01 * zz:
            seen.add(i)
    return seen


def inspect_point(nav, center, normal, gt=None, stain_ids=(), name="target", out_dir=None, n_views=3, radius=0.2):
    views = standoff_views(nav, center, normal, n_views, radius=radius)
    seen_all = set()
    result = {"id": name, "center": np.asarray(center).round(3).tolist(), "normal": np.asarray(normal).round(3).tolist(),
              "planned_views": len(views), "views": []}
    nav.say(f"inspect {name} at {np.round(center, 2).tolist()}: {len(views)} views")
    for k, (xy, dist, ang) in enumerate(views):
        ok = nav.go_to(xy, timeout=60.0, tol=0.25, final_tol=0.15)
        if not ok:
            nav.say(f"  view {k}: could not reach {np.round(xy, 2).tolist()}")
            continue
        nav.turn_to(center[:2])
        nav.hold(0.5)
        st = nav.sim.state()
        sensor = nav.sim.camera()
        close = nav.sim.evidence_frame(center, fovy=EVIDENCE_FOVY)
        v = {"view": k, "robot_xy": st.position[:2].round(3).tolist(), "yaw_deg": round(float(np.degrees(st.yaw)), 1),
             "distance_m": round(float(np.linalg.norm(close.T_world_cam[:3, 3] - center)), 3),
             "angle_deg": round(float(ang), 1)}
        if gt is not None and stain_ids:
            v["visible_fraction_closeup"] = round(stain_visibility(close, gt, stain_ids), 3)
            seen_all |= stains_seen(close, gt, stain_ids)
            v["in_sensor_frame_fraction"] = round(stain_visibility(sensor, gt, stain_ids, tol=0.15), 3)
        if out_dir:
            v["closeup"] = save_frame(close, Path(out_dir) / f"{name}_v{k}_closeup")
            v["sensor"] = save_frame(sensor, Path(out_dir) / f"{name}_v{k}_sensor", jpeg=False)
        result["views"].append(v)
        nav.say(f"  view {k}: {v.get('distance_m')} m, stains visible {v.get('visible_fraction_closeup')}")
    n_target = len([i for i in stain_ids if i in {s["id"] for s in gt["stains"]}]) if gt is not None else 0
    result["seen_fraction_all_views"] = round(len(seen_all) / n_target, 3) if n_target else None
    nav.say(f"  {name}: {result['seen_fraction_all_views']} of its stains seen across {len(result['views'])} views")
    return result


def inspect_clusters(nav, gt, out_dir=None, n_views=3, max_targets=8):
    targets = inspection_targets(gt)[:max_targets]
    nav.inspected = []
    for t in targets:
        res = inspect_point(nav, t["center"], t["normal"], gt, t["stain_ids"], name=t["id"], out_dir=out_dir,
                            n_views=n_views, radius=t["radius"])
        res.update(n_stains=t["n_stains"], clusters=t["clusters"], surface_kind=t["surface_kind"],
                   mechanisms=t["mechanisms"])
        nav.inspected.append(res)
    return nav.inspected
