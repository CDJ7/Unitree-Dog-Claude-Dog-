"""Close-up tour of detected evidence (C5): photograph every part of a cluster from close enough to measure
single droplets.

    frames = closeup_tour(nav, cluster)      # cluster: an EvidenceMap cluster (xyz, normal, pts)

Phase 2's measurement is accurate to ~1-2 deg of impact angle up to ~1 m from a stain and degrades beyond
(evaluation/results/bpa_eval.json), and the inspect behaviour's views stand 0.85-2.2 m back. So:
- aim points cover the cluster's detected member positions (greedy, `aim_radius` apart, the densest first);
- for each aim point, a standing spot puts the camera `floor_h` m (floor stains) or `wall_h` m (wall
  stains, horizontally) from it. The spot must be free, have room around it, not be on floor evidence
  (keep-out), and have a clear line of sight;
- the dog drives there (navigation, safety layer and keep-out on). Then every remaining aim point within
  `max_slant` m of the camera gets a close-up: turn to it, hold still, and take a 1920 x 1080, 40 deg frame
  (`evidence_frame`).

Simulation note: the close-up camera is pitched up to `max_pitch_deg` to look at the target (Phase 2's
evidence-camera emulation). The real Go2 can pitch its body only ~20-25 deg, and its front camera is
1280 x 720 at ~100 deg, far too coarse for droplets. On the real dog, close-ups need a dedicated
higher-resolution camera (plan step C5, real).
"""
from __future__ import annotations

import numpy as np

from navigation.inspect import line_of_sight


def aim_points(pts, radius=0.30, max_aims=12):
    """Greedy cover of the cluster's member positions: densest first, aims `radius` apart."""
    P = np.asarray(pts, float)
    if len(P) == 0:
        return []
    left = np.ones(len(P), bool)
    out = []
    while left.any() and len(out) < max_aims:
        Q = P[left]
        dens = (np.linalg.norm(Q[:, None] - Q[None], axis=2) < radius).sum(1)
        a = Q[int(np.argmax(dens))]
        out.append(a)
        left &= np.linalg.norm(P - a, axis=1) >= radius
    return out


def _camera_offset(sim):
    """Camera position relative to the base, in the base frame (x forward)."""
    st = sim.state()
    cam = sim.camera_pose()[:3, 3]
    c, s = np.cos(st.yaw), np.sin(st.yaw)
    d = cam[:2] - st.position[:2]
    return float(c * d[0] + s * d[1])


def standing_spot(nav, aim, normal, cam_fwd, floor_h=0.50, wall_h=0.50, min_clear=0.30):
    """Robot base (xy) for a close-up of `aim`, or None."""
    g = nav.grid
    aim = np.asarray(aim, float)
    n = np.asarray(normal, float)
    clear = g.obstacle_distance()
    free = g.free()
    ko = getattr(g, "keepout", None)
    ko_dist = None
    if ko is not None and ko.any():
        from scipy.ndimage import distance_transform_edt
        ko_dist = distance_transform_edt(~ko) * g.res
    st = nav.sim.state()
    if abs(n[2]) > 0.8:                                   # floor / top: approach from the robot's side
        base_dir = st.position[:2] - aim[:2]
        dists = (floor_h, floor_h + 0.15, floor_h - 0.1, floor_h + 0.3)
        angles = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]
    else:                                                 # wall: stand in front of it
        base_dir = n[:2]
        dists = (wall_h, wall_h + 0.2, wall_h + 0.4, wall_h + 0.6, wall_h + 0.8)   # furniture / a body may be in front
        angles = [0, 15, -15, 30, -30, 45, -45, 60, -60, 70, -70]
    base_dir = base_dir / max(np.linalg.norm(base_dir), 1e-9)
    for h in dists:
        for ang in angles:
            a = np.radians(ang)
            u = np.array([np.cos(a) * base_dir[0] - np.sin(a) * base_dir[1],
                          np.sin(a) * base_dir[0] + np.cos(a) * base_dir[1]])
            cam = aim[:2] + u * h                          # camera position (xy)
            base = cam + u * cam_fwd                       # the base sits behind the camera, facing the aim
            r, c = g.to_cell(base)
            if not g.inside(r, c) or not free[r, c] or clear[r, c] < min_clear:
                continue
            if ko_dist is not None and ko_dist[r, c] < 0.40:
                continue
            if not line_of_sight(g, base, aim[:2], stop_short=0.10 if abs(n[2]) > 0.8 else 0.15):
                continue
            return base
    return None


def closeup_tour(nav, cluster, max_slant=1.15, target_slant=1.8, max_pitch_deg=50.0, hold_s=0.4, out_dir=None,
                 log=None, camera_height=None):
    """Close-up frames [(aim, CameraFrame)] covering one evidence cluster."""
    sim = nav.sim
    say = log or nav.say
    pts = cluster.get("pts") or [cluster["xyz"]]
    aims = aim_points(pts)
    cam_fwd = _camera_offset(sim)
    shot = [False] * len(aims)
    frames = []
    for i, aim in enumerate(aims):
        if shot[i]:
            continue
        base = standing_spot(nav, aim, cluster["normal"], cam_fwd)
        if base is None:
            say(f"  close-ups {cluster['id']}: no standing spot for aim {i}")
            continue
        ok = nav.go_to(base, timeout=60.0, tol=0.25, final_tol=0.15)
        if not ok and np.linalg.norm(sim.state().position[:2] - base) > 0.6:
            say(f"  close-ups {cluster['id']}: could not reach the spot for aim {i}")
            continue
        cam = sim.camera_pose()[:3, 3].copy()
        if camera_height is not None:
            cam[2] = camera_height
        for k, a in enumerate(aims):
            # the aim the dog came for is always photographed (navigation may stop a little short);
            # other aims only when close enough to measure well
            limit = target_slant if k == i else max_slant
            if shot[k] or np.linalg.norm(a - cam) > limit:
                continue
            nav.turn_to(a[:2])
            nav.hold(hold_s)
            if camera_height is None:
                fr = sim.evidence_frame(a, fovy=40.0, width=1920, height=1080, max_pitch_deg=max_pitch_deg)
            else:                                      # emulated mast camera (investigate.py --camera-height)
                c = sim.camera_pose()[:3, 3].copy()
                c[2] = camera_height
                d = a - c
                pitch = np.clip(np.arctan2(d[2], np.hypot(d[0], d[1])), -np.radians(70), np.radians(max_pitch_deg))
                yaw = np.arctan2(d[1], d[0])
                fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
                fr = sim.render_camera(c, fwd, fovy=40.0, width=1920, height=1080)
            fr.frame_id = f"{cluster['id']}_a{k}"
            frames.append((a, fr))
            shot[k] = True
            if out_dir is not None:
                from navigation.inspect import save_frame
                save_frame(fr, f"{out_dir}/{fr.frame_id}")
            cam = sim.camera_pose()[:3, 3].copy()
            if camera_height is not None:
                cam[2] = camera_height
    say(f"  close-ups {cluster['id']}: {len(frames)} frames, {sum(shot)}/{len(aims)} aim points")
    return frames
