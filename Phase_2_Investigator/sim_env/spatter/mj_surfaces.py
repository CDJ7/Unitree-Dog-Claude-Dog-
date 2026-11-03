"""Droplet landing against the real simulated scene, by MuJoCo ray casting.

`MjSurfaces` is a drop-in replacement for `geometry.SurfaceSet`: it offers
`first_hits(p0, p1)` for flight segments, so `physics.fly` and the generator
run unchanged. Each flight step is tested with `mj_ray` against the *visible*
geometry (the visual meshes the camera renders, groups 0-2), so stains land
exactly where they are seen: on walls, floors, windows, the bed's duvet, the
sofa's cushions, not on the houses' coarse furniture collision boxes.

Hits are grouped into planar `Rect`s (same geom, same plane within 1 cm and 8
degrees) so stains on one wall share one surface frame, as in the analytic
version. The rect's normal points toward the side the droplet came from (the
visible face) and its extent is the geom's bounding box projected on the plane.

Not tested: mannequins and the robot (by geom id), so droplets launched from
inside a mannequin (e.g. the victim's head) are not blocked by it; this also
keeps the visuals non-graphic.

mj_multiRay is not used: in MuJoCo 3.3.7 it misses some geoms (see
docs/plan.md, section 10); mj_ray costs ~17 us per call here.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .geometry import Hit, Rect, frame_from_normal

VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)


class MjSurfaces:
    def __init__(self, model, data, exclude_geoms=(), groups=VISIBLE_GROUPS, room_prefix="room_geometry"):
        self.model, self.data = model, data
        self.exclude = set(int(g) for g in exclude_geoms)
        self.groups = np.asarray(groups, np.uint8)
        self.room_prefix = room_prefix
        self.rects: list[Rect] = []
        self.by_id: dict[str, Rect] = {}
        self.rect_geom: dict[str, int] = {}
        self._by_geom: dict[int, list[int]] = {}
        self._gid = np.zeros(1, np.int32)

    def __iter__(self):
        return iter(self.rects)

    def __len__(self):
        return len(self.rects)

    # ------------------------------------------------------------ ray casting
    def cast(self, p, vec, max_frac=1.0):
        """First non-excluded visible geom hit by p + t*vec, 0 <= t <= max_frac.

        Returns (t, geom) or (None, -1).
        """
        p = np.asarray(p, float)
        vec = np.asarray(vec, float)
        t0 = 0.0
        for _ in range(8):                      # step through excluded geoms
            t = mujoco.mj_ray(self.model, self.data, p + t0 * vec, vec, self.groups, 1, -1, self._gid)
            if t < 0 or t0 + t > max_frac:
                return None, -1
            g = int(self._gid[0])
            if g not in self.exclude:
                return t0 + t, g
            t0 += t + 1e-4 / max(np.linalg.norm(vec), 1e-12)
        return None, -1

    def _normal_at(self, point, direction, geom):
        """Surface normal at a hit, from two neighbouring parallel rays (faces the ray origin)."""
        d = direction / np.linalg.norm(direction)
        a, b = frame_from_normal(d)
        pts = []
        for off in (a, b):
            start = point - d * 0.05 + off * 0.003
            t, g = self.cast(start, d * 0.1)
            if t is None or g != geom:
                start = point - d * 0.05 - off * 0.003
                t, g = self.cast(start, d * 0.1)
                if t is None or g != geom:
                    return -d
                pts.append(-(start + d * 0.1 * t - point))
            else:
                pts.append(start + d * 0.1 * t - point)
        n = np.cross(pts[0], pts[1])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return -d
        n /= nn
        return n if n @ d < 0 else -n

    # --------------------------------------------------------------- surfaces
    def _kind(self, geom, normal, point):
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if normal[2] > 0.9 and point[2] < 0.05:
            return "floor"
        if normal[2] < -0.9 and point[2] > 2.0:
            return "ceiling"
        if name.startswith(self.room_prefix) and abs(normal[2]) < 0.3:
            return "wall"
        return "furniture"

    def _rect_for(self, geom, point, normal):
        for k in self._by_geom.get(geom, []):
            r = self.rects[k]
            if abs(r.normal @ normal) > np.cos(np.radians(8)) and abs((point - r.center) @ r.normal) < 0.01:
                if r.normal @ normal < 0:          # same plane seen from the other side
                    continue
                return k
        m, d = self.model, self.data
        R = d.geom_xmat[geom].reshape(3, 3)
        aabb = m.geom_aabb[geom]
        c_world = d.geom_xpos[geom] + R @ aabb[:3]
        half = aabb[3:]
        center = c_world - ((c_world - point) @ normal) * normal
        u, v = frame_from_normal(normal)
        hu = float(np.sum(np.abs(u @ R) * half))
        hv = float(np.sum(np.abs(v @ R) * half))
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"
        rid = f"{gname.replace('_visual_visual', '').replace('room_geometry_', '')}#{len(self._by_geom.get(geom, []))}"
        rect = Rect(rid, self._kind(geom, normal, point), center, u, v, max(hu, 0.01), max(hv, 0.01),
                    material=gname)
        self.rects.append(rect)
        self.by_id[rid] = rect
        self.rect_geom[rid] = geom
        self._by_geom.setdefault(geom, []).append(len(self.rects) - 1)
        return len(self.rects) - 1

    # ------------------------------------------------------- SurfaceSet API
    def first_hits(self, p0, p1):
        p0 = np.atleast_2d(np.asarray(p0, float))
        p1 = np.atleast_2d(np.asarray(p1, float))
        n = len(p0)
        idx = np.full(n, -1, int)
        frac = np.full(n, np.inf)
        pts = p0.copy()
        for i in range(n):
            seg = p1[i] - p0[i]
            t, g = self.cast(p0[i], seg)
            if t is None:
                pts[i] = p1[i]
                continue
            point = p0[i] + t * seg
            normal = self._normal_at(point, seg, g)
            idx[i] = self._rect_for(g, point, normal)
            frac[i] = t
            pts[i] = point
        return idx, frac, pts

    def first_hit(self, p0, p1):
        idx, frac, pts = self.first_hits(np.asarray(p0)[None], np.asarray(p1)[None])
        if idx[0] < 0:
            return None
        return Hit(self.rects[idx[0]], pts[0], float(frac[0]))

    def surface_at(self, point, normal_hint):
        """Rect under a point, found by casting toward it along -normal_hint (for smears)."""
        n = np.asarray(normal_hint, float)
        n /= np.linalg.norm(n)
        start = np.asarray(point, float) + 0.1 * n
        t, g = self.cast(start, -0.2 * n)
        if t is None:
            raise ValueError(f"no surface under {point} along {-n}")
        hit = start - 0.2 * n * t
        return self.rects[self._rect_for(g, hit, self._normal_at(hit, -n, g))], hit

    def snap(self, point, normal, surface_id, tol=0.02):
        """Project a point onto the real surface along -normal; None if it falls off that geom."""
        start = np.asarray(point, float) + tol * np.asarray(normal)
        t, g = self.cast(start, -2 * tol * np.asarray(normal))
        if t is None or g != self.rect_geom.get(surface_id, -2):
            return None
        return start - 2 * tol * np.asarray(normal) * t

    def to_dict(self):
        return [dict(r.to_dict(), geom=self.rect_geom[r.id]) for r in self.rects]
