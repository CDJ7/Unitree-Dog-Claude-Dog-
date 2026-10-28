"""Surfaces the droplets can land on, as oriented rectangles.

This is the stand-alone stand-in for MuJoCo ray casting: a room is a floor,
a ceiling and four walls, furniture is a box of five rectangles (no bottom).
Every rectangle has a local (u, v) frame with cross(u, v) = normal, the
normal pointing into the free space (into the room, out of the furniture).
Stain positions are stored in these (u, v) coordinates, which is also how
the decal textures are laid out. A MuJoCo-backed version only has to offer
`first_hit` with the same return value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def frame_from_normal(normal, up_hint=None):
    """(u, v) unit axes for a plane with this normal, with cross(u, v) = normal.

    v is the up direction projected onto the plane (so on walls v points up and
    u runs horizontally); for floors and ceilings v is world +y.
    """
    n = _unit(normal)
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(n[2]) > 0.9 else UP
    v = np.asarray(up_hint, dtype=float) - np.dot(up_hint, n) * n
    v = _unit(v)
    u = np.cross(v, n)
    return u, v


@dataclass
class Rect:
    id: str
    kind: str                 # floor | wall | ceiling | furniture
    center: np.ndarray
    u: np.ndarray
    v: np.ndarray
    half_u: float
    half_v: float
    material: str = "plaster"
    normal: np.ndarray = field(init=False)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.u = _unit(self.u)
        self.v = _unit(self.v)
        self.normal = _unit(np.cross(self.u, self.v))

    @classmethod
    def from_normal(cls, id, kind, center, normal, half_u, half_v, material="plaster", up_hint=None):
        u, v = frame_from_normal(normal, up_hint)
        return cls(id, kind, np.asarray(center, float), u, v, half_u, half_v, material)

    def to_local(self, p):
        d = np.asarray(p, float) - self.center
        return np.stack([d @ self.u, d @ self.v], axis=-1)

    def to_world(self, uv):
        uv = np.asarray(uv, float)
        return self.center + uv[..., :1] * self.u + uv[..., 1:2] * self.v

    def contains_local(self, uv, pad=0.0):
        uv = np.asarray(uv, float)
        return (np.abs(uv[..., 0]) <= self.half_u + pad) & (np.abs(uv[..., 1]) <= self.half_v + pad)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "material": self.material,
                "center": self.center.tolist(), "u": self.u.tolist(), "v": self.v.tolist(),
                "normal": self.normal.tolist(), "half_u": self.half_u, "half_v": self.half_v}

    @classmethod
    def from_dict(cls, d):
        return cls(d["id"], d["kind"], np.array(d["center"]), np.array(d["u"]), np.array(d["v"]),
                   d["half_u"], d["half_v"], d.get("material", "plaster"))


@dataclass
class Hit:
    surface: Rect
    point: np.ndarray
    frac: float               # fraction along the tested segment


class SurfaceSet:
    """A collection of rectangles with vectorised segment tests."""

    def __init__(self, rects):
        self.rects = list(rects)
        ids = [r.id for r in self.rects]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate surface ids")
        self.by_id = {r.id: r for r in self.rects}

    def __iter__(self):
        return iter(self.rects)

    def __len__(self):
        return len(self.rects)

    def first_hits(self, p0, p1):
        """For N segments p0[i] -> p1[i], the first rectangle each one crosses.

        Returns (index array, frac array, point array). index is -1 where no hit.
        Rectangles are two-sided: a droplet can land on either face.
        """
        p0 = np.atleast_2d(np.asarray(p0, float))
        p1 = np.atleast_2d(np.asarray(p1, float))
        seg = p1 - p0
        n_seg = len(p0)
        best = np.full(n_seg, np.inf)
        idx = np.full(n_seg, -1, dtype=int)
        for k, r in enumerate(self.rects):
            denom = seg @ r.normal
            num = (r.center - p0) @ r.normal
            ok = np.abs(denom) > 1e-12
            t = np.full(n_seg, np.inf)
            t[ok] = num[ok] / denom[ok]
            ok &= (t >= 0.0) & (t <= 1.0) & (t < best)
            if not ok.any():
                continue
            pts = p0[ok] + t[ok, None] * seg[ok]
            inside = r.contains_local(r.to_local(pts))
            sel = np.flatnonzero(ok)[inside]
            best[sel] = t[sel]
            idx[sel] = k
        pts = p0 + np.where(np.isfinite(best), best, 0.0)[:, None] * seg
        return idx, best, pts

    def first_hit(self, p0, p1):
        idx, frac, pts = self.first_hits(p0[None], p1[None])
        if idx[0] < 0:
            return None
        return Hit(self.rects[idx[0]], pts[0], float(frac[0]))

    def to_dict(self):
        return [r.to_dict() for r in self.rects]

    @classmethod
    def from_dict(cls, items):
        return cls([Rect.from_dict(d) for d in items])


# ---------------------------------------------------------------- builders

def room_surfaces(size, origin=(0.0, 0.0, 0.0), ceiling=True, prefix="room", materials=None):
    """Floor, four walls and (optionally) a ceiling for a box room.

    size = (width along x, depth along y, height); origin = the floor corner
    with the smallest x and y. Normals point into the room.
    """
    w, d, h = size
    ox, oy, oz = origin
    m = {"floor": "wood", "wall": "plaster", "ceiling": "plaster"}
    m.update(materials or {})
    cx, cy = ox + w / 2, oy + d / 2
    rects = [
        Rect.from_normal(f"{prefix}_floor", "floor", (cx, cy, oz), (0, 0, 1), w / 2, d / 2, m["floor"]),
        Rect.from_normal(f"{prefix}_wall_x0", "wall", (ox, cy, oz + h / 2), (1, 0, 0), d / 2, h / 2, m["wall"]),
        Rect.from_normal(f"{prefix}_wall_x1", "wall", (ox + w, cy, oz + h / 2), (-1, 0, 0), d / 2, h / 2, m["wall"]),
        Rect.from_normal(f"{prefix}_wall_y0", "wall", (cx, oy, oz + h / 2), (0, 1, 0), w / 2, h / 2, m["wall"]),
        Rect.from_normal(f"{prefix}_wall_y1", "wall", (cx, oy + d, oz + h / 2), (0, -1, 0), w / 2, h / 2, m["wall"]),
    ]
    if ceiling:
        rects.append(Rect.from_normal(f"{prefix}_ceiling", "ceiling", (cx, cy, oz + h), (0, 0, -1),
                                      w / 2, d / 2, m["ceiling"]))
    return rects


def box_surfaces(id, center, size, material="wood", yaw_deg=0.0):
    """Top and four sides of an axis-aligned (optionally yawed) box sitting anywhere.

    center = box centre, size = full extents (sx, sy, sz). Normals point outward.
    """
    c = np.asarray(center, float)
    sx, sy, sz = (float(s) for s in size)
    cy_, sy_ = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    ex = np.array([cy_, sy_, 0.0])
    ey = np.array([-sy_, cy_, 0.0])
    rects = [Rect.from_normal(f"{id}_top", "furniture", c + UP * sz / 2, UP, sx / 2, sy / 2, material,
                              up_hint=ey)]
    for name, n, half_w in (("px", ex, sy / 2), ("nx", -ex, sy / 2), ("py", ey, sx / 2), ("ny", -ey, sx / 2)):
        off = sx / 2 if name.endswith("x") else sy / 2
        rects.append(Rect.from_normal(f"{id}_{name}", "furniture", c + n * off, n, half_w, sz / 2, material))
    return rects
