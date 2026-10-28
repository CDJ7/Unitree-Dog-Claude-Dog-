"""From a droplet mask to a measured stain, and from stains to pattern groups.

Two entry points produce the same `Observation`:
  * from_uv_points  - points already in surface coordinates (ideal rendering,
                      tests, decal textures)
  * from_camera_mask - a boolean mask in a camera image + depth + intrinsics +
                      camera pose. The mask pixels are super-sampled and each
                      sample ray is intersected with the surface plane, which
                      undoes the perspective before the ellipse fit (an
                      ellipse measured in raw pixels is distorted unless the
                      camera looks straight at the surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ellipse import EllipseFit, fit_stain, usable
from .origin import MeasuredStain
from .projection import Plane, fit_plane, mask_to_world


@dataclass
class Observation:
    id: str
    surface_id: str
    surface_kind: str         # floor | wall | ceiling | furniture | unknown
    plane: Plane
    fit: EllipseFit
    position: np.ndarray      # world, ellipse centre
    direction: np.ndarray     # world unit travel direction along the surface
    usable: bool
    reason: str
    meta: dict = field(default_factory=dict)

    @property
    def alpha_deg(self):
        return self.fit.impact_angle_deg

    def measured(self):
        return MeasuredStain(self.position, self.plane.normal, self.direction, self.alpha_deg,
                             self.fit.width, self.surface_id, self.id)

    def to_dict(self):
        return {"id": self.id, "surface_id": self.surface_id, "surface_kind": self.surface_kind,
                "position": self.position.tolist(), "direction": self.direction.tolist(),
                "normal": self.plane.normal.tolist(), "usable": self.usable, "reason": self.reason,
                **{k: v for k, v in self.fit.to_dict().items() if k not in ("center", "axis")}, **self.meta}


def surface_kind_from_normal(n):
    n = np.asarray(n, float)
    if n[2] > 0.8:
        return "floor"
    if n[2] < -0.8:
        return "ceiling"
    if abs(n[2]) < 0.3:
        return "wall"
    return "unknown"


def from_uv_points(sid, points_uv, spacing, plane, surface_id="", surface_kind=None, **usable_kw):
    fit = fit_stain(points_uv, spacing)
    ok, why = usable(fit, **usable_kw)
    pos = plane.to_world(fit.center)
    d = plane.dir_to_world(fit.axis)
    kind = surface_kind or surface_kind_from_normal(plane.normal)
    return Observation(sid, surface_id, kind, plane, fit, pos, d / np.linalg.norm(d), ok, why)


def _ray_plane(px, py, intr, T_world_cam, plane):
    rays_c = np.stack([(px - intr.cx) / intr.fx, (py - intr.cy) / intr.fy, np.ones_like(px)], axis=-1)
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
    rays_w = rays_c @ R.T
    t = ((plane.origin - o) @ plane.normal) / (rays_w @ plane.normal)
    return o[None] + t[:, None] * rays_w


def from_camera_mask(sid, mask, depth, intr, T_world_cam, plane=None, surface_id="", surface_kind=None,
                     context_px=6, supersample=4, **usable_kw):
    """Measure a stain seen in a camera image.

    plane: the surface plane if known (e.g. from the map); otherwise it is fitted
    to the depth of the pixels around the stain (mask grown by context_px).
    """
    mask = np.asarray(mask, bool)
    cam_pos = T_world_cam[:3, 3]
    if plane is None:
        ys, xs = np.nonzero(mask)
        y0, y1 = max(ys.min() - context_px, 0), min(ys.max() + context_px + 1, mask.shape[0])
        x0, x1 = max(xs.min() - context_px, 0), min(xs.max() + context_px + 1, mask.shape[1])
        region = np.zeros_like(mask)
        region[y0:y1, x0:x1] = True
        pts = mask_to_world(region, depth, intr, T_world_cam)
        if len(pts) < 10:
            raise ValueError("not enough depth around the stain to fit a plane")
        plane = fit_plane(pts, toward=cam_pos)
    py, px = np.nonzero(mask)
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    ox, oy = np.meshgrid(off, off)
    sx = (px[:, None] + ox.ravel()[None]).ravel()
    sy = (py[:, None] + oy.ravel()[None]).ravel()
    world = _ray_plane(sx, sy, intr, T_world_cam, plane)
    uv = plane.to_local(world)
    # spacing of the samples on the surface: pixel footprint / supersample, worst direction
    dist = np.linalg.norm(world.mean(0) - cam_pos)
    view = (world.mean(0) - cam_pos) / dist
    cos_inc = abs(view @ plane.normal)
    spacing = dist / intr.fx / max(cos_inc, 0.2) / supersample
    obs = from_uv_points(sid, uv, spacing, plane, surface_id, surface_kind, **usable_kw)
    obs.meta.update(pixels=int(mask.sum()), distance=float(dist), incidence_deg=float(np.degrees(np.arccos(cos_inc))))
    return obs


# ------------------------------------------------------------------ grouping

def group_observations(obs, link=0.3, min_size=1, solo_area=1e-3):
    """Single-linkage clusters of stains closer than `link` metres (any surface).

    Observations larger than `solo_area` (m^2, default 10 cm^2) are contact
    stains / smears, not droplets, and each forms its own group.
    """
    solos = [[o] for o in obs if o.fit.area > solo_area]
    obs = [o for o in obs if o.fit.area <= solo_area]
    n = len(obs)
    if n == 0:
        return solos
    P = np.array([o.position for o in obs])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    ii, jj = np.nonzero(np.triu(d < link, 1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(obs[i])
    out = [g for g in groups.values() if len(g) >= min_size]
    return sorted(out, key=len, reverse=True) + solos
