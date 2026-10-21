"""Droplet launch generators for each bloodstain mechanism.

Every function takes a parameter dict (defaults below, overridable from the
scenario YAML) and a numpy Generator, and returns (launches, meta), where
meta holds the ground truth the analysis is scored against (origin, swing
path, jet path...). Angles are in degrees, lengths in metres, speeds in m/s.

The numbers are plausible textbook ranges, chosen so each mechanism leaves a
recognisably different pattern; they are not calibrated against lab data.
"""
from __future__ import annotations

import numpy as np

from .physics import Launch

UP = np.array([0.0, 0.0, 1.0])

DEFAULTS = {
    # medium-velocity impact spatter from a blunt-force blow
    "impact": {"n": 140, "speed": [1.5, 7.0], "cone_deg": 65.0, "axis": [1.0, 0.0, 0.15],
               "d_median_mm": 1.3, "d_sigma": 0.5, "d_range_mm": [0.4, 4.0], "jitter_m": 0.01},
    # cast-off from a swung object (overhead back-swing by default)
    "castoff": {"n": 40, "radius": 0.95, "omega": 10.0, "heading_deg": 0.0,
                "phi_start_deg": -40.0, "phi_end_deg": 200.0, "release_deg": [25.0, 150.0],
                "speed_frac": [0.5, 0.9], "cone_deg": 4.0,
                "d_median_mm": 1.6, "d_sigma": 0.35, "d_range_mm": [0.6, 3.5]},
    # projected / arterial-style pulsing jet
    "projected": {"duration": 2.5, "bpm": 80, "per_pulse": 30, "speed": [2.0, 3.8],
                  "azimuth_deg": 0.0, "elevation_deg": 20.0, "sweep_deg": 30.0, "elev_sweep_deg": 10.0,
                  "cone_deg": 5.0, "source_vel": [0.0, 0.0, -0.15],
                  "d_median_mm": 2.6, "d_sigma": 0.3, "d_range_mm": [1.0, 5.0]},
    # passive drip from a moving or stationary source
    "drip": {"n": 18, "end": None, "walk_speed": 0.9, "d_range_mm": [4.0, 5.0]},
    # transfer / smear: no flight, handled by the generator on a surface
    "transfer": {"width": 0.06, "feather_frac": 0.3},
}


def merged(mech, params):
    p = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS[mech].items()}
    p.update(params or {})
    return p


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _diameters(p, n, rng):
    lo, hi = (x / 1000.0 for x in p["d_range_mm"])
    if "d_median_mm" in p:
        d = np.exp(rng.normal(np.log(p["d_median_mm"] / 1000.0), p["d_sigma"], n))
    else:
        d = rng.uniform(lo, hi, n)
    return np.clip(d, lo, hi)


def _cone(axis, half_angle_deg, n, rng):
    """n unit vectors uniform in a cone around axis."""
    axis = _unit(axis)
    cos_max = np.cos(np.radians(half_angle_deg))
    z = rng.uniform(cos_max, 1.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - z ** 2)
    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(axis, a))
    e2 = np.cross(axis, e1)
    return (z[:, None] * axis + (r * np.cos(phi))[:, None] * e1 + (r * np.sin(phi))[:, None] * e2)


def impact(origin, params, rng):
    p = merged("impact", params)
    o = np.asarray(origin, float)
    dirs = _cone(p["axis"], p["cone_deg"], p["n"], rng)
    speeds = rng.uniform(*p["speed"], p["n"])
    d = _diameters(p, p["n"], rng)
    jit = rng.normal(0, p["jitter_m"], (p["n"], 3))
    launches = [Launch(o + jit[i], dirs[i] * speeds[i], float(d[i]), 0.0) for i in range(p["n"])]
    meta = {"origin": o.tolist(), "params": p}
    return launches, meta


def swing_point(pivot, heading_deg, radius, phi_deg):
    """Point on the swing arc. phi = 0 is straight ahead (horizontal), 90 overhead, 180 behind."""
    h = np.radians(heading_deg)
    f = np.array([np.cos(h), np.sin(h), 0.0])
    ph = np.radians(phi_deg)
    return np.asarray(pivot, float) + radius * (np.cos(ph) * f + np.sin(ph) * UP)


def castoff(origin, params, rng):
    """origin = the shoulder (pivot) of the person swinging."""
    p = merged("castoff", params)
    pivot = np.asarray(origin, float)
    h = np.radians(p["heading_deg"])
    f = np.array([np.cos(h), np.sin(h), 0.0])
    sign = 1.0 if p["phi_end_deg"] >= p["phi_start_deg"] else -1.0
    phis = rng.uniform(*sorted(p["release_deg"]), p["n"])
    launches = []
    tip_speed = p["omega"] * p["radius"]
    d = _diameters(p, p["n"], rng)
    for i, phi in enumerate(np.sort(phis)):
        ph = np.radians(phi)
        pos = swing_point(pivot, p["heading_deg"], p["radius"], phi)
        tangent = sign * (-np.sin(ph) * f + np.cos(ph) * UP)
        v_dir = _cone(tangent, p["cone_deg"], 1, rng)[0]
        speed = tip_speed * rng.uniform(*p["speed_frac"])
        t_rel = abs(np.radians(phi - p["phi_start_deg"])) / p["omega"]
        launches.append(Launch(pos, v_dir * speed, float(d[i]), float(t_rel)))
    path_phi = np.linspace(p["phi_start_deg"], p["phi_end_deg"], 49)
    swing = [swing_point(pivot, p["heading_deg"], p["radius"], a).tolist() for a in path_phi]
    meta = {"origin": pivot.tolist(), "params": p, "swing_path": swing,
            "swing_times": (np.abs(np.radians(path_phi - p["phi_start_deg"])) / p["omega"]).tolist(),
            "tip_speed": tip_speed}
    return launches, meta


def projected(origin, params, rng):
    """Pulsing jet from a moving source (arterial-style)."""
    p = merged("projected", params)
    o = np.asarray(origin, float)
    n_pulses = max(1, int(round(p["duration"] * p["bpm"] / 60.0)))
    period = 60.0 / p["bpm"]
    launches = []
    jet_path = []
    for k in range(n_pulses):
        s = k / max(n_pulses - 1, 1)
        az = np.radians(p["azimuth_deg"] + p["sweep_deg"] * (s - 0.5))
        el = np.radians(p["elevation_deg"] + p["elev_sweep_deg"] * np.sin(2 * np.pi * s))
        axis = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        t0 = k * period
        src = o + np.asarray(p["source_vel"]) * t0
        jet_path.append({"t": t0, "source": src.tolist(), "axis": axis.tolist()})
        m = p["per_pulse"]
        dirs = _cone(axis, p["cone_deg"], m, rng)
        # each spurt is strongest at its start and fades
        prof = rng.uniform(0, 1, m)
        speeds = p["speed"][0] + (p["speed"][1] - p["speed"][0]) * (1 - prof) ** 0.7
        d = _diameters(p, m, rng)
        for i in range(m):
            launches.append(Launch(src + rng.normal(0, 0.004, 3), dirs[i] * speeds[i], float(d[i]),
                                   float(t0 + 0.25 * period * prof[i])))
    meta = {"origin": o.tolist(), "params": p, "jet_path": jet_path}
    return launches, meta


def drip(origin, params, rng):
    """Drops falling from a source at `origin`, walking toward params['end'] if given."""
    p = merged("drip", params)
    start = np.asarray(origin, float)
    end = np.asarray(p["end"], float) if p.get("end") is not None else start.copy()
    seg = end - start
    dist = np.linalg.norm(seg[:2])
    walking = dist > 0.05
    vel = (_unit(np.r_[seg[:2], 0.0]) * p["walk_speed"]) if walking else np.zeros(3)
    n = p["n"]
    s = np.sort(rng.uniform(0, 1, n)) if walking else np.zeros(n)
    duration = dist / p["walk_speed"] if walking else 0.4 * n
    d = _diameters(p, n, rng)
    launches = []
    for i in range(n):
        pos = start + s[i] * seg + np.r_[rng.normal(0, 0.01, 2), 0.0]
        v = vel + np.r_[rng.normal(0, 0.03, 2), 0.0]
        t = s[i] * duration if walking else i * 0.4
        launches.append(Launch(pos, v, float(d[i]), float(t)))
    meta = {"origin": start.tolist(), "end": end.tolist(), "walking": bool(walking), "params": p,
            "path": [start.tolist(), end.tolist()]}
    return launches, meta


LAUNCHERS = {"impact": impact, "castoff": castoff, "projected": projected, "drip": drip}
