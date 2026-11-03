"""Per-run domain randomisation for the Q2 dataset: lighting and camera noise.

Lighting is applied to the MuJoCo model (headlight scales, colour tint, three dimmable ceiling
lamps that can be shifted). Camera noise is applied to each rendered frame before it is saved
(brightness/contrast/gamma, Gaussian sensor noise, optional blur, JPEG quality). All parameters are
drawn from one seeded generator and saved in the run's metadata.json.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def sample_lighting(rng, cfg):
    tints = cfg["tints"]
    return {
        "headlight_ambient": float(rng.uniform(*cfg["headlight_ambient"])),
        "headlight_diffuse": float(rng.uniform(*cfg["headlight_diffuse"])),
        "tint": [float(v) for v in tints[rng.integers(len(tints))]],
        "lamps": [float(rng.uniform(*cfg["lamp"])) for _ in range(3)],
        "lamp_shift": [float(v) for v in rng.uniform(-cfg["lamp_shift"], cfg["lamp_shift"], 2)],
    }


def sample_camera_noise(rng, cfg):
    return {
        "brightness": float(rng.uniform(*cfg["brightness"])),
        "contrast": float(rng.uniform(*cfg["contrast"])),
        "gamma": float(rng.uniform(*cfg["gamma"])),
        "gauss_sigma": float(rng.uniform(*cfg["gauss_sigma"])),
        "blur_radius": float(rng.uniform(*cfg["blur_radius"])) if rng.random() < cfg["blur_prob"] else 0.0,
        "jpeg_quality": int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1)),
    }


def apply_camera_noise(rgb, p, rng):
    """uint8 HxWx3 -> JPEG bytes with the run's camera model applied (fresh noise per frame)."""
    x = rgb.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * p["contrast"] + 0.5 + (p["brightness"] - 1.0) * 0.5, 0, 1)
    x = x ** p["gamma"]
    x = x * 255.0 + rng.normal(0.0, p["gauss_sigma"], x.shape)
    im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if p["blur_radius"] > 0:
        im = im.filter(ImageFilter.GaussianBlur(p["blur_radius"]))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=p["jpeg_quality"])
    return buf.getvalue()
