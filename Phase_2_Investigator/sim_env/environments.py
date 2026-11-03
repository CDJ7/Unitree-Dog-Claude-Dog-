"""Environment catalogue: the simulator's environments plus our packages.

The simulator's NativeSceneBuilder reads `<root>/assets/environments/manifest.json`
and `<root>/assets/robots`. We give it an overlay root in sim_env/build/overlay
(gitignored) made of symlinks to the simulator's assets and to our packages in
sim_env/scene_packages, with a merged manifest, so the simulator's tree is never
modified (docs/scenes.md package format, manifest schema 2).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = HERE / "scene_packages"
OVERLAY = HERE / "build" / "overlay"

# Our packages: id -> label
OUR_ENVIRONMENTS = {"csi_simple_room": "CSI simple room (6 x 5 m)"}


def _link(target, link):
    link = Path(link)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == Path(target):
            return
        link.unlink()
    link.symlink_to(target, target_is_directory=Path(target).is_dir())


def overlay_root(sim_root):
    """Create/refresh the overlay and return its path."""
    sim_root = Path(sim_root).resolve()
    envs = OVERLAY / "assets" / "environments"
    envs.mkdir(parents=True, exist_ok=True)
    _link(sim_root / "assets" / "robots", OVERLAY / "assets" / "robots")
    manifest = json.loads((sim_root / "assets/environments/manifest.json").read_text())
    for item in manifest["environments"]:
        env_dir = (sim_root / item["xmlPath"]).parent
        _link(env_dir, envs / env_dir.name)
    known = {e["id"] for e in manifest["environments"]}
    for env_id, label in OUR_ENVIRONMENTS.items():
        _link(PACKAGES / env_id, envs / env_id)
        if env_id not in known:
            manifest["environments"].append({
                "id": env_id, "label": label, "visualMode": "mesh",
                "xmlPath": f"./assets/environments/{env_id}/scene.xml",
                "filesPath": f"./assets/environments/{env_id}/index.json",
                "spawnPath": f"./assets/environments/{env_id}/spawn.json",
                "camera": None})
    text = json.dumps(manifest, indent=2)
    path = envs / "manifest.json"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return OVERLAY
