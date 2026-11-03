"""Paths shared by all Test 2 scripts, and access to the Phase 2 crime-scene simulator.

Test 2 does not copy the Phase 2 simulator: it imports it from ../Phase_2_Investigator
(scenarios, SimSession, planner). Importing this module puts that folder on sys.path
and picks an offscreen OpenGL backend before MuJoCo is imported.

Rendering backend: MUJOCO_GL is respected if set. Otherwise EGL is tried (GPU / Colab),
falling back to OSMesa (pure CPU software rendering, what the cloud container uses).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TEST2 = Path(__file__).resolve().parents[1]
REPO = TEST2.parent
PHASE2 = REPO / "Phase_2_Investigator"
CONFIG = TEST2 / "config"
DATA = TEST2 / "data"
RESULTS = TEST2 / "results"


def _pick_gl():
    if os.environ.get("MUJOCO_GL"):
        if os.environ["MUJOCO_GL"] == "osmesa":
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        return
    if os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "glfw"
        return
    try:   # probe EGL in a subprocess: a failed EGL init can't be undone in-process
        import subprocess
        ok = subprocess.run([sys.executable, "-c", "import os;os.environ['MUJOCO_GL']='egl';import mujoco;"
                             "mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'),8,8)"],
                            capture_output=True, timeout=60).returncode == 0
    except Exception:
        ok = False
    os.environ["MUJOCO_GL"] = "egl" if ok else "osmesa"
    if not ok:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"


_pick_gl()
for p in (str(PHASE2), str(TEST2)):
    if p not in sys.path:
        sys.path.insert(0, p)
