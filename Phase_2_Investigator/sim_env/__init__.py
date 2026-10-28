"""Simulation side of the Crime Scene Investigator: scenarios, spatter physics, scene packages."""
import os as _os

# Choose MuJoCo's offscreen renderer before anything imports mujoco:
# GLFW when a display exists (WSLg / desktop), else headless EGL.
if not _os.environ.get("MUJOCO_GL"):
    _os.environ["MUJOCO_GL"] = "glfw" if _os.environ.get("DISPLAY") else "egl"
