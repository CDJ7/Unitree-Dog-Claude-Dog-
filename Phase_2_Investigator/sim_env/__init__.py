"""Simulation side of the Crime Scene Investigator: scenarios, spatter physics, scene packages."""
import os as _os

# Choose MuJoCo's offscreen renderer before anything imports mujoco:
# GLFW when a display exists (WSLg / desktop), else headless EGL.
if not _os.environ.get("MUJOCO_GL"):
    _os.environ["MUJOCO_GL"] = "glfw" if _os.environ.get("DISPLAY") else "egl"

# CPU budget. The laptop also runs other robot software (e.g. a TurtleBot / ROS 2 terminal), so Phase 2 uses
# at most PHASE2_THREADS threads for numpy / OpenCV / PyTorch (default 2). Set it higher on a dedicated machine.
THREADS = max(1, int(_os.environ.get("PHASE2_THREADS", "2")))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, str(THREADS))
