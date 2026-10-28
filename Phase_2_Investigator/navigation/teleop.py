"""Keyboard teleop mapping shared by the MuJoCo viewer, OpenCV and sensor-viewer windows.

W/S forward/back, A/D turn left/right, Q/E strafe left/right, Space stop,
R reset the robot. Most windows only report key presses, not releases, so
each press steps the target speed and the speed stays latched until changed.
"""
from __future__ import annotations

import numpy as np

from navigation.velocity import TELEOP_LIMITS

HELP = "W/S forward/back  A/D turn  Q/E strafe  Space stop  R reset  Esc quit"
STEP = np.array([0.1, 0.05, 0.25])     # per key press: m/s, m/s, rad/s
KEYS = {"w": (0, +1), "s": (0, -1), "q": (1, +1), "e": (1, -1), "a": (2, +1), "d": (2, -1)}


def key_name(key):
    """Normalise a GLFW key code, an OpenCV code or a matplotlib key string."""
    if isinstance(key, str):
        return " " if key == " " else key.lower()
    if key in (32,):
        return " "
    if key == 27 or key == 256:      # Esc (OpenCV, GLFW)
        return "escape"
    if 0 <= key < 256:
        return chr(key).lower()
    return ""


class KeyTeleop:
    def __init__(self, limits=TELEOP_LIMITS):
        self.limits = limits
        self.target = np.zeros(3)
        self.reset_requested = False
        self.quit_requested = False

    def press(self, key):
        k = key_name(key)
        if k in KEYS:
            axis, sign = KEYS[k]
            self.target[axis] += sign * STEP[axis]
            self.target = self.limits.clip(self.target)
            self.target[np.abs(self.target) < 1e-9] = 0.0
        elif k in (" ", "space"):
            self.target[:] = 0
        elif k == "r":
            self.target[:] = 0
            self.reset_requested = True
        elif k == "escape":
            self.quit_requested = True
        else:
            return False
        return True

    def status(self):
        vx, vy, wz = self.target
        return f"cmd vx={vx:+.2f} m/s vy={vy:+.2f} m/s wz={wz:+.2f} rad/s"
