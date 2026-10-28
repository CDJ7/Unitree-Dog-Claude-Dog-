"""Velocity interface (cmd_vel) shared by teleop, exploration and inspection.

Everything that moves the robot goes through `VelocityInterface.send(vx, vy, wz)`
(body frame, ROS Twist linear.x / linear.y / angular.z). The simulator backend
is `SimVelocity`; a real-Go2 backend would wrap Phase 1's
`SportClient.Move(vx, vy, wz)` with the same limits and dead-man.

Sim vs real:
- The sim gait clamps to 0.4 m/s forward, 0.15 m/s sideways, 0.5 rad/s and
  stops 0.35 s after the last command; the real Go2 sport mode accepts more
  but we keep the same project limits on both.
- Commands must be re-sent continuously (at least every `timeout` seconds),
  on both the sim and the real robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedLimits:
    vx: float = 0.3      # m/s forward (backward is limited to half)
    vy: float = 0.15     # m/s sideways
    wz: float = 0.5      # rad/s
    accel: float = 0.6   # m/s^2 and rad/s^2 ramp, 0 = no ramp

    def clip(self, cmd):
        vx, vy, wz = cmd
        return np.array([np.clip(vx, -self.vx / 2, self.vx), np.clip(vy, -self.vy, self.vy),
                         np.clip(wz, -self.wz, self.wz)])


TELEOP_LIMITS = SpeedLimits(vx=0.4, vy=0.15, wz=0.5, accel=0.0)
NAV_LIMITS = SpeedLimits()


class VelocityInterface:
    def __init__(self, limits: SpeedLimits = NAV_LIMITS):
        self.limits = limits
        self.last = np.zeros(3)
        self._t_last = None

    def send(self, vx=0.0, vy=0.0, wz=0.0, t=None):
        """Clip to the limits, ramp by the acceleration limit, and forward to the robot."""
        cmd = self.limits.clip((vx, vy, wz))
        if self.limits.accel > 0 and t is not None and self._t_last is not None:
            step = self.limits.accel * max(0.0, t - self._t_last)
            cmd = self.last + np.clip(cmd - self.last, -step, step)
        self.last, self._t_last = cmd, t
        self._apply(cmd)
        return cmd

    def stop(self):
        self.last = np.zeros(3)
        self._stop()

    def _apply(self, cmd):
        raise NotImplementedError

    def _stop(self):
        self._apply(np.zeros(3))


class SimVelocity(VelocityInterface):
    def __init__(self, session, limits: SpeedLimits = NAV_LIMITS):
        super().__init__(limits)
        self.session = session

    def send(self, vx=0.0, vy=0.0, wz=0.0, t=None):
        return super().send(vx, vy, wz, self.session.time if t is None else t)

    def _apply(self, cmd):
        self.session.cmd_vel(*cmd)

    def _stop(self):
        self.session.stop()
