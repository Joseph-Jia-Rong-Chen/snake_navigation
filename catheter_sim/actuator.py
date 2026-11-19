"""Actuator model between the operator's command and the robot joints.

The operator sets a *target* (insertion length [mm], bend angle [deg]). A real
robot does not reach it instantly, so each joint is modelled as

    target --[transport delay]--> first-order lag (time constant) --> rate limit --> joint

The joint values are what drive the catheter physics. Set delay=0 and
time_constant=0 for an ideal (instant) actuator.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import ActuatorParams


class Actuator:
    def __init__(self, params: ActuatorParams, dt: float):
        self.p = params
        self.dt = dt
        self.n_delay = max(int(round(params.delay / dt)), 0)
        self.reset(0.0, 0.0)

    def reset(self, length: float, bend: float) -> None:
        self.joint = np.array([length, bend], float)
        self.buf: deque = deque([self.joint.copy()] * self.n_delay, maxlen=self.n_delay + 1)

    def update(self, target_length: float, target_bend: float) -> np.ndarray:
        """Feed this frame's target; returns the new joint values (length, bend)."""
        self.buf.append(np.array([target_length, target_bend], float))
        delayed = self.buf.popleft() if len(self.buf) > self.n_delay else self.buf[0]
        tau = self.p.time_constant
        alpha = 1.0 if tau <= 0 else 1.0 - np.exp(-self.dt / tau)
        step = alpha * (delayed - self.joint)
        vmax = np.array([self.p.max_insert_speed, self.p.max_bend_rate]) * self.dt
        self.joint = self.joint + np.clip(step, -vmax, vmax)
        return self.joint.copy()

    def get_state(self) -> dict:
        return {"act_joint": self.joint.copy(),
                "act_buf": np.array(list(self.buf), float).reshape(-1, 2)}

    def set_state(self, st: dict) -> None:
        self.joint = np.array(st["act_joint"], float)
        self.buf = deque([b for b in np.asarray(st["act_buf"], float)], maxlen=self.n_delay + 1)
