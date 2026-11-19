"""Scripted demonstrator: follows the vessel centreline to a target using the
same discrete key actions a human has (W/S, Left/Right), so its episodes look
like teleoperation data. Useful to sanity-check that a model/parameter set is
navigable and to bootstrap datasets.
"""
from __future__ import annotations

import numpy as np

from .config import Config


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Autopilot:
    def __init__(self, cfg: Config, path: np.ndarray, lookahead: float = 6.0, gain: float = 0.35):
        self.cfg = cfg
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.arange(0, s[-1], 0.5)
        self.path = np.stack([np.interp(u, s, path[:, 0]), np.interp(u, s, path[:, 1])], 1)
        self.s = u
        self.idx = 0
        self.lookahead = lookahead
        self.gain = gain
        self.retract_left = 0.0
        self.hist: list[float] = []

    def act(self, obs: dict, dt: float):
        """Returns keys [W, S, Left, Right]."""
        cp = self.cfg.catheter
        tip, ang = obs["tip_pos"], obs["tip_angle"]
        lo, hi = max(self.idx - 20, 0), min(self.idx + 40, len(self.path))
        d = np.linalg.norm(self.path[lo:hi] - tip, axis=1)
        self.idx = lo + int(np.argmin(d))
        off = float(d.min())
        self.hist.append(self.s[self.idx])

        la = min(np.searchsorted(self.s, self.s[self.idx] + self.lookahead), len(self.path) - 1)
        want = self.path[la] - tip
        err = _wrap(np.arctan2(want[1], want[0]) - ang)       # + = target is CCW of tip
        k_des = float(np.clip(-self.gain * err, -cp.kappa_max, cp.kappa_max))
        # steer the commanded bend angle (the operator's input) towards the desired one
        bend_des = np.degrees(k_des * cp.tip_length)
        tol = 4.0                                                # deg
        left = bool(obs["cmd_bend"] > bend_des + tol)
        right = bool(obs["cmd_bend"] < bend_des - tol)

        # stuck / wrong-branch detection -> back off and retry
        win = int(2.0 / dt)
        if self.retract_left <= 0 and len(self.hist) > win:
            progress = self.hist[-1] - self.hist[-win]
            if progress < 1.5 or off > 6.0:
                self.retract_left = 12.0
                self.hist.clear()
        if self.retract_left > 0:
            self.retract_left -= self.cfg.teleop.insert_speed * dt
            return [False, True, left, right]
        insert = bool(abs(err) < np.radians(70))
        return [insert, False, left, right]
