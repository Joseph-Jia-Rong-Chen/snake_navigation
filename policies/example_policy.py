"""Example policies showing the Policy API (see catheter_sim/policy_api.py).

Run one:
  python launch.py --commander policy --policy policies/example_policy.py:FeatureMapPolicy
"""
from __future__ import annotations

import numpy as np

from catheter_sim.policy_api import Policy, find_tip_pixel


class CreepPolicy(Policy):
    """Minimal template: insert slowly with a straight tip."""
    rate_hz = 10

    def __init__(self, speed: float = 8.0):
        self.speed = speed

    def act(self, obs):
        length, _ = obs["last_action"]
        return length + self.speed / self.rate_hz, 0.0


class FeatureMapPolicy(Policy):
    """Hand-written baseline that uses only real observations: it finds the red
    tip fiducial in the image, reads the feature maps there, and steers the tip
    towards the target. Navigates simple geometry; it is a demonstration of the
    inputs, not a good controller (it does not plan through bifurcations)."""
    rate_hz = 10
    input_size = (320, 180)

    def __init__(self, speed: float = 10.0, gain: float = 120.0):
        self.speed, self.gain = speed, gain
        self.prev_tip = None

    def reset(self, info):
        self.prev_tip = None

    def act(self, obs):
        length, bend = obs["last_action"]
        tip = find_tip_pixel(obs["image"])
        if tip is None:
            return length, bend
        i, j = np.clip(tip.astype(int), 0, np.array(obs["image"].shape[1::-1]) - 1)
        u, v = obs["feature_u"][j, i], obs["feature_v"][j, i]
        to_target = -np.array([u, v])                    # image coords, y down
        if self.prev_tip is not None and np.linalg.norm(tip - self.prev_tip) > 0.5:
            heading = tip - self.prev_tip
            # signed angle from heading to target direction; image y is down, so
            # positive cross = clockwise on screen = positive bend
            cross = heading[0] * to_target[1] - heading[1] * to_target[0]
            ang = np.arctan2(cross, heading @ to_target)
            bend = float(np.clip(np.degrees(ang) * self.gain / 90.0, -150, 150))
        self.prev_tip = tip
        return length + self.speed / self.rate_hz, bend


class ReplayPolicy(Policy):
    """Open-loop replay of the aligned actions of a raw episode (pipeline test)."""

    def __init__(self, episode: str):
        from catheter_sim.dataset import Episode
        ep = Episode(episode)
        self.rate_hz = ep.fps
        self.actions = ep.a["action"]
        self.k = 0

    def reset(self, info):
        self.k = 0

    def act(self, obs):
        a = self.actions[min(self.k, len(self.actions) - 1)]
        self.k += 1
        return float(a[0]), float(a[1])
