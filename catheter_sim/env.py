"""Gym-style wrapper: the same simulation the teleop app drives, usable from
scripts (scripted data collection, policy evaluation, RL)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .catheter import Catheter
from .config import Config
from .vessel import VesselModel

N_SHAPE = 200          # fixed-size shape descriptor: samples along the catheter


class CatheterEnv:
    """Actions are increments per step: (d_length [mm], d_kappa [1/mm])."""

    def __init__(self, cfg: Config, model_path: str | Path):
        self.cfg = cfg
        self.load_model(model_path)

    def load_model(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.vessel = VesselModel.load(model_path)
        self.cath = Catheter(self.cfg.catheter, self.vessel)
        self.target_idx = 0
        self.t = 0.0

    @property
    def target(self):
        ts = self.vessel.targets
        return ts[self.target_idx % len(ts)] if ts else None

    def reset(self) -> dict:
        self.cath.reset()
        self.t = 0.0
        return self.observe()

    def step(self, d_length: float, d_kappa: float, dt: float = 0.0) -> dict:
        self.cath.step(d_length, d_kappa)
        self.t += dt
        return self.observe()

    def observe(self) -> dict:
        c = self.cath
        tip, ang = c.tip_pose()
        ci = c.last_contact
        f = ci.force[ci.active]
        tgt = self.target
        dist = float(np.linalg.norm(tip - tgt.position)) if tgt else np.nan
        return {
            "t": self.t,
            "length": c.length,
            "kappa_cmd": c.kappa,
            "kappa_meas": c.measured_tip_curvature(),
            "tip_pos": tip,
            "tip_angle": ang,
            "base_angle": float(c.theta[0]),
            "shape": self.shape_descriptor(),
            "knots": c.knots(),
            "n_contacts": int(ci.active.sum()),
            "contact_force_total": float(np.linalg.norm(f, axis=1).sum()) if len(f) else 0.0,
            "contact_force_max": float(np.linalg.norm(f, axis=1).max()) if len(f) else 0.0,
            "insertion_force": c.insertion_force,
            "target_dist": dist,
            "success": bool(tgt is not None and dist <= tgt.radius),
        }

    def shape_descriptor(self, n: int = N_SHAPE) -> np.ndarray:
        """Catheter spline resampled to n points uniformly in arc length (base->tip)."""
        sp = self.cath.spline(step=0.25)
        seg = np.linalg.norm(np.diff(sp, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.linspace(0, s[-1], n)
        return np.stack([np.interp(u, s, sp[:, 0]), np.interp(u, s, sp[:, 1])], 1).astype(np.float32)
