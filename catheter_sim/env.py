"""Gym-style wrapper: the same simulation the teleop app drives, usable from
scripts (scripted data collection, policy evaluation, RL).

Joint space of the robot (2 DoF):
  length  insertion length [mm], + = deeper
  bend    tip bend angle [deg] = kappa * tip_length, + = clockwise, - = counter-clockwise

`command()` sets *targets*; they reach the joints through the actuator model
(delay + lag + rate limit, see actuator.py), and the joints drive the physics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .actuator import Actuator
from .catheter import Catheter
from .config import Config
from .vessel import VesselModel

N_SHAPE = 200          # fixed-size shape descriptor: samples along the catheter


class CatheterEnv:
    def __init__(self, cfg: Config, model: str | Path | VesselModel, dt: float | None = None):
        self.cfg = cfg
        self.dt = dt if dt is not None else 1.0 / cfg.sim.rate_hz
        self.load_model(model)

    def load_model(self, model: str | Path | VesselModel) -> None:
        if isinstance(model, VesselModel):
            self.vessel = model
            self.model_path = Path(model.path) if model.path else None
        else:
            self.model_path = Path(model)
            self.vessel = VesselModel.load(model)
        self.cath = Catheter(self.cfg.catheter, self.vessel)
        self.actuator = Actuator(self.cfg.actuator, self.dt)
        self.target_idx = 0
        self.reset()

    @property
    def target(self):
        ts = self.vessel.targets
        return ts[self.target_idx % len(ts)] if ts else None

    @property
    def bend_max(self) -> float:
        p = self.cfg.catheter
        return float(np.degrees(p.kappa_max * p.tip_length))

    def kappa_to_bend(self, kappa: float) -> float:
        return float(np.degrees(kappa * self.cfg.catheter.tip_length))

    def bend_to_kappa(self, bend: float) -> float:
        return float(np.radians(bend) / self.cfg.catheter.tip_length)

    # ----------------------------------------------------------------- control
    def reset(self) -> dict:
        self.cath.reset()
        self.t = 0.0
        self.cmd = np.array([self.cath.length, 0.0])
        self.actuator.reset(*self.cmd)
        return self.observe()

    def command(self, target_length: float, target_bend: float) -> dict:
        """Advance one frame (dt) towards absolute targets (mm, deg)."""
        p = self.cfg.catheter
        self.cmd = np.array([np.clip(target_length, p.min_length, p.max_length),
                             np.clip(target_bend, -self.bend_max, self.bend_max)])
        jl, jb = self.actuator.update(*self.cmd)
        self.cath.set_state(jl, self.bend_to_kappa(jb))
        self.t += self.dt
        return self.observe()

    def step(self, d_length: float, d_kappa: float) -> dict:
        """Direct joint increments, bypassing the actuator model."""
        self.cath.step(d_length, d_kappa)
        self.cmd = np.array([self.cath.length, self.kappa_to_bend(self.cath.kappa)])
        self.actuator.reset(*self.cmd)
        self.t += self.dt
        return self.observe()

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        """Everything needed to re-run deterministically from the commands."""
        return {**self.cath.get_state(), **self.actuator.get_state(),
                "cmd": self.cmd.copy(), "t": np.float64(self.t)}

    def set_state(self, st: dict) -> None:
        self.cath.set_full_state(st)
        if "act_joint" in st:
            self.actuator.set_state(st)
        if "cmd" in st:
            self.cmd = np.array(st["cmd"], float)
        self.t = float(st.get("t", 0.0))

    def observe(self) -> dict:
        c = self.cath
        tip, ang = c.tip_pose()
        ci = c.last_contact
        f = ci.force[ci.active]
        tgt = self.target
        dist = float(np.linalg.norm(tip - tgt.position)) if tgt else np.nan
        kappa_meas = c.measured_tip_curvature()
        return {
            "t": self.t,
            "cmd_length": float(self.cmd[0]),
            "cmd_bend": float(self.cmd[1]),
            "joint_length": c.length,
            "joint_bend": self.kappa_to_bend(c.kappa),
            "bend_meas": self.kappa_to_bend(kappa_meas),
            "kappa_joint": c.kappa,
            "kappa_meas": kappa_meas,
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
