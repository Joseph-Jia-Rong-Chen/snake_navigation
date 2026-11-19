"""Simulation parameters.

All lengths are in millimetres, forces in newtons, moments in N*mm,
bending stiffness (EI) in N*mm^2, curvature in 1/mm.

Defaults live in the dataclasses below. Override them in ``config.json``
(same nested structure) or on the command line, e.g.
``--set catheter.EI_shaft=30 --set catheter.tip_length=20``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class CatheterParams:
    # --- geometry ---------------------------------------------------------
    diameter: float = 1.67          # 5 Fr outer diameter [mm]
    segment_length: float = 1.5     # discretisation of the spline knots [mm]
    initial_length: float = 15.0    # inserted length at reset [mm]
    min_length: float = 3.0         # cannot retract past this [mm]
    max_length: float = 400.0       # cannot insert past this [mm]

    # --- elasticity -------------------------------------------------------
    # Bending stiffness of the shaft. 20 N*mm^2 ~ a 5 Fr Pebax catheter
    # (E ~ 50 MPa, I = pi d^4 / 64 ~ 0.38 mm^4).
    EI_shaft: float = 20.0
    # The steerable tip is usually softer than the shaft.
    EI_tip: float = 6.0
    # Rotational spring at the entry point pulling the base tangent towards
    # the entry direction (models the introducer sheath). 0 = free pivot.
    base_rot_stiffness: float = 10.0   # [N*mm/rad]

    # --- steerable tip (piecewise-constant-curvature arc) -----------------
    # Bending section length and curvature limit. 20 mm with |kappa| <= 0.16
    # (min radius 6.25 mm) gives up to ~180 deg of tip deflection, which is what
    # the sharpest (recurrent / hairpin) take-offs in the example models need;
    # 15 mm / 0.125 (~107 deg) reaches 14 of the 16 example targets.
    tip_length: float = 20.0
    kappa_max: float = 0.16          # [1/mm]

    # --- wall interaction -------------------------------------------------
    friction: float = 0.05           # Coulomb coefficient (hydrophilic coat)
    wall_stiffness: float = 40.0     # contact penalty stiffness [N/mm]
    friction_eps: float = 0.005      # stick/slip regularisation [mm]

    # --- solver ---------------------------------------------------------------
    max_substep: float = 0.3         # max insertion per quasi-static solve [mm]
    newton_iters: int = 30
    newton_tol: float = 1e-5          # gradient tolerance [N*mm]


@dataclass
class ActuatorParams:
    """Joint target -> joint motion (see actuator.py): clamp to joint limits,
    optional transport delay, first-order lag, then a hard speed limit, so even
    a huge jump in the target produces smooth, physically plausible motion."""
    delay: float = 0.0                # transport delay [s] (0 = none)
    time_constant: float = 0.08       # first-order lag [s]
    max_insert_speed: float = 40.0    # joint speed limit [mm/s]
    max_bend_rate: float = 360.0      # joint speed limit [deg/s]


@dataclass
class SimParams:
    rate_hz: int = 60                 # robot / physics update rate (bus tick)


@dataclass
class CameraParams:
    width: int = 900                  # square by default so resizing never distorts
    height: int = 900
    fps: int = 30
    margin: float = 6.0               # mm trimmed off the vessel bounding box


@dataclass
class TeleopParams:
    insert_speed: float = 15.0        # target change [mm/s] while W/S held
    fast_multiplier: float = 3.0      # holding Shift
    bend_rate: float = 170.0          # target change [deg/s] while Left/Right held


@dataclass
class RecordingParams:
    data_dir: str = "data"
    record_video: bool = True         # plain .mp4 of /camera/image
    record_images: bool = False       # also store lossless downsampled frames in the .npz
    image_size: int = 256


@dataclass
class Config:
    catheter: CatheterParams = field(default_factory=CatheterParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    sim: SimParams = field(default_factory=SimParams)
    camera: CameraParams = field(default_factory=CameraParams)
    teleop: TeleopParams = field(default_factory=TeleopParams)
    recording: RecordingParams = field(default_factory=RecordingParams)

    def to_dict(self) -> dict:
        return asdict(self)

    def apply_dict(self, d: dict, strict: bool = True) -> None:
        """strict=False skips unknown keys (e.g. configs stored in old recordings)."""
        for section, values in d.items():
            target = getattr(self, section, None)
            if target is None:
                if strict:
                    raise KeyError(f"unknown config section '{section}'")
                continue
            valid = {f.name: f.type for f in fields(target)}
            for k, v in values.items():
                if k not in valid:
                    if strict:
                        raise KeyError(f"unknown parameter '{section}.{k}'")
                    continue
                cur = getattr(target, k)
                setattr(target, k, type(cur)(v) if not isinstance(cur, bool) else _to_bool(v))

    def apply_overrides(self, overrides: list[str]) -> None:
        for item in overrides or []:
            key, _, val = item.partition("=")
            section, _, name = key.strip().partition(".")
            self.apply_dict({section: {name: _parse(val.strip())}})


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


def _parse(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None and Path(path).exists():
        cfg.apply_dict(json.loads(Path(path).read_text()))
    cfg.apply_overrides(overrides or [])
    return cfg
