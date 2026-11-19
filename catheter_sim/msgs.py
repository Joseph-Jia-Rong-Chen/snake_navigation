"""Message types for the bus (the analogue of ROS2 .msg files).

Every message has a `stamp` (sim time [s]) filled in on publish.

Topics:
  /robot/joint_target   JointCommand  target joints (the action), after the mux
  /teleop/joint_target, /policy/joint_target, /autopilot/joint_target
                        JointCommand  candidate targets from each commander
  /robot/joint_state    JointState    where the robot actually is (+ the target it tracks)
  /camera/image         Image         plain frame: vessel + catheter, no UI
  /camera/info          CameraInfo    (latched) image size + world<->pixel transform
  /nav/target           NavTarget     (latched) current navigation target
  /nav/feature_map      FeatureMap    (latched) U, V maps for the current target
  /sim/world            VesselInfo    (latched) the loaded vessel model
  /sim/ground_truth     GroundTruth   sim-only labels (tip, shape, forces): never a policy input
  /sim/snapshot         StateSnapshot full sim state (for exact re-simulation of recordings)
  /task/status          TaskStatus    distance to target, success
  /task/catalog         Catalog       (latched) available models / targets
  /mux/active           ActiveSource  (latched) which commander drives the robot
  /ui/keys              KeyState      keyboard state from the viewer window
  /recorder/status      RecorderStatus
  /episode/events       Event         reset, record_*, next_model, next_target, ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Msg:
    stamp: float | None = field(default=None, kw_only=True)


@dataclass
class JointCommand(Msg):
    length: float                     # target insertion length [mm]
    bend: float                       # target bend angle [deg], + = clockwise
    source: str = ""                  # teleop | policy | autopilot | takeover
    engaged: bool = False             # teleop: an operator key is pressed this tick
    obs_stamp: float | None = None    # stamp of the observation this was computed from


@dataclass
class JointState(Msg):
    length: float                     # actual insertion length [mm]
    bend: float                       # actual bend angle [deg]
    length_vel: float = 0.0           # [mm/s]
    bend_vel: float = 0.0             # [deg/s]
    target_length: float = 0.0        # the target the robot is tracking this tick
    target_bend: float = 0.0


@dataclass
class Image(Msg):
    data: np.ndarray                  # (H, W, 3) uint8 RGB
    frame: int = 0


@dataclass
class CameraInfo(Msg):
    """Pixel coordinates are continuous: pixel (i, j) covers [i, i+1) x [j, j+1),
    i = column (x, + right), j = row (y, + down). World is mm with y up."""
    width: int
    height: int
    scale: float                      # pixels per mm
    center_world: np.ndarray          # world point mapped to center_px
    center_px: np.ndarray
    model: str = ""

    def world_to_px(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        d = (p - self.center_world) * self.scale
        return np.stack([self.center_px[0] + d[:, 0], self.center_px[1] - d[:, 1]], 1)

    def px_to_world(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.center_world[0] + (u - self.center_px[0]) / self.scale,
                self.center_world[1] - (v - self.center_px[1]) / self.scale)

    def world_to_index(self, p: np.ndarray) -> np.ndarray:
        """Pixel *index* coordinates (pixel centres at integers), as used by the feature map."""
        return self.world_to_px(p) - 0.5

    def scaled(self, width: int, height: int) -> "CameraInfo":
        """Same view at another resolution (uniform scale; keep the aspect ratio)."""
        s = min(width / self.width, height / self.height)
        return CameraInfo(width=width, height=height, scale=self.scale * s,
                          center_world=self.center_world.copy(),
                          center_px=np.array([width / 2, height / 2]), model=self.model, stamp=self.stamp)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "scale": self.scale,
                "center_world": self.center_world.tolist(), "center_px": self.center_px.tolist(),
                "model": self.model}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraInfo":
        return cls(width=d["width"], height=d["height"], scale=d["scale"],
                   center_world=np.array(d["center_world"]), center_px=np.array(d["center_px"]),
                   model=d.get("model", ""))


@dataclass
class VesselInfo(Msg):
    name: str
    path: str
    vessel: Any                       # catheter_sim.vessel.VesselModel (in-process)


@dataclass
class NavTarget(Msg):
    name: str
    index: int
    position: np.ndarray              # world [mm]
    radius: float


@dataclass
class FeatureMap(Msg):
    """U = (i - x_a)/c * m + (1 - m),  V = (j - y_a)/c * m + (1 - m)."""
    U: np.ndarray                     # (H, W) float32
    V: np.ndarray                     # (H, W) float32
    mask: np.ndarray                  # (H, W) uint8, 1 inside the vessel
    c: float
    target_index: np.ndarray          # (x_a, y_a) in pixel-index coordinates
    target_name: str
    target_world: np.ndarray
    map_id: int = 0


@dataclass
class GroundTruth(Msg):
    centerline: np.ndarray            # dense spline (M,2) [mm] for rendering
    shape: np.ndarray                 # (200,2) spline resampled base->tip [mm]
    tip_pos: np.ndarray
    tip_angle: float                  # [rad], y up, CCW +
    base_angle: float
    bend_meas: float                  # bend measured from the deformed tip [deg]
    n_contacts: int
    contact_force_total: float
    contact_force_max: float
    insertion_force: float
    contacts: tuple | None = None     # (points, normal forces, friction forces)


@dataclass
class StateSnapshot(Msg):
    state: dict


@dataclass
class TaskStatus(Msg):
    target_name: str
    target_dist: float
    success: bool
    episode_time: float


@dataclass
class Catalog(Msg):
    models: list
    model_index: int
    targets: list
    target_index: int


@dataclass
class ActiveSource(Msg):
    active: str
    primary: str


@dataclass
class KeyState(Msg):
    insert: bool = False              # W
    retract: bool = False             # S
    left: bool = False                # bend CCW
    right: bool = False               # bend CW
    fast: bool = False                # Shift
    straighten: bool = False          # Space (edge: true on the tick it was pressed)

    @property
    def motion(self) -> bool:
        return self.insert or self.retract or self.left or self.right or self.straighten

    def as_array(self) -> np.ndarray:
        return np.array([self.insert, self.retract, self.left, self.right], bool)


@dataclass
class RecorderStatus(Msg):
    recording: bool
    frames: int = 0
    path: str = ""
    message: str = ""


@dataclass
class Event(Msg):
    kind: str                         # reset | record_start | record_stop | record_toggle |
                                      # next_model | prev_model | next_target | set_scenario |
                                      # resume | takeover | snapshot_request | shutdown
    data: dict = field(default_factory=dict)
