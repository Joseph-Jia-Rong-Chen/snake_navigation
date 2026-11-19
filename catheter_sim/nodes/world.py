"""World-side nodes: scenario (task), simulated robot, simulated camera, feature maps.

For a physical setup, RobotSimNode and CameraNode are the two nodes to replace
(a hardware driver publishing /robot/joint_state and a frame grabber publishing
/camera/image); everything else stays the same.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import featuremap
from ..bus import Bus, Node
from ..config import Config
from ..env import CatheterEnv
from ..msgs import (Catalog, CameraInfo, Event, FeatureMap, GroundTruth, Image, JointCommand,
                    JointState, NavTarget, StateSnapshot, TaskStatus, VesselInfo)
from ..render import SceneRenderer, fit_camera, surface_to_array
from ..vessel import VesselModel


class TaskNode(Node):
    """Owns the scenario: which vessel model is loaded and which target is active.
    Publishes /sim/world, /nav/target, /task/catalog, /task/status."""

    def __init__(self, bus: Bus, models: list[Path], model_index: int = 0, target: str | int | None = None):
        super().__init__(bus, "task")
        self.models = models
        self.model_index = model_index
        self.vessel: VesselModel | None = None
        self.target_index = 0
        self._initial_target = target
        self.t0 = 0.0
        self.pub_world = self.create_publisher("/sim/world", latched=True)
        self.pub_target = self.create_publisher("/nav/target", latched=True)
        self.pub_catalog = self.create_publisher("/task/catalog", latched=True)
        self.pub_status = self.create_publisher("/task/status")
        self.pub_event = self.create_publisher("/episode/events")
        self.create_subscription("/episode/events", self.on_event)
        self.create_subscription("/sim/ground_truth", self.on_truth)

    def start(self) -> None:
        self._load(self.model_index)
        if self._initial_target is not None:
            names = [t.name for t in self.vessel.targets]
            t = self._initial_target
            self.target_index = names.index(t) if isinstance(t, str) and t in names else int(t) if str(t).isdigit() else 0
            self._publish_target()
        self.pub_event.publish(Event("reset"))

    # --------------------------------------------------------------------
    def _load(self, idx: int) -> None:
        self.model_index = idx % len(self.models)
        path = self.models[self.model_index]
        self.vessel = VesselModel.load(path)
        self.target_index = 0
        self.pub_world.publish(VesselInfo(self.vessel.name, str(path), self.vessel))
        self._publish_target()

    def _publish_target(self) -> None:
        ts = self.vessel.targets
        if ts:
            self.target_index %= len(ts)
            t = ts[self.target_index]
            self.pub_target.publish(NavTarget(t.name, self.target_index, t.position.copy(), t.radius))
        self.pub_catalog.publish(Catalog([m.stem for m in self.models], self.model_index,
                                         [t.name for t in ts], self.target_index))

    @property
    def target(self):
        ts = self.vessel.targets
        return ts[self.target_index % len(ts)] if ts else None

    def on_event(self, ev: Event) -> None:
        if ev.kind in ("next_model", "prev_model"):
            self._load(self.model_index + (1 if ev.kind == "next_model" else -1))
            self.pub_event.publish(Event("reset"))
        elif ev.kind == "next_target" and self.vessel.targets:
            self.target_index += 1
            self._publish_target()
        elif ev.kind == "set_scenario":
            m = ev.data.get("model", self.model_index)
            if m != self.model_index:
                self._load(m)
            self.target_index = ev.data.get("target", 0)
            self._publish_target()
            self.pub_event.publish(Event("reset"))
        elif ev.kind == "reset":
            self.t0 = self.now

    def on_truth(self, gt: GroundTruth) -> None:
        t = self.target
        if t is None:
            return
        d = float(np.linalg.norm(gt.tip_pos - t.position))
        self.pub_status.publish(TaskStatus(t.name, d, d <= t.radius, self.now - self.t0))


class RobotSimNode(Node):
    """Simulated robot. Input: /robot/joint_target. Every tick the target is clamped
    to the joint limits and passed through the actuator (optional delay, lag,
    speed limit), the catheter physics is solved, and the node publishes
    /robot/joint_state and /sim/ground_truth."""

    def __init__(self, bus: Bus, cfg: Config):
        super().__init__(bus, "robot")
        self.cfg = cfg
        self.env: CatheterEnv | None = None
        self.target: tuple[float, float] | None = None
        self.prev = None
        self.pub_state = self.create_publisher("/robot/joint_state")
        self.pub_truth = self.create_publisher("/sim/ground_truth")
        self.pub_snapshot = self.create_publisher("/sim/snapshot")
        self.create_subscription("/sim/world", self.on_world)
        self.create_subscription("/robot/joint_target", self.on_target)
        self.create_subscription("/episode/events", self.on_event)
        self.create_timer(1.0 / cfg.sim.rate_hz, self.update, priority=40)

    def on_world(self, w: VesselInfo) -> None:
        self.env = CatheterEnv(self.cfg, w.vessel, dt=self.bus.dt)
        self._reset()

    def on_target(self, cmd: JointCommand) -> None:
        self.target = (cmd.length, cmd.bend)

    def on_event(self, ev: Event) -> None:
        if ev.kind == "reset" and self.env is not None:
            self._reset()
        elif ev.kind == "snapshot_request" and self.env is not None:
            self.pub_snapshot.publish(StateSnapshot(self.env.get_state()))

    def _reset(self) -> None:
        self.env.reset()
        self.target = None
        self.prev = None
        self._publish(self.env.observe())

    def update(self) -> None:
        if self.env is None:
            return
        if self.target is None:                       # no command yet: hold position
            self.target = (self.env.cath.length, self.env.kappa_to_bend(self.env.cath.kappa))
        self._publish(self.env.command(*self.target))

    def _publish(self, o: dict) -> None:
        jl, jb = o["joint_length"], o["joint_bend"]
        vl = vb = 0.0
        if self.prev is not None:
            vl, vb = (jl - self.prev[0]) / self.bus.dt, (jb - self.prev[1]) / self.bus.dt
        self.prev = (jl, jb)
        self.pub_state.publish(JointState(jl, jb, vl, vb, o["cmd_length"], o["cmd_bend"]))
        c = self.env.cath
        ci = c.last_contact
        self.pub_truth.publish(GroundTruth(
            centerline=c.spline(step=0.5), shape=o["shape"], tip_pos=o["tip_pos"],
            tip_angle=o["tip_angle"], base_angle=o["base_angle"], bend_meas=o["bend_meas"],
            n_contacts=o["n_contacts"], contact_force_total=o["contact_force_total"],
            contact_force_max=o["contact_force_max"], insertion_force=o["insertion_force"],
            contacts=(ci.points[ci.active], ci.force[ci.active], ci.friction[ci.active])))


class CameraNode(Node):
    """Simulated camera: renders the plain scene (vessel + catheter + red tip,
    no UI) at camera.fps and publishes /camera/image (+ latched /camera/info)."""

    def __init__(self, bus: Bus, cfg: Config):
        super().__init__(bus, "camera")
        self.cfg = cfg
        self.renderer: SceneRenderer | None = None
        self.centerline = None
        self.frame = 0
        self.pub_image = self.create_publisher("/camera/image")
        self.pub_info = self.create_publisher("/camera/info", latched=True)
        self.create_subscription("/sim/world", self.on_world)
        self.create_subscription("/sim/ground_truth", self.on_truth)
        self.create_timer(1.0 / cfg.camera.fps, self.capture, priority=50)

    def on_world(self, w: VesselInfo) -> None:
        c = self.cfg.camera
        cam = fit_camera(w.vessel, c.width, c.height, c.margin)
        self.renderer = SceneRenderer(w.vessel, cam)
        self.centerline = None
        self.pub_info.publish(cam)

    def on_truth(self, gt: GroundTruth) -> None:
        self.centerline = gt.centerline

    def capture(self) -> None:
        if self.renderer is None or self.centerline is None:
            return
        surf = self.renderer.render(self.centerline, self.cfg.catheter.diameter, roadmap=True)
        self.pub_image.publish(Image(surface_to_array(surf), self.frame))
        self.frame += 1


class FeatureMapNode(Node):
    """Publishes the U/V target feature maps (latched) at the camera resolution,
    recomputed whenever the vessel, camera or target changes."""

    def __init__(self, bus: Bus):
        super().__init__(bus, "feature_map")
        self.vessel = None
        self.cam: CameraInfo | None = None
        self.target: NavTarget | None = None
        self.map_id = -1
        self.pub = self.create_publisher("/nav/feature_map", latched=True)
        self.create_subscription("/sim/world", self.on_world)
        self.create_subscription("/camera/info", self.on_cam)
        self.create_subscription("/nav/target", self.on_target)

    def on_world(self, w: VesselInfo) -> None:
        # (camera info is checked against the model name, so it need not be cleared)
        self.vessel, self.target = w.vessel, None

    def on_cam(self, cam: CameraInfo) -> None:
        self.cam = cam
        self._update()

    def on_target(self, t: NavTarget) -> None:
        self.target = t
        self._update()

    def _update(self) -> None:
        if self.vessel is None or self.cam is None or self.target is None:
            return
        if self.cam.model != self.vessel.name:
            return
        U, V, mask, c, xa_ya = featuremap.compute(self.vessel, self.cam, self.target.position)
        self.map_id += 1
        self.pub.publish(FeatureMap(U, V, mask, c, xa_ya, self.target.name,
                                    self.target.position.copy(), self.map_id))
