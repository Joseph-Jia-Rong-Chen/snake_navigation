"""Commanders (things that produce joint targets) and the command mux.

  TeleopNode     /ui/keys            -> /teleop/joint_target
  AutopilotNode  privileged sim data -> /autopilot/joint_target   (scripted expert)
  PolicyNode     image, feature map, joint state -> /policy/joint_target
  MuxNode        picks one source    -> /robot/joint_target

With --commander policy (or autopilot), pressing any motion key hands control to
the operator (takeover); G gives it back.
"""
from __future__ import annotations

import concurrent.futures as cf

import numpy as np

from ..autopilot import Autopilot
from ..bus import Bus, Node
from ..config import Config
from ..featuremap import compute as compute_feature_maps
from ..msgs import (ActiveSource, CameraInfo, Event, FeatureMap, GroundTruth, Image, JointCommand,
                    JointState, KeyState, NavTarget, VesselInfo)
from ..policy_api import Policy
from ..render import resize_image


class KeyIntegrator:
    """Teleop mapping: held keys move the target at a fixed rate (W/S length,
    Left/Right bend, Shift faster, Space straightens)."""

    def __init__(self, cfg: Config, dt: float):
        self.cfg, self.dt = cfg, dt
        p = cfg.catheter
        self.bend_max = float(np.degrees(p.kappa_max * p.tip_length))
        self.target: np.ndarray | None = None

    def sync(self, length: float, bend: float) -> None:
        self.target = np.array([length, bend], float)

    def step(self, k: KeyState) -> np.ndarray:
        tp, p = self.cfg.teleop, self.cfg.catheter
        mult = tp.fast_multiplier if k.fast else 1.0
        L = self.target[0] + (k.insert - k.retract) * tp.insert_speed * mult * self.dt
        b = 0.0 if k.straighten else self.target[1] + (k.right - k.left) * tp.bend_rate * mult * self.dt
        self.target = np.array([np.clip(L, p.min_length, p.max_length),
                                np.clip(b, -self.bend_max, self.bend_max)])
        return self.target.copy()


class _SyncedCommander(Node):
    """Base: keeps its target in sync with the robot after resets and while another
    commander is driving, so switching sources never makes the target jump."""

    source = ""

    def __init__(self, bus: Bus, cfg: Config, name: str):
        super().__init__(bus, name)
        self.cfg = cfg
        self.integ = KeyIntegrator(cfg, bus.dt)
        self.need_sync = True
        self.active = ""
        self.last_out: JointCommand | None = None
        self.create_subscription("/robot/joint_state", self._on_state)
        self.create_subscription("/robot/joint_target", self._on_out)
        self.create_subscription("/mux/active", self._on_active)
        self.create_subscription("/episode/events", self._on_event_base)

    def _on_state(self, js: JointState) -> None:
        if self.need_sync:
            self.integ.sync(js.length, js.bend)
            self.need_sync = False

    def _on_out(self, cmd: JointCommand) -> None:
        self.last_out = cmd

    def _on_active(self, a: ActiveSource) -> None:
        self.active = a.active

    def _on_event_base(self, ev: Event) -> None:
        if ev.kind == "reset":
            self.need_sync = True
            self.last_out = None

    def follow_if_inactive(self) -> None:
        if self.active != self.source and self.last_out is not None:
            self.integ.sync(self.last_out.length, self.last_out.bend)


class TeleopNode(_SyncedCommander):
    source = "teleop"

    def __init__(self, bus: Bus, cfg: Config):
        super().__init__(bus, cfg, "teleop")
        self.keys: KeyState | None = None
        self.pub = self.create_publisher("/teleop/joint_target")
        self.create_subscription("/ui/keys", self._on_keys)
        self.create_timer(bus.dt, self.update, priority=20)

    def _on_keys(self, k: KeyState) -> None:
        self.keys = k

    def update(self) -> None:
        if self.keys is None or self.need_sync:
            return
        self.follow_if_inactive()
        engaged = self.keys.motion
        target = self.integ.step(self.keys) if (self.active == "teleop" or engaged) else self.integ.target
        self.pub.publish(JointCommand(float(target[0]), float(target[1]), "teleop", engaged))


class AutopilotNode(_SyncedCommander):
    """Scripted expert that follows the vessel centreline using privileged sim
    data, pressing the same virtual keys an operator would."""
    source = "autopilot"

    def __init__(self, bus: Bus, cfg: Config, noise: float = 0.0, seed: int = 0):
        super().__init__(bus, cfg, "autopilot")
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.vessel = None
        self.target: NavTarget | None = None
        self.gt: GroundTruth | None = None
        self.pilot: Autopilot | None = None
        self.pub = self.create_publisher("/autopilot/joint_target")
        self.create_subscription("/sim/world", self._on_world)
        self.create_subscription("/nav/target", self._on_target)
        self.create_subscription("/sim/ground_truth", self._on_truth)
        self.create_subscription("/episode/events", self._on_event)
        self.create_timer(bus.dt, self.update, priority=20)

    def _on_world(self, w: VesselInfo) -> None:
        self.vessel, self.pilot = w.vessel, None

    def _on_target(self, t: NavTarget) -> None:
        self.target, self.pilot = t, None

    def _on_truth(self, gt: GroundTruth) -> None:
        self.gt = gt

    def _on_event(self, ev: Event) -> None:
        if ev.kind == "reset":
            self.pilot = None

    def update(self) -> None:
        if self.need_sync or self.gt is None or self.vessel is None or self.target is None:
            return
        self.follow_if_inactive()
        if self.pilot is None:
            n = self.noise
            self.pilot = Autopilot(self.cfg, self.vessel.path_to(self.target),
                                   lookahead=6.0 * (1 + n * self.rng.uniform(-0.4, 0.4)),
                                   gain=0.35 * (1 + n * self.rng.uniform(-0.4, 0.4)))
        obs = {"tip_pos": self.gt.tip_pos, "tip_angle": self.gt.tip_angle, "cmd_bend": self.integ.target[1]}
        w, s, l, r = self.pilot.act(obs, self.bus.dt)
        if self.noise > 0 and self.rng.random() < 0.15 * self.noise:      # imperfect operator
            w, s, l, r = [k if self.rng.random() > 0.5 else not k for k in (w, s, l, r)]
            w, s = w and not s, s and not w
        target = self.integ.step(KeyState(insert=w, retract=s, left=l, right=r))
        self.pub.publish(JointCommand(float(target[0]), float(target[1]), "autopilot"))


class PolicyNode(Node):
    """Runs a Policy on the live streams and publishes its targets.
    asynchronous=True (real-time mode): inference runs in a worker thread and the
    robot keeps tracking the previous target meanwhile, like on hardware.
    asynchronous=False (lockstep): the sim waits for every act() call."""

    def __init__(self, bus: Bus, cfg: Config, policy: Policy, asynchronous: bool = True):
        super().__init__(bus, "policy")
        self.cfg, self.policy, self.asynchronous = cfg, policy, asynchronous
        self.image: Image | None = None
        self.fmap: FeatureMap | None = None
        self.state: JointState | None = None
        self.vessel = None
        self.cam: CameraInfo | None = None
        self.target: NavTarget | None = None
        self._maps_cache: tuple | None = None
        self.episode = 0
        self.pool = cf.ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending: tuple[int, float, cf.Future] | None = None
        self.last_latency = 0.0
        self.pub = self.create_publisher("/policy/joint_target")
        for topic, cb in [("/camera/image", self._set("image")), ("/nav/feature_map", self._set("fmap")),
                          ("/robot/joint_state", self._set("state")), ("/camera/info", self._set("cam")),
                          ("/nav/target", self._set("target")), ("/sim/world", self._on_world),
                          ("/episode/events", self._on_event)]:
            self.create_subscription(topic, cb)
        self.create_timer(bus.dt, self._poll, priority=19)
        self.create_timer(1.0 / policy.rate_hz, self.update, priority=20)

    def _set(self, attr):
        def cb(msg):
            setattr(self, attr, msg)
        return cb

    def _on_world(self, w: VesselInfo) -> None:
        self.vessel = w.vessel
        self._maps_cache = None

    def _on_event(self, ev: Event) -> None:
        hook = getattr(self.policy, "on_event", None)
        if hook is not None and ev.kind in ("takeover", "resume"):
            hook(ev.kind)                                # optional Policy hook
        if ev.kind == "reset":
            self.episode += 1
            self.pending = None
            self.image = None
            self.policy.reset({"model": self.vessel.name if self.vessel else None,
                               "target": self.target.name if self.target else None,
                               "camera_info": self.cam.to_dict() if self.cam else None})

    def _maps(self):
        """Feature maps at the policy's input size (computed, not resized, so the
        vessel edge stays exact)."""
        size = self.policy.input_size
        if size is None or self.fmap is None:
            return (self.fmap.U, self.fmap.V) if self.fmap is not None else (None, None)
        key = (self.fmap.map_id, tuple(size))
        if self._maps_cache is None or self._maps_cache[0] != key:
            U, V, *_ = compute_feature_maps(self.vessel, self.cam, self.fmap.target_world, size)
            self._maps_cache = (key, U, V)
        return self._maps_cache[1], self._maps_cache[2]

    def _observation(self) -> dict:
        img = self.image.data
        size = self.policy.input_size
        if size is not None:
            img = resize_image(img, size)
        U, V = self._maps()
        s = self.state
        return {"image": img, "feature_u": U, "feature_v": V,
                "joint": np.array([s.length, s.bend]), "joint_vel": np.array([s.length_vel, s.bend_vel]),
                "last_action": np.array([s.target_length, s.target_bend]),
                "t": self.image.stamp, "target_name": self.target.name if self.target else None}

    def update(self) -> None:
        if self.image is None or self.state is None or self.fmap is None or self.cam is None:
            return
        if self.pending is not None:                     # previous inference still running
            return
        obs = self._observation()
        if self.asynchronous:
            self.pending = (self.episode, obs["t"], self.pool.submit(self.policy.act, obs))
        else:
            self._publish(self.policy.act(obs), obs["t"])

    def _poll(self) -> None:
        if self.pending is None or not self.pending[2].done():
            return
        ep, t_obs, fut = self.pending
        self.pending = None
        if ep == self.episode:                           # drop results from a previous episode
            self._publish(fut.result(), t_obs)

    def _publish(self, action, t_obs: float) -> None:
        length, bend = float(action[0]), float(action[1])
        if not (np.isfinite(length) and np.isfinite(bend)):
            return
        self.last_latency = self.now - t_obs
        self.pub.publish(JointCommand(length, bend, "policy", obs_stamp=t_obs))

    def on_shutdown(self) -> None:
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)


class MuxNode(Node):
    """Forwards the active commander's target to /robot/joint_target.
    If the primary commander is not teleop, any operator motion key takes over;
    the 'resume' event (G key) hands control back."""

    SOURCES = ("teleop", "policy", "autopilot")

    def __init__(self, bus: Bus, primary: str):
        super().__init__(bus, "mux")
        self.primary = self.active = primary
        self.latest: dict[str, JointCommand] = {}
        self.pub = self.create_publisher("/robot/joint_target")
        self.pub_active = self.create_publisher("/mux/active", latched=True)
        self.pub_event = self.create_publisher("/episode/events")
        for s in self.SOURCES:
            self.create_subscription(f"/{s}/joint_target", self._make_cb(s))
        self.create_subscription("/episode/events", self._on_event)
        self.create_timer(bus.dt, self.update, priority=30)
        self.pub_active.publish(ActiveSource(self.active, self.primary))

    def _make_cb(self, source: str):
        def cb(cmd: JointCommand):
            self.latest[source] = cmd
            if source == "teleop" and cmd.engaged and self.active != "teleop":
                self._set_active("teleop")
                self.pub_event.publish(Event("takeover"))
        return cb

    def _set_active(self, a: str) -> None:
        self.active = a
        self.pub_active.publish(ActiveSource(a, self.primary))

    def _on_event(self, ev: Event) -> None:
        if ev.kind == "resume" and self.active != self.primary:
            self.latest.pop(self.primary, None)          # don't replay a stale command
            self._set_active(self.primary)
        elif ev.kind == "reset":
            self.latest.clear()
            if self.active != self.primary:
                self._set_active(self.primary)

    def update(self) -> None:
        cmd = self.latest.get(self.active)
        if cmd is None:
            return                                        # robot holds its last target
        src = "takeover" if (self.active == "teleop" and self.primary != "teleop") else self.active
        self.pub.publish(JointCommand(cmd.length, cmd.bend, src, obs_stamp=cmd.obs_stamp))
