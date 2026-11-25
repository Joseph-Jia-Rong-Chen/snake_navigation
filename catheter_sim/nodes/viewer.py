"""ViewerNode: the pygame window. Publishes the keyboard (/ui/keys) and session
events; draws the scene, optional overlays and the HUD from bus topics. It never
publishes images, so nothing drawn here can end up in the recorded data."""
from __future__ import annotations

import numpy as np
import pygame

from ..bus import Bus, Node
from ..config import Config
from ..msgs import (ActiveSource, Catalog, CameraInfo, Event, FeatureMap, GroundTruth, Image,
                    JointCommand, JointState, KeyState, RecorderStatus, TaskStatus, VesselInfo)
from ..render import (PANEL_W, Panel, SceneRenderer, array_to_surface, draw_contacts, draw_entry,
                      draw_targets, feature_overlay)

HELP = [
    "W / S      insert / retract (target)",
    "<- / ->    bend CCW / CW (target)",
    "Shift fast   Space straighten",
    "Enter      start / stop recording",
    "R reset    T next target",
    "N / P      next / prev vessel",
    "G          hand control to the policy",
    "F          feature map overlay U/V/off",
    "I          show raw /camera/image",
    "V roadmap  C forces  H help",
    "Esc        quit",
]


class ViewerNode(Node):
    def __init__(self, bus: Bus, cfg: Config, primary: str):
        super().__init__(bus, "viewer")
        self.cfg, self.primary = cfg, primary
        cw, ch = cfg.camera.width, cfg.camera.height
        self.screen = pygame.display.set_mode((cw + PANEL_W, ch))
        pygame.display.set_caption(f"Catheter - {primary}")
        self.panel = Panel(f"Catheter  [{primary}]")
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 15)
        self.view = pygame.Rect(0, 0, cw, ch)
        self.sr: SceneRenderer | None = None
        self.roadmap, self.contacts_on, self.help_on, self.raw_camera = True, False, True, False
        self.fmode = 0                                   # 0 off, 1 U, 2 V
        self._overlay = None
        self.straighten = False
        self.world: VesselInfo | None = None
        self.cam: CameraInfo | None = None
        for topic, attr in [("/camera/image", "image"), ("/sim/ground_truth", "gt"),
                            ("/robot/joint_state", "state"), ("/robot/joint_target", "cmd"),
                            ("/nav/feature_map", "fmap"), ("/task/status", "status"),
                            ("/task/catalog", "catalog"), ("/mux/active", "active"),
                            ("/recorder/status", "rec")]:
            setattr(self, attr, None)
            self.create_subscription(topic, self._set(attr))
        self.create_subscription("/sim/world", self._on_world)
        self.create_subscription("/camera/info", self._on_cam)
        self.pub_keys = self.create_publisher("/ui/keys")
        self.pub_event = self.create_publisher("/episode/events")
        self.create_timer(bus.dt, self.poll_input, priority=0)
        self.create_timer(bus.dt, self.draw, priority=90)

    def _set(self, attr):
        def cb(msg):
            setattr(self, attr, msg)
        return cb

    def _on_world(self, w: VesselInfo) -> None:
        self.world = w
        self._rebuild()

    def _on_cam(self, c: CameraInfo) -> None:
        self.cam = c
        self._rebuild()

    def _rebuild(self) -> None:
        if self.world and self.cam and self.cam.model == self.world.name:
            self.sr = SceneRenderer(self.world.vessel, self.cam)
            pygame.display.set_caption(f"Catheter - {self.primary} - {self.world.name}")

    # ------------------------------------------------------------------ input
    def poll_input(self) -> None:
        ev_pub = self.pub_event.publish
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                ev_pub(Event("shutdown"))
                self.bus.shutdown()
                return
            if ev.type != pygame.KEYDOWN:
                continue
            k = ev.key
            actions = {pygame.K_RETURN: "record_toggle", pygame.K_r: "reset", pygame.K_n: "next_model",
                       pygame.K_p: "prev_model", pygame.K_t: "next_target", pygame.K_g: "resume"}
            if k in actions:
                ev_pub(Event(actions[k]))
            elif k == pygame.K_SPACE:
                self.straighten = True
            elif k == pygame.K_v:
                self.roadmap = not self.roadmap
            elif k == pygame.K_c:
                self.contacts_on = not self.contacts_on
            elif k == pygame.K_h:
                self.help_on = not self.help_on
            elif k == pygame.K_i:
                self.raw_camera = not self.raw_camera
            elif k == pygame.K_f:
                self.fmode = (self.fmode + 1) % 3
                self._overlay = None
        p = pygame.key.get_pressed()
        self.pub_keys.publish(KeyState(insert=p[pygame.K_w], retract=p[pygame.K_s], left=p[pygame.K_LEFT],
                                       right=p[pygame.K_RIGHT], fast=p[pygame.K_LSHIFT] or p[pygame.K_RSHIFT],
                                       straighten=self.straighten))
        self.straighten = False

    # ------------------------------------------------------------------- draw
    def draw(self) -> None:
        s = self.screen
        if self.sr is None or self.gt is None:
            s.fill((20, 20, 20))
            pygame.display.flip()
            return
        if self.raw_camera and self.image is not None:
            s.blit(array_to_surface(self.image.data), (0, 0))
        else:
            s.blit(self.sr.render(self.gt.centerline, self.cfg.catheter.diameter, self.roadmap), (0, 0))
        if self.fmode and self.fmap is not None:
            key = (self.fmap.map_id, self.fmode)
            if self._overlay is None or self._overlay[0] != key:
                vals = self.fmap.U if self.fmode == 1 else self.fmap.V
                self._overlay = (key, feature_overlay(vals, self.fmap.mask))
            s.blit(self._overlay[1], (0, 0))
            s.blit(self.font.render(f"feature map {'U (x)' if self.fmode == 1 else 'V (y)'}: "
                                    "blue -1 .. white 0 .. red +1", True, (20, 20, 20)), (10, 8))
        if not self.raw_camera:
            draw_targets(s, self.sr, self.font, self.world.vessel.targets,
                         self.status.target_name if self.status else None, self.help_on)
            draw_entry(s, self.sr)
            if self.contacts_on and self.gt.contacts is not None:
                draw_contacts(s, self.sr, self.gt.contacts)
        else:
            s.blit(self.font.render("raw /camera/image", True, (200, 30, 30)), (10, 28))
        self._draw_panel()
        pygame.display.flip()

    def _draw_panel(self) -> None:
        st, js, gt = self.status, self.state, self.gt
        white, grey, dim = (230, 230, 230), (160, 160, 165), (110, 110, 115)
        active = self.active.active if self.active else self.primary
        auto = self.active.auto if self.active else ""
        if active == "teleop":
            mode, mode_col = "driver  you" + (f"   (G: {auto})" if auto else ""), (120, 200, 255)
        else:
            mode, mode_col = f"driver  {active}   (keys: take over)", (240, 170, 60)
        lines = [
            (f"model   {self.world.name}", white),
            (f"target  {st.target_name}  ({st.target_dist:.1f} mm)" if st else "target  -", (90, 220, 120)),
            (mode, mode_col),
            ("", white),
            ("              target    joint", grey),
        ]
        if js is not None:
            lines += [(f"insert [mm] {js.target_length:7.1f}  {js.length:7.1f}", white),
                      (f"bend  [deg] {js.target_bend:+7.1f}  {js.bend:+7.1f}", white)]
        lines += [
            (f"bend meas   {gt.bend_meas:+7.1f} deg", grey),
            (f"tip         ({gt.tip_pos[0]:6.1f}, {gt.tip_pos[1]:6.1f})", grey),
            (f"contacts {gt.n_contacts:3d}  sum|F| {gt.contact_force_total:.3f} N", grey),
            (f"insert force {gt.insertion_force:+.3f} N", grey),
            (f"t {self.now:6.1f}s  rate {self.bus.wall_rate:4.0f} Hz", dim),
        ]
        rec = self.rec
        if rec is not None and rec.recording:
            lines.append((f"frames {rec.frames}", (230, 80, 80)))
        elif rec is not None and rec.message:
            for part in rec.message.split(", "):
                lines.append((part[:40], (240, 200, 90)))
        bend_max = float(np.degrees(self.cfg.catheter.kappa_max * self.cfg.catheter.tip_length))
        self.panel.draw(self.screen, pygame.Rect(self.view.w, 0, PANEL_W, self.view.h), lines,
                        js.target_bend if js else 0.0, js.bend if js else 0.0, gt.bend_meas, bend_max,
                        bool(rec and rec.recording), "TARGET REACHED" if (st and st.success) else None,
                        HELP if self.help_on else None)

    def on_shutdown(self) -> None:
        pygame.display.quit()
