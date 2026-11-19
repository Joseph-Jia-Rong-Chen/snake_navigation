"""RecorderNode: subscribes to the bus and writes episodes (format: recorder.py).

Control: /episode/events record_start | record_stop | record_toggle.
A running recording is saved on reset, model change and shutdown.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from ..bus import Bus, Node
from ..config import Config
from ..msgs import (CameraInfo, Event, FeatureMap, GroundTruth, Image, JointCommand, JointState,
                    KeyState, RecorderStatus, StateSnapshot, TaskStatus, VesselInfo)
from ..recorder import VideoWriter, save_episode
from ..render import array_to_surface, surface_to_array

F64 = {"t", "joint_length", "joint_bend", "action_length", "action_bend",
       "next_action_length", "next_action_bend"}


class RecorderNode(Node):
    def __init__(self, bus: Bus, cfg: Config, data_dir: Path, autorecord: bool = False,
                 meta: dict | None = None):
        super().__init__(bus, "recorder")
        self.cfg, self.data_dir, self.autorecord = cfg, Path(data_dir), autorecord
        self.extra_meta = meta or {}
        self.world: VesselInfo | None = None
        self.cam: CameraInfo | None = None
        self.state: JointState | None = None
        self.cmd: JointCommand | None = None
        self.gt: GroundTruth | None = None
        self.status: TaskStatus | None = None
        self.keys: KeyState | None = None
        self.fmap: FeatureMap | None = None
        self.recording = False
        self.pending_start = False
        self.last_message = ""
        self.pub_status = self.create_publisher("/recorder/status", latched=True)
        self.pub_event = self.create_publisher("/episode/events")
        subs = {"/camera/info": "cam", "/robot/joint_state": "state", "/robot/joint_target": "cmd",
                "/task/status": "status", "/ui/keys": "keys"}
        for topic, attr in subs.items():
            self.create_subscription(topic, self._set(attr))
        self.create_subscription("/sim/world", self.on_world)
        self.create_subscription("/sim/ground_truth", self.on_truth)
        self.create_subscription("/camera/image", self.on_image)
        self.create_subscription("/nav/feature_map", self.on_map)
        self.create_subscription("/sim/snapshot", self.on_snapshot)
        self.create_subscription("/episode/events", self.on_event)
        self.create_timer(bus.dt, self._maybe_start, priority=35)   # after the mux, before the robot
        self._publish_status()

    def _set(self, attr):
        def cb(msg):
            setattr(self, attr, msg)
        return cb

    # ---------------------------------------------------------------- control
    def on_event(self, ev: Event) -> None:
        k = ev.kind
        if k == "record_toggle":
            self.stop() if self.recording else self._request_start()
        elif k == "record_start" and not self.recording:
            self._request_start()
        elif k == "record_stop":
            self.pending_start = False
            self.stop()
        elif k == "reset":
            self.stop()
            if self.autorecord:
                self._request_start()

    def _request_start(self) -> None:
        self.pending_start = True

    def on_world(self, w: VesselInfo) -> None:
        self.stop()
        self.world = w

    def _maybe_start(self) -> None:
        if self.pending_start and not self.recording:
            self.pending_start = False
            self._start()

    def _start(self) -> None:
        if self.world is None or self.cam is None or self.cam.model != self.world.name:
            return
        out_dir = self.data_dir / self.world.name
        out_dir.mkdir(parents=True, exist_ok=True)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tgt = self.fmap.target_name if self.fmap else "none"
        self.path = out_dir / f"{self.stamp}_{tgt}.npz"
        self.video = None
        if self.cfg.recording.record_video:
            try:
                self.video = VideoWriter(self.path.with_suffix(".mp4"), (self.cam.width, self.cam.height),
                                         self.cfg.camera.fps)
            except RuntimeError as e:
                print("video disabled:", e)
        self.rows: list[dict] = []
        self.ctrl: list[tuple] = []
        self.maps: list[FeatureMap] = []
        self.map_ids: dict[int, int] = {}
        self.init_state: dict = {}
        self.t0 = self.now
        self.recording = True
        if self.fmap is not None:
            self._add_map(self.fmap)
        self.pub_event.publish(Event("snapshot_request"))
        self.last_message = "recording..."
        self._publish_status()

    def on_snapshot(self, s: StateSnapshot) -> None:
        if self.recording and not self.init_state:
            self.init_state = s.state

    # ------------------------------------------------------------------ data
    def _add_map(self, m: FeatureMap) -> None:
        if m.map_id not in self.map_ids:
            self.map_ids[m.map_id] = len(self.maps)
            self.maps.append(m)

    def on_map(self, m: FeatureMap) -> None:
        self.fmap = m
        if self.recording:
            self._add_map(m)

    def on_truth(self, gt: GroundTruth) -> None:
        self.gt = gt
        if self.recording and self.state is not None:
            s = self.state
            self.ctrl.append((s.stamp, s.target_length, s.target_bend, s.length, s.bend, *gt.tip_pos))
            if self.rows and self.rows[-1]["next_action_length"] is None and s.stamp > self.rows[-1]["t"]:
                # first command after the last frame = the response to that frame
                self.rows[-1]["next_action_length"] = s.target_length
                self.rows[-1]["next_action_bend"] = s.target_bend

    def on_image(self, img: Image) -> None:
        if not self.recording or self.state is None or self.gt is None:
            return
        s, gt, st = self.state, self.gt, self.status
        keys = self.keys.as_array() if self.keys is not None else np.zeros(4, bool)
        row = {
            "t": img.stamp, "frame": img.frame,
            "joint_length": s.length, "joint_bend": s.bend,
            "joint_length_vel": s.length_vel, "joint_bend_vel": s.bend_vel,
            "action_length": s.target_length, "action_bend": s.target_bend,
            "next_action_length": None, "next_action_bend": None,
            "action_source": self.cmd.source if self.cmd is not None else "hold",
            "keys": keys,
            "feature_map_index": self.map_ids.get(self.fmap.map_id, -1) if self.fmap else -1,
            "tip_pos": gt.tip_pos, "tip_angle": gt.tip_angle, "base_angle": gt.base_angle,
            "shape": gt.shape, "bend_meas": gt.bend_meas, "n_contacts": gt.n_contacts,
            "contact_force_total": gt.contact_force_total, "contact_force_max": gt.contact_force_max,
            "insertion_force": gt.insertion_force,
            "target_dist": st.target_dist if st else np.nan, "success": bool(st.success) if st else False,
        }
        if self.cfg.recording.record_images:
            n = self.cfg.recording.image_size
            row["images"] = surface_to_array(pygame.transform.smoothscale(array_to_surface(img.data), (n, n)))
        if self.video is not None:
            self.video.write(img.data)
        self.rows.append(row)
        if len(self.rows) % 15 == 0:
            self._publish_status()

    # ------------------------------------------------------------------ save
    def stop(self) -> Path | None:
        if not self.recording:
            return None
        self.recording = False
        if self.video is not None:
            self.video.close()
        if not self.rows:
            if self.video is not None:
                self.video.path.unlink(missing_ok=True)
            self.last_message = "nothing recorded"
            self._publish_status()
            return None
        last = self.rows[-1]
        if last["next_action_length"] is None:                 # no command after the last frame
            last["next_action_length"], last["next_action_bend"] = last["action_length"], last["action_bend"]
        arrays = {}
        for k in self.rows[0]:
            a = np.stack([np.asarray(r[k]) for r in self.rows])
            arrays[k] = a.astype(np.float32) if (a.dtype == np.float64 and k not in F64) else a
        if self.maps:
            arrays.update(
                feature_U=np.stack([m.U for m in self.maps]), feature_V=np.stack([m.V for m in self.maps]),
                vessel_mask=np.stack([m.mask for m in self.maps]),
                feature_target_index=np.stack([m.target_index for m in self.maps]),
                feature_target_world=np.stack([m.target_world for m in self.maps]),
                feature_target_name=np.array([m.target_name for m in self.maps]),
                feature_c=np.array([m.c for m in self.maps]))
        c = np.array(self.ctrl, np.float64)
        arrays.update(ctrl_t=c[:, 0], ctrl_target_length=c[:, 1], ctrl_target_bend=c[:, 2],
                      ctrl_joint_length=c[:, 3], ctrl_joint_bend=c[:, 4], ctrl_tip_pos=c[:, 5:7])
        arrays.update({f"init_{k}": np.asarray(v) for k, v in self.init_state.items()})

        succ = arrays["success"]
        final_target = self.maps[-1].target_name if self.maps else "none"
        meta = {
            "model": self.world.name, "model_path": self.world.path,
            "target": final_target, "targets": [m.target_name for m in self.maps],
            "camera_info": self.cam.to_dict(), "camera_fps": self.cfg.camera.fps,
            "sim_dt": self.bus.dt, "start_time": self.stamp, "start_t": self.t0,
            "n_frames": len(self.rows), "duration": float(arrays["t"][-1] - arrays["t"][0]),
            "reached_target": bool(succ.any()),
            "time_to_target": float(arrays["t"][succ.argmax()] - self.t0) if succ.any() else None,
            "sources": sorted(set(arrays["action_source"].tolist())),
            "config": self.cfg.to_dict(), **self.extra_meta,
        }
        path = self.path.with_name(f"{self.stamp}_{final_target}.npz")
        if self.video is not None:
            if path != self.path:
                self.video.path.rename(path.with_suffix(".mp4"))
            meta["video"] = path.with_suffix(".mp4").name
        save_episode(path, meta, arrays)
        self.last_message = f"saved {path.name} ({len(self.rows)} frames)"
        self.last_path = path
        self._publish_status()
        return path

    def _publish_status(self) -> None:
        self.pub_status.publish(RecorderStatus(self.recording, len(getattr(self, "rows", [])) if self.recording else 0,
                                               str(getattr(self, "path", "")), self.last_message))

    def on_shutdown(self) -> None:
        p = self.stop()
        if p:
            print("saved", p)
