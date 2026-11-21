"""RecorderNode: logs every relevant topic as its own timestamped stream
(format: recorder.py). No pairing or resampling happens here; catheter_sim.sync
aligns the streams to the camera timeline afterwards.

Control: /episode/events record_start | record_stop | record_toggle.
A running recording is saved on reset, model change and shutdown, and, with
recording.stop_on_success, as soon as the tip reaches the target: the recorder
stops right after the first camera frame captured after the success, so the
saved video ends on the frame showing the tip in the target.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import featuremap
from ..bus import Bus, Node
from ..config import Config
from ..msgs import (CameraInfo, Event, FeatureMap, GroundTruth, Image, JointCommand, JointState,
                    KeyState, RecorderStatus, StateSnapshot, TaskStatus, VesselInfo)
from ..recorder import FORMAT, VideoWriter, even_size, save_raw
from ..render import resize_image, to_gray

# stream -> field names (after the common t, seq, recv)
STREAMS = {
    "frame": ["index"],
    "state": ["length", "bend", "length_vel", "bend_vel", "target_length", "target_bend"],
    "cmd": ["length", "bend", "source", "engaged", "obs_stamp"],
    "keys": ["keys"],
    "event": ["kind", "data"],
    "gt": ["tip_pos", "tip_angle", "base_angle", "bend_meas", "n_contacts", "contact_force_total",
           "contact_force_max", "insertion_force", "shape"],
    "status": ["target_name", "target_dist", "success"],
}
FLOAT32 = {"gt_shape", "gt_contact_force_total", "gt_contact_force_max", "gt_insertion_force"}


class RecorderNode(Node):
    def __init__(self, bus: Bus, cfg: Config, data_dir: Path, autorecord: bool = False,
                 meta: dict | None = None):
        super().__init__(bus, "recorder")
        self.cfg, self.data_dir, self.autorecord = cfg, Path(data_dir), autorecord
        self.extra_meta = meta or {}
        self.world: VesselInfo | None = None
        self.cam: CameraInfo | None = None
        self.fmap: FeatureMap | None = None
        self.recording = False
        self.pending_start = False
        self.last_message = ""
        self.n_frames = 0
        self.success_seq: int | None = None    # seq of the status that reported success
        self.was_success = False
        self.pub_status = self.create_publisher("/recorder/status", latched=True)
        self.pub_event = self.create_publisher("/episode/events")
        self.create_subscription("/camera/info", self.on_cam)
        self.create_subscription("/sim/world", self.on_world)
        self.create_subscription("/camera/image", self.on_image)
        self.create_subscription("/robot/joint_state", self.on_state)
        self.create_subscription("/robot/joint_target", self.on_cmd)
        self.create_subscription("/nav/feature_map", self.on_map)
        self.create_subscription("/sim/ground_truth", self.on_truth)
        self.create_subscription("/task/status", self.on_status)
        self.create_subscription("/ui/keys", self.on_keys)
        self.create_subscription("/sim/snapshot", self.on_snapshot)
        self.create_subscription("/episode/events", self.on_event)
        self.create_timer(bus.dt, self._maybe_start, priority=35)   # after the mux, before the robot
        self._publish_status()

    # ---------------------------------------------------------------- control
    def on_event(self, ev: Event) -> None:
        if self.recording:
            self._log("event", ev, ev.kind, json.dumps(ev.data, default=str))
        k = ev.kind
        if k == "record_toggle":
            self.stop() if self.recording else self._request_start()
        elif k == "record_start" and not self.recording:
            self._request_start()
        elif k == "record_stop":
            self.pending_start = False
            self.stop("manual")
        elif k == "reset":
            self.stop("reset")
            if self.autorecord:
                self._request_start()

    def _request_start(self) -> None:
        self.pending_start = True

    def on_world(self, w: VesselInfo) -> None:
        self.stop("model_change")
        self.world = w

    def on_cam(self, c: CameraInfo) -> None:
        self.cam = c

    def _maybe_start(self) -> None:
        if self.pending_start and not self.recording:
            self.pending_start = False
            self._start()

    def _start(self) -> None:
        if self.world is None or self.cam is None or self.cam.model != self.world.name:
            return
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.data_dir / self.world.name / f"{self.stamp}_recording"
        self.dir.mkdir(parents=True, exist_ok=True)
        s = self.cfg.recording.video_scale
        self.vsize = even_size(self.cam.width * s, self.cam.height * s)
        self.vcam = self.cam.scaled(*self.vsize)
        self.mono = self.cfg.camera.encoding == "mono8"
        self.video = None
        if self.cfg.recording.record_video:
            try:
                self.video = VideoWriter(self.dir / "video.mp4", (self.cam.width, self.cam.height),
                                         self.cfg.camera.fps, self.vsize, gray=self.mono)
            except RuntimeError as e:
                print("video disabled:", e)
        self.rows: dict[str, list] = {k: [] for k in STREAMS}
        self.images: list[np.ndarray] = []
        self.maps: list[dict] = []
        self.map_ids: set[int] = set()
        self.init_state: dict = {}
        self.t0, self.seq0, self.wall0 = self.now, self.bus.seq, time.time()
        self.n_frames = 0
        self.success_seq = None
        self.recording = True
        if self.fmap is not None:
            self._add_map(self.fmap)          # the active target's map (published before the start)
        self.pub_event.publish(Event("snapshot_request"))
        self.last_message = "recording..."
        self._publish_status()

    def on_snapshot(self, s: StateSnapshot) -> None:
        if self.recording and not self.init_state:
            self.init_state = s.state

    # ------------------------------------------------------------------ data
    def _log(self, stream: str, msg, *values) -> None:
        self.rows[stream].append((msg.stamp, msg.seq, self.now, *values))

    def _add_map(self, m: FeatureMap) -> None:
        """Store the map at the VIDEO resolution (computed from the vessel's distance
        field at the video's pixel centres, not resized), so map pixel (i, j) is
        video pixel (i, j)."""
        if m.map_id in self.map_ids:
            return
        self.map_ids.add(m.map_id)
        if self.vsize == (m.U.shape[1], m.U.shape[0]):
            U, V, mask, c, xa_ya = m.U, m.V, m.mask, m.c, m.target_index
        else:
            U, V, mask, c, xa_ya = featuremap.compute(self.world.vessel, self.cam, m.target_world, self.vsize)
        self.maps.append(dict(t=m.stamp, seq=m.seq, recv=self.now, id=m.map_id, target_name=m.target_name,
                              target_world=m.target_world, target_index=xa_ya, c=c, U=U, V=V, mask=mask))

    def on_map(self, m: FeatureMap) -> None:
        self.fmap = m
        if self.recording:
            self._add_map(m)

    def on_state(self, s: JointState) -> None:
        if self.recording:
            self._log("state", s, s.length, s.bend, s.length_vel, s.bend_vel, s.target_length, s.target_bend)

    def on_cmd(self, c: JointCommand) -> None:
        if self.recording:
            self._log("cmd", c, c.length, c.bend, c.source, c.engaged,
                      np.nan if c.obs_stamp is None else c.obs_stamp)

    def on_keys(self, k: KeyState) -> None:
        if self.recording:
            self._log("keys", k, np.array([k.insert, k.retract, k.left, k.right, k.fast, k.straighten], bool))

    def on_truth(self, g: GroundTruth) -> None:
        if self.recording:
            self._log("gt", g, g.tip_pos, g.tip_angle, g.base_angle, g.bend_meas, g.n_contacts,
                      g.contact_force_total, g.contact_force_max, g.insertion_force, g.shape)

    def on_status(self, st: TaskStatus) -> None:
        if self.recording:
            self._log("status", st, st.target_name, st.target_dist, st.success)
            # rising edge only, so a recording started inside the target does not stop at once
            if (self.cfg.recording.stop_on_success and st.success and not self.was_success
                    and self.success_seq is None):
                self.success_seq = st.seq
        self.was_success = st.success

    def on_image(self, img: Image) -> None:
        if not self.recording or img.seq < self.seq0:       # frame captured before the start
            return
        data = img.data
        if self.mono and data.ndim == 3:
            data = to_gray(data)
        if self.video is not None:
            self.video.write(data)
        if self.cfg.recording.record_images:
            self.images.append(resize_image(data, self.vsize))
        self._log("frame", img, img.frame)
        self.n_frames += 1
        if self.success_seq is not None and img.seq > self.success_seq:
            self.stop("target_reached")          # this frame shows the tip in the target
            return
        if self.n_frames % 15 == 0:
            self._publish_status()

    # ------------------------------------------------------------------ save
    def _arrays(self) -> dict:
        out = {}
        for name, fields in STREAMS.items():
            rows = self.rows[name]
            cols = list(zip(*rows)) if rows else [[]] * (3 + len(fields))
            out[f"{name}_t"] = np.asarray(cols[0], np.float64)
            out[f"{name}_seq"] = np.asarray(cols[1], np.int64)
            out[f"{name}_recv"] = np.asarray(cols[2], np.float64)
            for f, col in zip(fields, cols[3:]):
                key = f"{name}_{f}"
                a = np.asarray(col) if rows else np.zeros(0)
                if a.dtype.kind == "U" or a.dtype == object:
                    a = np.asarray([str(x) for x in col])
                out[key] = a.astype(np.float32) if key in FLOAT32 else a
        if self.images:
            out["frame_images"] = np.stack(self.images)
        if self.maps:
            for k in ("t", "seq", "recv", "id", "target_name", "target_world", "target_index", "c"):
                out[f"fmap_{k}"] = np.asarray([m[k] for m in self.maps])
            for k in ("U", "V", "mask"):
                out[f"fmap_{k}"] = np.stack([m[k] for m in self.maps])
        out.update({f"init_{k}": np.asarray(v) for k, v in self.init_state.items()})
        return out

    def stop(self, reason: str = "manual") -> Path | None:
        if not self.recording:
            return None
        self.recording = False
        if self.video is not None:
            self.video.close()
        if not self.rows["frame"]:
            for f in self.dir.iterdir():
                f.unlink()
            self.dir.rmdir()
            self.last_message = "nothing recorded"
            self._publish_status()
            return None
        a = self._arrays()
        succ = a["status_success"].astype(bool) if len(a["status_success"]) else np.zeros(0, bool)
        final_target = self.maps[-1]["target_name"] if self.maps else "none"
        meta = {
            "format": FORMAT,
            "model": self.world.name, "model_path": self.world.path,
            "split": getattr(self.world.vessel, "split", "train"),
            "target": final_target, "targets": [m["target_name"] for m in self.maps],
            "camera_info": self.cam.to_dict(), "video_camera_info": self.vcam.to_dict(),
            "video_size": list(self.vsize), "camera_fps": self.cfg.camera.fps,
            "encoding": "mono8" if self.mono else "rgb8", "camera_latency": self.cfg.camera.latency,
            "sim_dt": self.bus.dt, "rate_hz": self.bus.rate_hz,
            "start_t": self.t0, "stop_t": self.now, "start_seq": self.seq0, "stop_seq": self.bus.seq,
            "start_time": self.stamp, "wall_duration": time.time() - self.wall0,
            "n_frames": len(a["frame_t"]),
            "duration": float(a["frame_t"][-1] - a["frame_t"][0]),
            "reached_target": bool(succ.any()),
            "time_to_target": float(a["status_t"][succ.argmax()] - self.t0) if succ.any() else None,
            "sources": sorted(set(a["cmd_source"].tolist())) or ["hold"],
            "stop_reason": reason,
            "streams": {k: int(len(a[f"{k}_t"])) for k in STREAMS},
            "video": "video.mp4" if self.video is not None else None,
            "config": self.cfg.to_dict(), **self.extra_meta,
        }
        save_raw(self.dir, meta, a)
        final = self.dir.with_name(f"{self.stamp}_{final_target}")
        n = 1
        while final.exists():
            final = self.dir.with_name(f"{self.stamp}_{final_target}_{n}")
            n += 1
        self.dir.rename(final)
        self.last_message = (("target reached, " if reason == "target_reached" else "")
                             + f"saved {final.name} ({meta['n_frames']} frames)")
        self.last_path = final
        self._publish_status()
        return final

    def _publish_status(self) -> None:
        self.pub_status.publish(RecorderStatus(self.recording, self.n_frames if self.recording else 0,
                                               str(getattr(self, "dir", "")), self.last_message))

    def on_shutdown(self) -> None:
        p = self.stop("shutdown")
        if p:
            print("saved", p)
