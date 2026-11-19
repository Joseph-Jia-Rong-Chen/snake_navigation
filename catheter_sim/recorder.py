"""Episode file format + low-level I/O (video writer, loader).

Each episode = three files with the same stem in data/<model>/:
  <stem>.mp4    plain /camera/image stream (vessel + catheter, no UI), camera fps.
                Row k of the .npz is video frame k.
  <stem>.npz    arrays below
  <stem>.json   metadata (model, targets, camera_info, config, success, ...)

Per camera frame (T rows, camera fps):
  t                         sim time of the frame [s]
  frame                     camera frame counter
  joint_length, joint_bend  OBSERVATION: robot joint state when the frame was taken [mm, deg]
  joint_length_vel, joint_bend_vel
  action_length, action_bend  target the robot was tracking at that moment [mm, deg]
  next_action_length, next_action_bend
                            ACTION LABEL: first target commanded after the frame
                            (what the operator/policy did in response to it)
  action_source             teleop | policy | autopilot | takeover | hold (no command yet)
  keys                      (T,4) bool [W, S, Left, Right] (operator keyboard, if any)
  feature_map_index         row into the feature_* arrays (the target can change mid-episode)
  tip_pos, tip_angle, base_angle, shape (T,200,2), bend_meas,
  n_contacts, contact_force_total, contact_force_max, insertion_force,
  target_dist, success      sim-only labels (never available on a real robot)
  images                    (T,S,S,3) uint8 lossless frames, only with --record-images

Feature maps (K = number of targets used in the episode, usually 1):
  feature_U, feature_V      (K,H,W) float32  U=(i-x_a)/c*m+(1-m), V=(j-y_a)/c*m+(1-m)
  vessel_mask               (K,H,W) uint8
  feature_target_index      (K,2)  (x_a, y_a) in pixel-index coordinates
  feature_target_world      (K,2)  target position [mm]
  feature_target_name       (K,)
  feature_c                 (K,)

Control stream at the sim rate (exact re-simulation, actuator analysis):
  ctrl_t, ctrl_target_length, ctrl_target_bend, ctrl_joint_length, ctrl_joint_bend, ctrl_tip_pos
  init_*                    full sim state before the first control tick
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoWriter:
    """Pipes raw RGB frames to ffmpeg (H.264) from a background thread."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float):
        exe = find_ffmpeg()
        if exe is None:
            raise RuntimeError("ffmpeg not found (install it or `pip install imageio-ffmpeg`)")
        w, h = size[0] // 2 * 2, size[1] // 2 * 2          # H.264 needs even dimensions
        self.size, self.path, self.n = (w, h), Path(path), 0
        self.proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=120)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (buf := self.q.get()) is not None:
            self.proc.stdin.write(buf)

    def write(self, rgb: np.ndarray) -> int:
        w, h = self.size
        self.q.put(np.ascontiguousarray(rgb[:h, :w]).tobytes())
        self.n += 1
        return self.n - 1

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        self.proc.stdin.close()
        self.proc.wait()


def read_video(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Decode an episode .mp4 to (T,H,W,3) uint8, optionally resized to size=(W,H)."""
    exe = find_ffmpeg()
    probe = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True).stderr
    import re
    w, h = map(int, re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", probe).groups())
    args = [exe, "-loglevel", "error", "-i", str(path)]
    if size is not None:
        args += ["-vf", f"scale={size[0]}:{size[1]}:flags=area"]
        w, h = size
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(args, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def save_episode(path: Path, meta: dict, arrays: dict) -> None:
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_episode(path: str | Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta, {k: z[k] for k in z.files if k != "meta"}


def init_state(data: dict) -> dict | None:
    st = {k[5:]: v for k, v in data.items() if k.startswith("init_")}
    return st or None
