"""Raw episode format + low-level I/O (video writer / reader, save / load).

A raw episode is a directory  <data_dir>/<model>/<YYYYmmdd_HHMMSS>_<target>/ :

  meta.json     format, model, camera_info (camera), video_camera_info (video
                resolution), camera_fps, encoding, sim_dt, start/stop time, targets,
                success, sources, full config, commander / policy
  video.mp4     the /camera/image stream (no UI), grayscale, H.264, resized to the
                video resolution; video frame k  <->  frame_* row k
  streams.npz   every recorded topic as its own stream, exactly as it arrived

Nothing is paired or resampled at record time. Each stream row has
  <s>_t      stamp: sim time at which the data was valid [s] (float64)
  <s>_seq    bus sequence number: total order of all messages (int64)
  <s>_recv   sim time the recorder received it (= _t unless there is latency)

Streams (prefix: fields)
  frame_   index (camera frame counter)              /camera/image   camera fps
           images (T,h,w) uint8 lossless, only with recording.record_images
  state_   length, bend, length_vel, bend_vel,       /robot/joint_state  sim rate
           target_length, target_bend (the target the actuator tracked that tick)
  cmd_     length, bend, source, engaged, obs_stamp  /robot/joint_target (mux output)
  fmap_    id, target_name, target_world, target_index (x_a, y_a in VIDEO pixels),
           c, U, V, mask (K,h,w) at the VIDEO resolution  /nav/feature_map, per target
  keys_    (N,6) bool  W S Left Right Shift Space    /ui/keys (window only)
  event_   kind, data (json)                         /episode/events
  gt_      tip_pos, tip_angle, base_angle, bend_meas, n_contacts, contact_force_total,
           contact_force_max, insertion_force, shape (N,200,2)
                                                     /sim/ground_truth (sim-only labels)
  status_  target_name, target_dist, success         /task/status
  init_*   full sim state before the first recorded tick (exact re-simulation)

Alignment to the camera timeline is done afterwards by catheter_sim.sync.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

FORMAT = "catheter-raw/2"


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def even_size(width: float, height: float) -> tuple[int, int]:
    """Nearest even (W, H) (H.264 with 4:2:0 chroma needs even dimensions)."""
    return max(int(round(width / 2)) * 2, 2), max(int(round(height / 2)) * 2, 2)


class VideoWriter:
    """Pipes raw frames to ffmpeg (H.264, crf 18) from a background thread.
    Frames are given at `in_size` (W, H); ffmpeg resizes them to `out_size`
    (area filter) before encoding."""

    def __init__(self, path: Path, in_size: tuple[int, int], fps: float,
                 out_size: tuple[int, int] | None = None, gray: bool = True):
        exe = find_ffmpeg()
        if exe is None:
            raise RuntimeError("ffmpeg not found (install it or `pip install imageio-ffmpeg`)")
        self.in_size = tuple(in_size)
        self.size = tuple(out_size) if out_size else even_size(*in_size)
        self.path, self.n, self.gray = Path(path), 0, gray
        w, h = self.in_size
        args = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24",
                "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
        if self.size != self.in_size:
            args += ["-vf", f"scale={self.size[0]}:{self.size[1]}:flags=area"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=120)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (buf := self.q.get()) is not None:
            self.proc.stdin.write(buf)

    def write(self, frame: np.ndarray) -> int:
        self.q.put(np.ascontiguousarray(frame).tobytes())
        self.n += 1
        return self.n - 1

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        self.proc.stdin.close()
        self.proc.wait()


def probe_video(path: str | Path) -> tuple[int, int]:
    probe = subprocess.run([find_ffmpeg(), "-i", str(path)], capture_output=True, text=True).stderr
    w, h = map(int, re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", probe).groups())
    return w, h


def iter_video(path: str | Path, size: tuple[int, int] | None = None, gray: bool = True) -> Iterator[np.ndarray]:
    """Stream decoded frames: (H, W) uint8 if gray else (H, W, 3), optionally
    resized to size = (W, H) with the area filter."""
    w, h = probe_video(path)
    args = [find_ffmpeg(), "-loglevel", "error", "-i", str(path)]
    if size is not None and tuple(size) != (w, h):
        args += ["-vf", f"scale={size[0]}:{size[1]}:flags=area"]
        w, h = size
    c = 1 if gray else 3
    args += ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    n = w * h * c
    try:
        while len(buf := proc.stdout.read(n)) == n:
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(h, w) if gray else a.reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def read_video(path: str | Path, size: tuple[int, int] | None = None, gray: bool = True) -> np.ndarray:
    """All frames as one array (T, H, W) or (T, H, W, 3)."""
    return np.stack(list(iter_video(path, size, gray)))


def save_raw(directory: Path, meta: dict, arrays: dict) -> None:
    directory = Path(directory)
    np.savez_compressed(directory / "streams.npz", **arrays)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def load_raw(directory: str | Path) -> tuple[dict, dict]:
    directory = Path(directory)
    if directory.name in ("meta.json", "streams.npz", "video.mp4"):
        directory = directory.parent
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "streams.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    return meta, d


def list_raw(root: str | Path) -> list[Path]:
    """All raw episode directories below root (sorted)."""
    return sorted(p.parent for p in Path(root).rglob("meta.json") if (p.parent / "streams.npz").exists())


def init_state(data: dict) -> dict | None:
    st = {k[5:]: v for k, v in data.items() if k.startswith("init_")}
    return st or None
