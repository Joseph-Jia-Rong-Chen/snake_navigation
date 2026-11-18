"""Episode recording to compressed .npz (+ a .json sidecar with metadata).

Per-frame arrays (T = number of frames):
  t                 (T,)       sim time [s]
  wall_time         (T,)       real time since episode start [s]
  keys              (T,4)      bool  [W, S, Left, Right]
  fast              (T,)       bool  Shift held
  action            (T,2)      commanded increments this frame (d_length [mm], d_kappa [1/mm]), float64
  init_*                       solver state before the first frame (exact re-simulation)
  length            (T,)       inserted length after the step [mm]
  kappa_cmd         (T,)       commanded tip curvature [1/mm] (+ = clockwise)
  kappa_meas        (T,)       measured mean tip curvature [1/mm]
  tip_pos           (T,2)      tip position [mm]
  tip_angle         (T,)       tip heading [rad] (y up, CCW positive)
  base_angle        (T,)       catheter angle at the entry point [rad]
  shape             (T,200,2)  centreline spline resampled uniformly base->tip [mm]
  n_contacts        (T,)
  contact_force_total, contact_force_max, insertion_force   (T,) [N]
  target_dist       (T,)       tip-to-target distance [mm]
  success           (T,)       bool, tip within the target radius
  images            (T,S,S,3)  uint8 fluoroscopy frames (only with --record-images)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

FIELDS = ["t", "length", "kappa_cmd", "kappa_meas", "tip_pos", "tip_angle", "base_angle",
          "shape", "n_contacts", "contact_force_total", "contact_force_max",
          "insertion_force", "target_dist", "success"]


class Recorder:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.active = False
        self.frames: list[dict] = []
        self.meta: dict = {}

    def start(self, meta: dict, init_state: dict | None = None) -> None:
        """`init_state` = Catheter.get_state() at the first frame (saved as init_*)."""
        self.init_state = init_state or {}
        self.active = True
        self.frames = []
        self.meta = dict(meta)
        self.meta["start_time"] = datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()

    def add(self, obs: dict, keys, fast: bool, action, image: np.ndarray | None = None) -> None:
        if not self.active:
            return
        f = {k: obs[k] for k in FIELDS}
        f["wall_time"] = time.perf_counter() - self._t0
        f["keys"] = np.asarray(keys, bool)
        f["fast"] = bool(fast)
        f["action"] = np.asarray(action, np.float64)
        if image is not None:
            f["images"] = image
        self.frames.append(f)

    def stop(self, extra_meta: dict | None = None) -> Path | None:
        """Finish the episode; returns the saved path (None if empty)."""
        self.active = False
        if not self.frames:
            return None
        meta = {**self.meta, **(extra_meta or {})}
        arrays = {}
        for k in self.frames[0]:
            arrays[k] = np.stack([np.asarray(f[k]) for f in self.frames])
        for k, a in arrays.items():
            if a.dtype == np.float64 and k != "action":
                arrays[k] = a.astype(np.float32)
        succ = arrays["success"]
        meta.update(n_frames=len(self.frames), reached_target=bool(succ.any()),
                    time_to_target=float(arrays["t"][succ.argmax()]) if succ.any() else None,
                    duration=float(arrays["t"][-1]))
        out_dir = self.data_dir / meta.get("model", "unknown")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{meta.get('target', 'none')}"
        path = out_dir / f"{stem}.npz"
        init = {f"init_{k}": v for k, v in self.init_state.items()}
        np.savez_compressed(path, meta=json.dumps(meta), **arrays, **init)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        self.frames = []
        return path


def load_episode(path: str | Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta, {k: z[k] for k in z.files if k != "meta"}


def init_state(data: dict) -> dict | None:
    st = {k[5:]: v for k, v in data.items() if k.startswith("init_")}
    return st or None
