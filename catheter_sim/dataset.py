"""Training-side loader: turns a recorded episode into (observation, action) samples
that match exactly what PolicyNode gives a policy at deployment.

    from catheter_sim.dataset import Episode
    ep = Episode("data/vessel_02/2026..._left_up.npz", size=(256, 256))
    sample = ep[k]   # {image, feature_u, feature_v, joint, joint_vel, last_action, action, ...}

  image                frame k from the .mp4, resized to `size`
  feature_u/v          maps for the frame's target, computed at `size` (like PolicyNode)
  joint, joint_vel     robot state when the frame was taken
  last_action          target the robot was tracking at that moment
  action               label: first target commanded after the frame [length_mm, bend_deg]

Note the .mp4 is H.264 (lossy, but visually near-lossless at crf 18). For
bit-exact frames record with --record-images (stored at recording.image_size).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .featuremap import compute as compute_feature_maps
from .msgs import CameraInfo
from .recorder import load_episode, read_video
from .vessel import VesselModel

ROOT = Path(__file__).resolve().parent.parent


class Episode:
    def __init__(self, path: str | Path, size: tuple[int, int] | None = None, load_video: bool = True):
        self.path = Path(path)
        self.meta, self.d = load_episode(self.path)
        self.cam = CameraInfo.from_dict(self.meta["camera_info"])
        self.size = tuple(size) if size else (self.cam.width, self.cam.height)
        self.frames = read_video(self.path.with_suffix(".mp4"), size) if load_video else None
        self._maps: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.d["t"])

    def feature_maps(self, k: int):
        mi = int(self.d["feature_map_index"][k])
        if mi not in self._maps:
            if self.size == (self.cam.width, self.cam.height):
                self._maps[mi] = (self.d["feature_U"][mi], self.d["feature_V"][mi])
            else:
                mp = Path(self.meta["model_path"])
                vessel = VesselModel.load(mp if mp.exists() else ROOT / "models" / f"{self.meta['model']}.json")
                U, V, *_ = compute_feature_maps(vessel, self.cam, self.d["feature_target_world"][mi], self.size)
                self._maps[mi] = (U, V)
        return self._maps[mi]

    def __getitem__(self, k: int) -> dict:
        d = self.d
        U, V = self.feature_maps(k)
        return {
            "image": self.frames[k] if self.frames is not None else None,
            "feature_u": U, "feature_v": V,
            "joint": np.array([d["joint_length"][k], d["joint_bend"][k]]),
            "joint_vel": np.array([d["joint_length_vel"][k], d["joint_bend_vel"][k]]),
            "last_action": np.array([d["action_length"][k], d["action_bend"][k]]),
            "action": np.array([d["next_action_length"][k], d["next_action_bend"][k]]),
            "source": str(d["action_source"][k]),
            "t": float(d["t"][k]),
        }


def list_episodes(data_dir: str | Path = ROOT / "data") -> list[Path]:
    return sorted(p for p in Path(data_dir).rglob("*.npz") if p.with_suffix(".mp4").exists())
