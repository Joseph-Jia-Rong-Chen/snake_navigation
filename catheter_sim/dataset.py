"""Training-side access to raw episodes: streams aligned to the video frames
(catheter_sim.sync) + frames + feature maps at the same resolution.

    from catheter_sim.dataset import Episode
    ep = Episode("data/raw/vessel_02/20261119_101500_left_up", scale=0.5)
    ep.size                     # (W, H) of everything below
    for k, s in ep.samples():   # streams the video, no need to hold it in memory
        s["image"]              # (H, W) uint8 grayscale frame k
        s["feature_u"], s["feature_v"], s["mask"]    # maps of frame k's target at (H, W)
        s["state"]              # [length_mm, bend_deg] when the frame was taken
        s["action"]             # [length_mm, bend_deg] target commanded in response to it
    ep[k]                       # random access (decodes the whole video once)

`scale` is relative to the recorded video size. Maps at a size other than the
stored one are recomputed from the vessel model (not resized), on the same pixel
grid as the resized frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .featuremap import compute as compute_feature_maps
from .msgs import CameraInfo
from .recorder import iter_video, list_raw, load_raw, read_video
from .sync import align, report
from .vessel import VesselModel

ROOT = Path(__file__).resolve().parent.parent


class Episode:
    def __init__(self, path: str | Path, scale: float = 1.0, size: tuple[int, int] | None = None,
                 action_align: str = "last", state_align: str = "hold"):
        self.meta, self.d = load_raw(path)
        self.dir = Path(path) if Path(path).is_dir() else Path(path).parent
        self.cam = CameraInfo.from_dict(self.meta["camera_info"])
        self.vcam = CameraInfo.from_dict(self.meta["video_camera_info"])
        vw, vh = self.vcam.width, self.vcam.height
        self.size = tuple(size) if size else (max(int(round(vw * scale)), 1), max(int(round(vh * scale)), 1))
        self.out_cam = self.cam.scaled(*self.size)
        self.fps = self.meta["camera_fps"]
        self.a = align(self.d, self.fps, action_align, state_align)
        self._maps: dict[int, tuple] = {}
        self._frames: np.ndarray | None = None
        self._vessel = None

    # --------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.a["t"])

    @property
    def video_path(self) -> Path:
        return self.dir / "video.mp4"

    @property
    def vessel(self) -> VesselModel:
        if self._vessel is None:
            mp = Path(self.meta["model_path"])
            self._vessel = VesselModel.load(mp if mp.exists() else ROOT / "models" / f"{self.meta['model']}.json")
        return self._vessel

    def sync_report(self) -> dict:
        return report(self.a)

    # ----------------------------------------------------------------- maps
    def maps(self, mi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(U, V, mask) of feature map `mi` at self.size."""
        if mi not in self._maps:
            d = self.d
            if self.size == (d["fmap_U"].shape[2], d["fmap_U"].shape[1]):
                self._maps[mi] = (d["fmap_U"][mi], d["fmap_V"][mi], d["fmap_mask"][mi])
            else:
                U, V, mask, *_ = compute_feature_maps(self.vessel, self.cam, d["fmap_target_world"][mi], self.size)
                self._maps[mi] = (U, V, mask)
        return self._maps[mi]

    def target_name(self, k: int) -> str:
        return str(self.d["fmap_target_name"][self.a["map_index"][k]])

    # --------------------------------------------------------------- frames
    def iter_frames(self) -> Iterator[np.ndarray]:
        if "frame_images" in self.d and self.size == (self.vcam.width, self.vcam.height):
            yield from self.d["frame_images"]
        else:
            yield from iter_video(self.video_path, self.size, gray=True)

    def frames(self) -> np.ndarray:
        if self._frames is None:
            self._frames = np.stack(list(self.iter_frames()))
            if len(self._frames) != len(self):
                raise ValueError(f"{self.dir.name}: video has {len(self._frames)} frames, streams {len(self)}")
        return self._frames

    # -------------------------------------------------------------- samples
    def sample(self, k: int, image: np.ndarray | None) -> dict:
        a = self.a
        U, V, mask = self.maps(int(a["map_index"][k]))
        return {
            "image": image, "feature_u": U, "feature_v": V, "mask": mask,
            "state": a["state"][k], "state_vel": a["state_vel"][k], "action": a["action"][k],
            "action_source": a["action_source"][k], "target_name": self.target_name(k),
            "t": float(a["t"][k]), "frame": k,
        }

    def samples(self) -> Iterator[tuple[int, dict]]:
        """(k, sample) for every frame, streaming the video. Raises if the video and
        the frame stream disagree in length."""
        k = -1
        for k, img in enumerate(self.iter_frames()):
            if k >= len(self):
                raise ValueError(f"{self.dir.name}: video has more frames than the frame stream ({len(self)})")
            yield k, self.sample(k, img)
        if k + 1 != len(self):
            raise ValueError(f"{self.dir.name}: video has {k + 1} frames, frame stream {len(self)}")

    def __getitem__(self, k: int) -> dict:
        return self.sample(k, self.frames()[k])


def list_episodes(data_dir: str | Path = ROOT / "data" / "raw") -> list[Path]:
    return list_raw(data_dir)


__all__ = ["Episode", "list_episodes", "read_video"]
