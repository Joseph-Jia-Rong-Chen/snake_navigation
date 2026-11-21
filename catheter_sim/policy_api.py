"""The interface a learned policy implements to drive the robot.

    class MyPolicy(Policy):
        rate_hz = 10                  # how often act() is called
        input_size = (320, 180)       # (W, H) of obs image + feature maps (None = camera size);
                                      # use the size of the dataset the policy was trained on

        def reset(self, info):        # new episode: info = {model, target, camera_info}
            ...

        def act(self, obs):           # -> (target_length_mm, target_bend_deg)
            ...

        def on_event(self, kind):     # optional: "takeover" / "resume" (operator took / gave back control)
            ...

`obs` contains only what a real system can observe:
    image          (H, W) uint8      plain grayscale camera frame (vessel + black catheter +
                                     bright tip fiducial); (H, W, 3) if camera.encoding = rgb8
    feature_u      (H, W) float32    U = (i - x_a)/c * m + (1 - m)
    feature_v      (H, W) float32    V = (j - y_a)/c * m + (1 - m)
    joint          [length_mm, bend_deg]       current joint state
    joint_vel      [mm/s, deg/s]
    last_action    [length_mm, bend_deg]       target the robot is currently tracking
    t              capture stamp of the image [s]
    target_name    str

The returned target goes through the same mux -> robot (clamp, lag, speed limit)
path as teleop, so a large jump is executed smoothly. Return absolute targets;
for a delta-style policy use `obs["last_action"] + delta`.

For a policy trained on the LeRobot export, catheter_sim.lerobot_export.
observation_to_frame(obs) gives the observation in exactly the dataset's layout.

Load from the command line with  --policy path/to/file.py:ClassName
(constructor kwargs via --policy-arg key=value).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np


class Policy:
    rate_hz: float = 10.0
    input_size: tuple[int, int] | None = None

    def reset(self, info: dict) -> None:
        pass

    def act(self, obs: dict) -> tuple[float, float]:
        raise NotImplementedError


def load_policy(spec: str, kwargs: dict | None = None) -> Policy:
    """`spec` = 'path/to/file.py:ClassName' or 'package.module:ClassName'."""
    target, _, cls_name = spec.rpartition(":")
    if not target:
        raise ValueError("policy spec must look like file.py:ClassName")
    if target.endswith(".py"):
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, cls_name)(**(kwargs or {}))


def parse_kwargs(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def find_tip_pixel(image: np.ndarray, gray_threshold: int = 215) -> np.ndarray | None:
    """(x, y) pixel of the tip fiducial in a camera frame, or None. Grayscale frames:
    the bright marker (camera.tip_gray); RGB frames: the red marker."""
    if image.ndim == 2:
        ys, xs = np.nonzero(image >= gray_threshold)
    else:
        r, g, b = image[..., 0].astype(int), image[..., 1].astype(int), image[..., 2].astype(int)
        ys, xs = np.nonzero((r > 170) & (g < 90) & (b < 90))
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])
