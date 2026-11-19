"""Target feature maps.

    U[j, i] = (i - x_a) / c * m[j, i] + (1 - m[j, i])
    V[j, i] = (j - y_a) / c * m[j, i] + (1 - m[j, i])

  i, j      pixel column (x, + right) and row (y, + down) indices
  m         vessel mask: 1 if the pixel centre is inside the lumen, else 0
  x_a, y_a  target centre in the same pixel-index coordinates
  c         max(W, H) - 1: the largest possible per-axis pixel distance, so
            U, V lie in [-1, 1] inside the vessel; outside they are exactly 1.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .msgs import CameraInfo


def vessel_mask(vessel, cam: CameraInfo) -> np.ndarray:
    """(H, W) uint8 mask, sampled at pixel centres from the vessel's distance field."""
    jj, ii = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    x, y = cam.px_to_world(ii + 0.5, jj + 0.5)
    fx = (x - vessel.origin[0]) / vessel.h
    fy = (y - vessel.origin[1]) / vessel.h
    sdf = map_coordinates(vessel.sdf, [fy.ravel(), fx.ravel()], order=1, mode="nearest")
    return (sdf.reshape(cam.height, cam.width) < 0).astype(np.uint8)


def feature_maps(mask: np.ndarray, target_index: np.ndarray, c: float | None = None):
    """U, V (float32) from a mask and the target centre (x_a, y_a) in pixel indices."""
    H, W = mask.shape
    c = float(max(W, H) - 1) if c is None else float(c)
    m = mask.astype(np.float32)
    i = np.arange(W, dtype=np.float32)[None, :]
    j = np.arange(H, dtype=np.float32)[:, None]
    xa, ya = float(target_index[0]), float(target_index[1])
    U = (i - xa) / c * m + (1.0 - m)
    V = (j - ya) / c * m + (1.0 - m)
    return U.astype(np.float32), V.astype(np.float32), c


def compute(vessel, cam: CameraInfo, target_world: np.ndarray, size: tuple[int, int] | None = None):
    """Everything for one target at the camera resolution or another `size` (W, H).
    Returns U, V, mask, c, (x_a, y_a)."""
    if size is not None and (size[0], size[1]) != (cam.width, cam.height):
        cam = cam.scaled(*size)
    mask = vessel_mask(vessel, cam)
    xa_ya = cam.world_to_index(np.asarray(target_world, float))[0]
    U, V, c = feature_maps(mask, xa_ya)
    return U, V, mask, c, xa_ya
