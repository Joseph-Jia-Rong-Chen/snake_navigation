"""2D vessel phantoms.

A vessel model is a JSON file (see ``models/``) holding a list of vessels.
Each vessel is a centreline given by a few control points (interpolated with
a chord-length cubic spline) and a radius per control point (or a single
radius). The lumen is the smooth union of all tapered tubes, so a branch that
starts on its parent's centreline forms a bifurcation automatically.

For contact, the lumen is turned into a signed distance field on a regular
grid: sdf < 0 inside the lumen, sdf > 0 inside the wall. The catheter must keep
``sdf(x) + r_catheter <= 0``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt


@dataclass
class Vessel:
    name: str
    points: np.ndarray   # (M,2) dense centreline samples [mm]
    radii: np.ndarray    # (M,)
    parent: str | None = None


@dataclass
class Target:
    name: str
    position: np.ndarray
    radius: float


def smooth_centerline(ctrl: np.ndarray, radii: np.ndarray, step: float = 0.5):
    """Chord-length cubic spline through control points, sampled every `step` mm."""
    ctrl = np.asarray(ctrl, float)
    radii = np.broadcast_to(np.asarray(radii, float), (len(ctrl),)).copy()
    if len(ctrl) == 2:
        L = np.linalg.norm(ctrl[1] - ctrl[0])
        n = max(int(np.ceil(L / step)), 1) + 1
        t = np.linspace(0, 1, n)[:, None]
        return ctrl[0] + t * (ctrl[1] - ctrl[0]), radii[0] + t[:, 0] * (radii[1] - radii[0])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))])
    cs = CubicSpline(s, ctrl, bc_type="natural")
    rs = CubicSpline(s, radii, bc_type="natural")
    # resample by true arc length
    fine = np.linspace(0, s[-1], max(int(s[-1] / (step / 4)), 8))
    pf = cs(fine)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    n = max(int(np.ceil(arc[-1] / step)), 1) + 1
    u = np.interp(np.linspace(0, arc[-1], n), arc, fine)
    return cs(u), np.clip(rs(u), 0.3, None)


class VesselModel:
    def __init__(self, data: dict, path: str | None = None, grid_h: float = 0.25):
        self.path = path
        self.name = data.get("name", Path(path).stem if path else "vessel")
        self.description = data.get("description", "")
        self.split = data.get("split", "train")          # train | val | test
        self.blend = float(data.get("blend_radius", 2.0))
        self.wall_thickness = float(data.get("wall_thickness", 1.5))
        self.vessels: list[Vessel] = []
        for v in data["vessels"]:
            pts, rad = smooth_centerline(np.array(v["control_points"], float), v["radius"])
            self.vessels.append(Vessel(v["name"], pts, rad, v.get("parent")))

        entry = data.get("entry", {})
        trunk = self.vessels[0]
        self.entry_pos = np.array(entry.get("position", trunk.points[0]), float)
        if "direction_deg" in entry:
            a = np.deg2rad(entry["direction_deg"])
            self.entry_dir = np.array([np.cos(a), np.sin(a)])
        else:
            d = trunk.points[min(4, len(trunk.points) - 1)] - trunk.points[0]
            self.entry_dir = d / np.linalg.norm(d)
        self.entry_angle = float(np.arctan2(self.entry_dir[1], self.entry_dir[0]))

        self.targets = self._make_targets(data.get("targets"))
        self._build_sdf(grid_h)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, **kw) -> "VesselModel":
        path = Path(path)
        return cls(json.loads(path.read_text()), str(path), **kw)

    def _make_targets(self, spec) -> list[Target]:
        if spec:
            return [Target(t["name"], np.array(t["position"], float), float(t.get("radius", 4.0)))
                    for t in spec]
        # Default: the distal end of every vessel that does not continue into
        # another vessel starting at that end.
        out = []
        starts = [v.points[0] for v in self.vessels]
        for v in self.vessels:
            end, r = v.points[-1], v.radii[-1]
            if any(np.linalg.norm(s - end) < 2 * r for s in starts):
                continue
            # pull the target slightly inside the closed end
            back = v.points[max(len(v.points) - 1 - int(r / 0.5), 0)]
            out.append(Target(v.name, back.copy(), max(3.0, r)))
        return out

    # ------------------------------------------------------------------
    def path_to(self, target: Target) -> np.ndarray:
        """Centreline polyline from the entry to `target` through the vessel tree."""
        by_name = {v.name: v for v in self.vessels}
        # vessel whose centreline passes closest to the target
        leaf = min(self.vessels, key=lambda v: np.linalg.norm(v.points - target.position, axis=1).min())
        chain = [leaf]
        while chain[-1].parent and chain[-1].parent in by_name:
            chain.append(by_name[chain[-1].parent])
        chain.reverse()
        pieces = []
        for k, v in enumerate(chain):
            i0 = 0 if k == 0 else int(np.argmin(np.linalg.norm(v.points - pieces[-1][-1], axis=1)))
            if k + 1 < len(chain):
                i1 = int(np.argmin(np.linalg.norm(v.points - chain[k + 1].points[0], axis=1)))
            else:
                i1 = int(np.argmin(np.linalg.norm(v.points - target.position, axis=1)))
            pieces.append(v.points[i0:i1 + 1] if i1 >= i0 else v.points[i0:i0 + 1])
        path = np.vstack(pieces)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6])
        return path[keep]

    # ------------------------------------------------------------------
    def _build_sdf(self, h: float) -> None:
        allp = np.concatenate([v.points for v in self.vessels])
        rmax = max(v.radii.max() for v in self.vessels)
        margin = rmax + 12.0
        lo = allp.min(0) - margin
        hi = allp.max(0) + margin
        nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
        ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
        self.origin, self.h, self.shape = lo, h, (ny, nx)
        xs = lo[0] + h * np.arange(nx)
        ys = lo[1] + h * np.arange(ny)

        band = 8.0
        big = np.float32(1e3)
        total = None
        for v in self.vessels:
            dv = np.full((ny, nx), big, np.float32)
            P, R = v.points[::2], v.radii[::2]          # 1 mm segments
            if len(v.points) % 2 == 0:
                P, R = np.vstack([P, v.points[-1]]), np.append(R, v.radii[-1])
            for a, b, ra, rb in zip(P[:-1], P[1:], R[:-1], R[1:]):
                ext = max(ra, rb) + band
                x0 = max(int((min(a[0], b[0]) - ext - lo[0]) / h), 0)
                x1 = min(int((max(a[0], b[0]) + ext - lo[0]) / h) + 2, nx)
                y0 = max(int((min(a[1], b[1]) - ext - lo[1]) / h), 0)
                y1 = min(int((max(a[1], b[1]) + ext - lo[1]) / h) + 2, ny)
                X, Y = np.meshgrid(xs[x0:x1], ys[y0:y1])
                ab = b - a
                t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab + 1e-12), 0, 1)
                d = np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])) - (ra + t * (rb - ra))
                np.minimum(dv[y0:y1, x0:x1], d, out=dv[y0:y1, x0:x1])
            total = dv if total is None else _smin(total, dv, self.blend)

        # Points far from every vessel: continue the distance field outward.
        far = total >= band
        if far.any():
            edt = distance_transform_edt(far) * h
            total = np.where(far, band + edt, total).astype(np.float32)
        self.sdf = total
        self.bounds = (lo, lo + h * np.array([nx - 1, ny - 1]))
        self._sdf64 = total.astype(np.float64)

    # ------------------------------------------------------------------
    def query(self, p: np.ndarray):
        """SDF value and its exact gradient at points p (N,2).

        Catmull-Rom bicubic interpolation: C1, so value and gradient are
        consistent (the Newton line search relies on that).
        """
        p = np.atleast_2d(p)
        ny, nx = self.shape
        fx = np.clip((p[:, 0] - self.origin[0]) / self.h, 1, nx - 3.0001)
        fy = np.clip((p[:, 1] - self.origin[1]) / self.h, 1, ny - 3.0001)
        j, i = fx.astype(int), fy.astype(int)
        wx, dwx = _catmull_rom(fx - j)
        wy, dwy = _catmull_rom(fy - i)
        off = np.arange(-1, 3)
        patch = self._sdf64[(i[:, None] + off)[:, :, None], (j[:, None] + off)[:, None, :]]  # (N,4,4)
        rowx = np.einsum("nab,nb->na", patch, wx)
        rowdx = np.einsum("nab,nb->na", patch, dwx)
        d = np.einsum("na,na->n", rowx, wy)
        gx = np.einsum("na,na->n", rowdx, wy) / self.h
        gy = np.einsum("na,na->n", rowx, dwy) / self.h
        return d, np.stack([gx, gy], axis=1)


def _catmull_rom(t: np.ndarray):
    """Weights (and their derivatives) for samples at offsets -1,0,1,2."""
    t2, t3 = t * t, t * t * t
    w = 0.5 * np.stack([-t + 2 * t2 - t3, 2 - 5 * t2 + 3 * t3, t + 4 * t2 - 3 * t3, -t2 + t3], 1)
    dw = 0.5 * np.stack([-1 + 4 * t - 3 * t2, -10 * t + 9 * t2, 1 + 8 * t - 9 * t2, -2 * t + 3 * t2], 1)
    return w, dw


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum: rounds the creases where two vessels join."""
    if k <= 0:
        return np.minimum(a, b)
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return (b * (1 - hh) + a * hh - k * hh * (1 - hh)).astype(np.float32)


def model_split(path: str | Path) -> str:
    """train | val | test, from the model file (default train)."""
    return json.loads(Path(path).read_text()).get("split", "train")


def list_models(folder: str | Path = "models", split: str | None = None) -> list[Path]:
    """Model files in `folder`, optionally only one split (comma-separated allowed)."""
    ms = sorted(Path(folder).glob("*.json"))
    if split:
        want = set(split.split(","))
        ms = [m for m in ms if model_split(m) in want]
    return ms
