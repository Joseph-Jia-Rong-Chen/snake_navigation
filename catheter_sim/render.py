"""Rendering: the plain fluoroscopy-style scene (used by the camera node) and the
viewer's overlays / HUD panel. The camera and the viewer share SceneRenderer, so
what the operator sees is pixel-identical to what is published and recorded."""
from __future__ import annotations

import numpy as np
import pygame

from .msgs import CameraInfo
from .vessel import VesselModel

PANEL_W = 330
BG = np.array([182, 182, 178])
LUMEN = np.array([118, 118, 122])
WALL = np.array([78, 78, 84])
RED = (225, 30, 30)
FILL = (150, 150, 150)


def fit_camera(vessel: VesselModel, width: int, height: int, margin: float = 6.0) -> CameraInfo:
    """Camera that frames the whole vessel model (uniform scale, centred)."""
    lo, hi = vessel.bounds
    lo, hi = lo + margin, hi - margin
    span = hi - lo
    scale = min(width / span[0], height / span[1])
    return CameraInfo(width=width, height=height, scale=float(scale), center_world=(lo + hi) / 2,
                      center_px=np.array([width / 2, height / 2]), model=vessel.name)


def surface_to_array(surf: pygame.Surface) -> np.ndarray:
    """(H, W, 3) uint8 RGB copy of a surface (single C-level copy)."""
    w, h = surf.get_size()
    return np.frombuffer(pygame.image.tobytes(surf, "RGB"), np.uint8).reshape(h, w, 3).copy()


def array_to_surface(arr: np.ndarray) -> pygame.Surface:
    """Surface from an (H, W, 3) RGB or (H, W) grayscale uint8 array."""
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    h, w = arr.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(), (w, h), "RGB")


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """(H, W) uint8 luma (ITU-R BT.601 weights 0.299/0.587/0.114, integer arithmetic)."""
    r, g, b = (rgb[..., k].astype(np.uint16) for k in range(3))
    return ((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8)


def resize_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB or grayscale uint8 image to size = (W, H) (area-style smoothing;
    image edges map onto image edges, like CameraInfo.scaled)."""
    if (img.shape[1], img.shape[0]) == tuple(size):
        return img
    out = surface_to_array(pygame.transform.smoothscale(array_to_surface(img), tuple(size)))
    return out[..., 0].copy() if img.ndim == 2 else out


class SceneRenderer:
    def __init__(self, vessel: VesselModel, cam: CameraInfo):
        self.vessel, self.cam = vessel, cam
        self.surface = pygame.Surface((cam.width, cam.height))
        self._bake()

    def w2s(self, p: np.ndarray) -> np.ndarray:
        return self.cam.world_to_px(p)

    def _bake(self) -> None:
        v = self.vessel
        sdf = v.sdf
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 4, sdf.shape)[..., None]
        ny, nx = sdf.shape                                   # vignette like an image intensifier
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot((xx - nx / 2) / nx, (yy - ny / 2) / ny)
        vig = (1 - 0.35 * np.clip(rr - 0.25, 0, None) * 2)[..., None]
        base = BG[None, None, :] * vig + noise
        lumen_w = np.clip(0.5 - sdf / 0.6, 0, 1)[..., None]
        wall_w = np.clip(1 - np.abs(sdf - v.wall_thickness / 2) / (v.wall_thickness / 2), 0, 1)[..., None]
        road = base * (1 - lumen_w) + (LUMEN[None, None] + noise) * lumen_w
        road = road * (1 - 0.6 * wall_w) + WALL[None, None] * 0.6 * wall_w
        self._bg_road = self._grid_to_surface(road)
        self._bg_plain = self._grid_to_surface(base)

    def _grid_to_surface(self, rgb: np.ndarray) -> pygame.Surface:
        arr = np.clip(rgb, 0, 255).astype(np.uint8)[::-1].transpose(1, 0, 2).copy()
        surf = pygame.surfarray.make_surface(arr)
        lo, hi = self.vessel.bounds
        tl = self.w2s(np.array([lo[0], hi[1]]))[0]
        br = self.w2s(np.array([hi[0], lo[1]]))[0]
        size = (int(round(br[0] - tl[0])), int(round(br[1] - tl[1])))
        self._bg_pos = (int(round(tl[0])), int(round(tl[1])))
        return pygame.transform.smoothscale(surf, size)

    def render(self, centerline: np.ndarray, diameter: float, roadmap: bool = True,
               surf: pygame.Surface | None = None, tip_color=RED) -> pygame.Surface:
        """Plain scene: background (+ vessel roadmap), black catheter, tip fiducial
        (red in the viewer; the camera passes a bright gray for mono8 images)."""
        s = surf or self.surface
        s.fill(FILL)
        s.blit(self._bg_road if roadmap else self._bg_plain, self._bg_pos)
        if centerline is None or len(centerline) < 2:
            return s
        sp = self.w2s(centerline)
        w = max(int(round(diameter * self.cam.scale)), 2)
        pygame.draw.lines(s, (0, 0, 0), False, sp, w)
        for q in sp[::2]:
            pygame.draw.circle(s, (0, 0, 0), q, w / 2)
        seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        from_tip = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        band = self.w2s(centerline[from_tip <= 2.0 + 1e-6])
        if len(band) >= 2:
            pygame.draw.lines(s, tip_color, False, band, w + 2)
        pygame.draw.circle(s, tip_color, sp[-1], w * 0.9 + 1)
        return s


# ---------------------------------------------------------------- viewer overlays
def feature_overlay(values: np.ndarray, mask: np.ndarray, alpha: int = 105) -> pygame.Surface:
    """Diverging colour map of a feature map (-1 blue .. 0 white .. +1 red), only
    inside the vessel; transparent outside."""
    v = np.clip(values, -1, 1)[..., None]
    white = np.array([245, 245, 245], float)
    rgb = np.where(v >= 0, white + v * (np.array([215, 40, 40]) - white),
                   white - v * (np.array([40, 90, 220]) - white))
    surf = pygame.Surface((values.shape[1], values.shape[0]), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, rgb.astype(np.uint8).transpose(1, 0, 2))
    a = pygame.surfarray.pixels_alpha(surf)
    a[:] = (mask.T > 0) * alpha
    del a
    return surf


def draw_targets(s, sr: SceneRenderer, font, targets, active_name: str | None, labels: bool) -> None:
    for t in targets:
        pc = sr.w2s(t.position)[0]
        active = t.name == active_name
        col = (40, 200, 90) if active else (90, 140, 105)
        pygame.draw.circle(s, col, pc, max(t.radius * sr.cam.scale, 4), 3 if active else 1)
        if active or labels:
            s.blit(font.render(t.name, True, col), pc + np.array([8, -20]))


def draw_entry(s, sr: SceneRenderer) -> None:
    v = sr.vessel
    e = sr.w2s(v.entry_pos)[0]
    d = v.entry_dir * np.array([1, -1])
    n = np.array([-d[1], d[0]])
    pygame.draw.polygon(s, (60, 60, 200), [e - d * 14 + n * 8, e - d * 14 - n * 8, e - d * 2])


def draw_contacts(s, sr: SceneRenderer, contacts) -> None:
    for x, f, ff in zip(*contacts):
        a = sr.w2s(x)[0]
        pygame.draw.line(s, (250, 210, 0), a, a + np.array([f[0], -f[1]]) * 200, 2)
        pygame.draw.line(s, (0, 200, 250), a, a + np.array([ff[0], -ff[1]]) * 2000, 2)


class Panel:
    """Side panel: title, status lines, bend gauge, banner, help."""

    def __init__(self, title: str = "Catheter"):
        self.title = title
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 15)
        self.big = pygame.font.SysFont("dejavusans,sans", 20, bold=True)
        self.small = pygame.font.SysFont("dejavusansmono,monospace", 12)

    def draw(self, s: pygame.Surface, rect: pygame.Rect, lines, bend_cmd: float, bend_joint: float,
             bend_meas: float, bend_max: float, recording: bool, banner: str | None,
             help_lines: list[str] | None) -> None:
        s.fill((28, 30, 34), rect)
        x, y = rect.x + 16, 14
        s.blit(self.big.render(self.title, True, (235, 235, 235)), (x, y))
        y += 34
        if recording:
            pygame.draw.circle(s, (230, 40, 40), (x + 8, y + 9), 8)
            s.blit(self.big.render("REC", True, (230, 40, 40)), (x + 22, y - 2))
        y += 30
        for text, col in lines:
            s.blit(self.font.render(text, True, col), (x, y))
            y += 20
        y += 8
        s.blit(self.font.render("tip bend   CCW  <->  CW", True, (200, 200, 200)), (x, y))
        y += 22
        gw = rect.w - 40
        pygame.draw.rect(s, (70, 70, 75), (x, y, gw, 14), border_radius=4)
        cx = x + gw / 2
        pygame.draw.line(s, (160, 160, 160), (cx, y - 3), (cx, y + 17), 1)
        f = np.clip(bend_joint / bend_max, -1, 1)
        pygame.draw.rect(s, (230, 120, 40), (min(cx, cx + f * gw / 2), y, abs(f) * gw / 2, 14), border_radius=4)
        m = cx + np.clip(bend_meas / bend_max, -1.2, 1.2) * gw / 2
        pygame.draw.line(s, (80, 200, 255), (m, y - 4), (m, y + 18), 3)
        c = cx + np.clip(bend_cmd / bend_max, -1, 1) * gw / 2
        pygame.draw.polygon(s, (245, 245, 245), [(c - 6, y - 9), (c + 6, y - 9), (c, y - 1)])
        y += 26
        s.blit(self.small.render("white=target  orange=joint  blue=meas", True, (140, 140, 140)), (x, y))
        y += 30
        if banner:
            s.blit(self.big.render(banner, True, (60, 220, 110)), (x, y))
            y += 34
        if help_lines:
            y = max(y, rect.h - 18 * len(help_lines) - 12)
            for t in help_lines:
                s.blit(self.small.render(t, True, (150, 150, 160)), (x, y))
                y += 18
