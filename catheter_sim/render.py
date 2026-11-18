"""Pygame renderer: fluoroscopy-style view + HUD."""
from __future__ import annotations

import numpy as np
import pygame

from .vessel import VesselModel

PANEL_W = 320
BG = np.array([182, 182, 178])
LUMEN = np.array([118, 118, 122])
WALL = np.array([78, 78, 84])
RED = (225, 30, 30)


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 15)
        self.big = pygame.font.SysFont("dejavusans,sans", 20, bold=True)
        self.show_roadmap = True
        self.show_contacts = False
        self.show_help = True
        self.vessel: VesselModel | None = None

    # ------------------------------------------------------------ geometry
    def set_vessel(self, vessel: VesselModel) -> None:
        self.vessel = vessel
        W, H = self.screen.get_size()
        self.view = pygame.Rect(0, 0, W - PANEL_W, H)
        lo, hi = vessel.bounds
        margin = 6.0
        lo, hi = lo + margin, hi - margin
        span = hi - lo
        self.scale = min(self.view.w / span[0], self.view.h / span[1])
        c = (lo + hi) / 2
        self.center_world = c
        self.center_px = np.array(self.view.center, float)
        self._bake_backgrounds()

    def w2s(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        d = (p - self.center_world) * self.scale
        return np.stack([self.center_px[0] + d[:, 0], self.center_px[1] - d[:, 1]], 1)

    def _bake_backgrounds(self) -> None:
        v = self.vessel
        sdf = v.sdf
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 4, sdf.shape)[..., None]
        # vignette like an image intensifier
        ny, nx = sdf.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot((xx - nx / 2) / nx, (yy - ny / 2) / ny)
        vig = (1 - 0.35 * np.clip(rr - 0.25, 0, None) * 2)[..., None]
        base = BG[None, None, :] * vig + noise
        lumen_w = np.clip(0.5 - sdf / 0.6, 0, 1)[..., None]           # anti-aliased
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

    # ------------------------------------------------------------- drawing
    def draw_scene(self, centerline: np.ndarray, diameter: float, surf: pygame.Surface | None = None,
                   overlays: bool = True, target=None, contacts=None) -> None:
        """Draw background, optional overlays and a catheter given its centreline
        (base->tip, world mm). `contacts` = (points, normal_forces, friction_forces)."""
        s = surf or self.screen
        s.fill((150, 150, 150), self.view)
        s.blit(self._bg_road if self.show_roadmap else self._bg_plain, self._bg_pos)
        v = self.vessel

        if overlays:
            for t in v.targets:
                pc = self.w2s(t.position)[0]
                active = target is not None and t is target
                col = (40, 200, 90) if active else (90, 140, 105)
                pygame.draw.circle(s, col, pc, max(t.radius * self.scale, 4), 3 if active else 1)
                if active or self.show_help:
                    s.blit(self.font.render(t.name, True, col), pc + np.array([8, -20]))
            e = self.w2s(v.entry_pos)[0]
            d = v.entry_dir * np.array([1, -1])
            n = np.array([-d[1], d[0]])
            pygame.draw.polygon(s, (60, 60, 200), [e - d * 14 + n * 8, e - d * 14 - n * 8, e - d * 2])

        # catheter body (black, radiopaque)
        sp = self.w2s(centerline)
        w = max(int(round(diameter * self.scale)), 2)
        if len(sp) >= 2:
            pygame.draw.lines(s, (0, 0, 0), False, sp, w)
            for q in sp[::2]:
                pygame.draw.circle(s, (0, 0, 0), q, w / 2)
        # tip fiducial: red marker over the distal 2 mm + dot at the tip
        seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        from_tip = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        band = self.w2s(centerline[from_tip <= 2.0 + 1e-6])
        if len(band) >= 2:
            pygame.draw.lines(s, RED, False, band, w + 2)
        pygame.draw.circle(s, RED, sp[-1], w * 0.9 + 1)

        if overlays and self.show_contacts and contacts is not None:
            for x, f, ff in zip(*contacts):
                a = self.w2s(x)[0]
                pygame.draw.line(s, (250, 210, 0), a, a + np.array([f[0], -f[1]]) * 200, 2)
                pygame.draw.line(s, (0, 200, 250), a, a + np.array([ff[0], -ff[1]]) * 2000, 2)

    def capture(self, centerline: np.ndarray, diameter: float, size: int) -> np.ndarray:
        """Clean fluoroscopy frame (no overlays) of the view, resized to size x size, RGB uint8."""
        tmp = pygame.Surface(self.screen.get_size())
        self.draw_scene(centerline, diameter, tmp, overlays=False)
        sub = tmp.subsurface(self.view)
        img = pygame.transform.smoothscale(sub, (size, size))
        return pygame.surfarray.array3d(img).transpose(1, 0, 2).copy()

    def draw_panel(self, lines: list[tuple[str, tuple]], kappa: float, kappa_max: float,
                   kappa_meas: float, recording: bool, banner: str | None) -> None:
        s = self.screen
        W, H = s.get_size()
        panel = pygame.Rect(W - PANEL_W, 0, PANEL_W, H)
        s.fill((28, 30, 34), panel)
        x, y = panel.x + 16, 14
        s.blit(self.big.render("Catheter Teleop", True, (235, 235, 235)), (x, y))
        y += 34
        if recording:
            pygame.draw.circle(s, (230, 40, 40), (x + 8, y + 9), 8)
            s.blit(self.big.render("REC", True, (230, 40, 40)), (x + 22, y - 2))
        y += 30
        for text, col in lines:
            s.blit(self.font.render(text, True, col), (x, y))
            y += 20
        # curvature gauge
        y += 8
        s.blit(self.font.render("tip curvature  CCW  <->  CW", True, (200, 200, 200)), (x, y))
        y += 22
        gw = PANEL_W - 40
        pygame.draw.rect(s, (70, 70, 75), (x, y, gw, 14), border_radius=4)
        cx = x + gw / 2
        pygame.draw.line(s, (160, 160, 160), (cx, y - 3), (cx, y + 17), 1)
        f = kappa / kappa_max
        pygame.draw.rect(s, (230, 120, 40), (min(cx, cx + f * gw / 2), y, abs(f) * gw / 2, 14), border_radius=4)
        m = cx + np.clip(kappa_meas / kappa_max, -1.2, 1.2) * gw / 2
        pygame.draw.line(s, (80, 200, 255), (m, y - 4), (m, y + 18), 3)
        y += 26
        s.blit(self.font.render("orange=command  blue=measured", True, (140, 140, 140)), (x, y))
        y += 30
        if banner:
            s.blit(self.big.render(banner, True, (60, 220, 110)), (x, y))
            y += 34
        if self.show_help:
            help_lines = [
                "W / S      insert / retract",
                "<- / ->    curve CCW / CW",
                "Shift      fast",
                "Space      straighten tip",
                "Enter      start/stop recording",
                "R          reset episode",
                "N / P      next / prev vessel",
                "T          next target",
                "V          toggle roadmap",
                "C          contact forces",
                "H          hide help",
                "Esc        quit",
            ]
            y = max(y, H - 20 * len(help_lines) - 16)
            for t in help_lines:
                s.blit(self.font.render(t, True, (150, 150, 160)), (x, y))
                y += 20
