#!/usr/bin/env python
"""Replay a recorded episode.

  python replay.py data/vessel_02/20261117_120000_left_up.npz
  python replay.py <episode.npz> --resim      # re-run the physics from the logged actions
                                              # and report the deviation from the log

Keys: Space pause, Left/Right step (when paused), +/- speed, V roadmap, Esc quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import catheter_sim  # noqa: F401
import numpy as np
import pygame

from catheter_sim.config import Config
from catheter_sim.env import CatheterEnv
from catheter_sim.recorder import init_state, load_episode
from catheter_sim.render import Renderer
from catheter_sim.vessel import VesselModel

ROOT = Path(__file__).resolve().parent


def build_cfg(meta: dict) -> Config:
    cfg = Config()
    cfg.apply_dict(meta["config"])
    return cfg


def model_path(meta: dict) -> Path:
    p = Path(meta["model_path"])
    return p if p.exists() else ROOT / "models" / f"{meta['model']}.json"


def resim(meta: dict, d: dict) -> None:
    env = CatheterEnv(build_cfg(meta), model_path(meta))
    env.reset()
    st = init_state(d)
    if st is not None:
        env.cath.set_full_state(st)
    err = []
    for a, tip in zip(d["action"], d["tip_pos"]):
        o = env.step(float(a[0]), float(a[1]))
        err.append(np.linalg.norm(o["tip_pos"] - tip))
    err = np.array(err)
    print(f"re-simulated {len(err)} frames: tip error mean {err.mean():.4f} mm, max {err.max():.4f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--resim", action="store_true")
    args = ap.parse_args()
    meta, d = load_episode(args.episode)
    print({k: meta[k] for k in ("model", "target", "n_frames", "duration", "reached_target", "time_to_target")})
    if args.resim:
        resim(meta, d)
        return

    cfg = build_cfg(meta)
    pygame.init()
    screen = pygame.display.set_mode((cfg.teleop.window_width, cfg.teleop.window_height))
    pygame.display.set_caption(f"Replay - {Path(args.episode).name}")
    r = Renderer(screen)
    vessel = VesselModel.load(model_path(meta))
    r.set_vessel(vessel)
    r.show_help = False
    target = next((t for t in vessel.targets if t.name == meta.get("target")), None)
    clock = pygame.time.Clock()
    i, T, paused, speed = 0, len(d["t"]), False, 1
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    i = min(i + 1, T - 1)
                elif ev.key == pygame.K_LEFT:
                    i = max(i - 1, 0)
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed * 2, 16)
                elif ev.key == pygame.K_MINUS:
                    speed = max(speed // 2, 1)
                elif ev.key == pygame.K_v:
                    r.show_roadmap = not r.show_roadmap
        r.draw_scene(d["shape"][i], cfg.catheter.diameter, target=target)
        k = d["keys"][i]
        lines = [
            (f"frame {i + 1}/{T}   t={d['t'][i]:.2f}s   x{speed}", (230, 230, 230)),
            (f"insertion  {d['length'][i]:7.1f} mm", (230, 230, 230)),
            (f"kappa cmd  {d['kappa_cmd'][i]:+.4f}", (230, 230, 230)),
            ("keys  " + " ".join(n if b else "." for n, b in zip("WSLR", k)), (160, 160, 165)),
            (f"target dist {d['target_dist'][i]:.1f} mm", (90, 220, 120)),
        ]
        r.draw_panel(lines, float(d["kappa_cmd"][i]), cfg.catheter.kappa_max, float(d["kappa_meas"][i]),
                     False, "TARGET REACHED" if d["success"][i] else None)
        pygame.display.flip()
        if not paused:
            i = min(i + speed, T - 1)
        clock.tick(round(1 / meta["dt"]))
    pygame.quit()


if __name__ == "__main__":
    main()
