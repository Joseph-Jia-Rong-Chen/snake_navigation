#!/usr/bin/env python
"""Replay a recorded episode.

  python replay.py data/vessel_02/<episode>.npz            # view (redrawn from logged shapes)
  python replay.py data/vessel_02/<episode>.npz --resim    # re-run actuator + physics from the
                                                           # logged targets, compare with the log
Keys: Space pause, Left/Right step, +/- speed, F feature map U/V/off, V roadmap, Esc quit.
The raw camera stream of the episode is <episode>.mp4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import catheter_sim  # noqa: F401
import numpy as np

from catheter_sim.config import Config
from catheter_sim.env import CatheterEnv
from catheter_sim.msgs import CameraInfo
from catheter_sim.recorder import init_state, load_episode
from catheter_sim.vessel import VesselModel

ROOT = Path(__file__).resolve().parent


def build_cfg(meta: dict) -> Config:
    cfg = Config()
    cfg.apply_dict(meta["config"], strict=False)
    return cfg


def model_path(meta: dict) -> Path:
    p = Path(meta["model_path"])
    return p if p.exists() else ROOT / "models" / f"{meta['model']}.json"


def resim(meta: dict, d: dict) -> None:
    env = CatheterEnv(build_cfg(meta), model_path(meta), dt=meta["sim_dt"])
    env.set_state(init_state(d))
    joint_err, tip_err = [], []
    for k in range(len(d["ctrl_t"])):
        o = env.command(float(d["ctrl_target_length"][k]), float(d["ctrl_target_bend"][k]))
        joint_err.append(max(abs(o["joint_length"] - d["ctrl_joint_length"][k]),
                             abs(o["joint_bend"] - d["ctrl_joint_bend"][k])))
        tip_err.append(np.linalg.norm(o["tip_pos"] - d["ctrl_tip_pos"][k]))
    print(f"re-simulated {len(tip_err)} control ticks: max joint error {max(joint_err):.2e}, "
          f"tip error mean {np.mean(tip_err):.2e} mm, max {max(tip_err):.2e} mm")


def view(meta: dict, d: dict, name: str) -> None:
    import pygame

    from catheter_sim.render import PANEL_W, Panel, SceneRenderer, draw_targets, feature_overlay
    cfg = build_cfg(meta)
    cam = CameraInfo.from_dict(meta["camera_info"])
    pygame.init()
    screen = pygame.display.set_mode((cam.width + PANEL_W, cam.height))
    pygame.display.set_caption(f"Replay - {name}")
    vessel = VesselModel.load(model_path(meta))
    sr = SceneRenderer(vessel, cam)
    panel, font = Panel("Replay"), pygame.font.SysFont("dejavusansmono,monospace", 15)
    bend_max = float(np.degrees(cfg.catheter.kappa_max * cfg.catheter.tip_length))
    clock = pygame.time.Clock()
    i, T, paused, speed, fmode, roadmap, overlay = 0, len(d["t"]), False, 1, 0, True, None
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RIGHT:
                    i = min(i + 1, T - 1)
                elif k == pygame.K_LEFT:
                    i = max(i - 1, 0)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed * 2, 16)
                elif k == pygame.K_MINUS:
                    speed = max(speed // 2, 1)
                elif k == pygame.K_v:
                    roadmap = not roadmap
                elif k == pygame.K_f:
                    fmode, overlay = (fmode + 1) % 3, None
        screen.blit(sr.render(d["shape"][i], cfg.catheter.diameter, roadmap), (0, 0))
        mi = int(d["feature_map_index"][i])
        if fmode and mi >= 0:
            if overlay is None or overlay[0] != (mi, fmode):
                vals = d["feature_U" if fmode == 1 else "feature_V"][mi]
                overlay = ((mi, fmode), feature_overlay(vals, d["vessel_mask"][mi]))
            screen.blit(overlay[1], (0, 0))
        tname = str(d["feature_target_name"][mi]) if mi >= 0 else None
        draw_targets(screen, sr, font, vessel.targets, tname, False)
        keys = "".join(n if b else "." for n, b in zip("WSLR", d["keys"][i]))
        lines = [
            (f"frame {i + 1}/{T}   t={d['t'][i] - d['t'][0]:.2f}s   x{speed}", (230, 230, 230)),
            (f"source {d['action_source'][i]}   keys {keys}", (160, 160, 165)),
            ("              target    joint", (160, 160, 165)),
            (f"insert [mm] {d['action_length'][i]:7.1f}  {d['joint_length'][i]:7.1f}", (230, 230, 230)),
            (f"bend  [deg] {d['action_bend'][i]:+7.1f}  {d['joint_bend'][i]:+7.1f}", (230, 230, 230)),
            (f"target {tname}  {d['target_dist'][i]:.1f} mm", (90, 220, 120)),
        ]
        panel.draw(screen, pygame.Rect(cam.width, 0, PANEL_W, cam.height), lines,
                   float(d["action_bend"][i]), float(d["joint_bend"][i]), float(d["bend_meas"][i]),
                   bend_max, False, "TARGET REACHED" if d["success"][i] else None, None)
        pygame.display.flip()
        if not paused:
            i = min(i + speed, T - 1)
        clock.tick(meta["camera_fps"])
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--resim", action="store_true")
    args = ap.parse_args()
    meta, d = load_episode(args.episode)
    if "ctrl_t" not in d:
        raise SystemExit("this recording uses an older format; replay supports the launch.py format")
    print({k: meta.get(k) for k in ("model", "target", "n_frames", "duration", "reached_target",
                                    "time_to_target", "sources")})
    resim(meta, d) if args.resim else view(meta, d, Path(args.episode).name)


if __name__ == "__main__":
    main()
