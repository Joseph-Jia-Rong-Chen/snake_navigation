#!/usr/bin/env python
"""Replay a raw recorded episode.

  python replay.py data/raw/vessel_02/<episode>            # view (redrawn from the logged shapes)
  python replay.py data/raw/vessel_02/<episode> --resim    # re-run actuator + physics from the
                                                           # logged targets, compare with the log
  python replay.py data/raw/vessel_02/<episode> --sync     # print the alignment / timing report
Keys: Space pause, Left/Right step, +/- speed, F feature map U/V/off, I recorded video frame,
V roadmap, Esc quit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import catheter_sim  # noqa: F401
import numpy as np

from catheter_sim.config import Config
from catheter_sim.dataset import Episode
from catheter_sim.env import CatheterEnv
from catheter_sim.recorder import init_state


def build_cfg(meta: dict) -> Config:
    cfg = Config()
    cfg.apply_dict(meta["config"], strict=False)
    return cfg


def resim(ep: Episode) -> None:
    """Feed the target the actuator tracked on every logged tick back into a fresh
    sim started from the logged initial state; joints and tip must match exactly."""
    d, meta = ep.d, ep.meta
    env = CatheterEnv(build_cfg(meta), ep.vessel, dt=meta["sim_dt"])
    env.set_state(init_state(d))
    joint_err, tip_err = [], []
    for k in range(len(d["state_t"])):
        o = env.command(float(d["state_target_length"][k]), float(d["state_target_bend"][k]))
        joint_err.append(max(abs(o["joint_length"] - d["state_length"][k]),
                             abs(o["joint_bend"] - d["state_bend"][k])))
        tip_err.append(np.linalg.norm(o["tip_pos"] - d["gt_tip_pos"][k]))
    print(f"re-simulated {len(tip_err)} control ticks: max joint error {max(joint_err):.2e}, "
          f"tip error mean {np.mean(tip_err):.2e} mm, max {max(tip_err):.2e} mm")


def view(ep: Episode) -> None:
    import pygame

    from catheter_sim.featuremap import compute
    from catheter_sim.render import (PANEL_W, Panel, SceneRenderer, array_to_surface, draw_targets,
                                     feature_overlay, resize_image)
    meta, d, a = ep.meta, ep.d, ep.a
    cfg = build_cfg(meta)
    cam = ep.cam
    pygame.init()
    screen = pygame.display.set_mode((cam.width + PANEL_W, cam.height))
    pygame.display.set_caption(f"Replay - {ep.dir.name}")
    sr = SceneRenderer(ep.vessel, cam)
    panel, font = Panel("Replay"), pygame.font.SysFont("dejavusansmono,monospace", 15)
    bend_max = float(np.degrees(cfg.catheter.kappa_max * cfg.catheter.tip_length))
    video = None
    clock = pygame.time.Clock()
    i, T, paused, speed, fmode, roadmap, show_video, overlay = 0, len(ep), False, 1, 0, True, False, None
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
                elif k == pygame.K_i:
                    show_video = not show_video
                    if show_video and video is None:
                        video = ep.frames()                     # at the recorded video size
                elif k == pygame.K_f:
                    fmode, overlay = (fmode + 1) % 3, None
        gi = int(a["gt_index"][i])
        if show_video:
            screen.blit(array_to_surface(resize_image(video[i], (cam.width, cam.height))), (0, 0))
        else:
            screen.blit(sr.render(d["gt_shape"][gi], cfg.catheter.diameter, roadmap), (0, 0))
        mi = int(a["map_index"][i])
        if fmode:
            if overlay is None or overlay[0] != (mi, fmode):
                U, V, mask, *_ = compute(ep.vessel, cam, d["fmap_target_world"][mi])
                overlay = ((mi, fmode), feature_overlay(U if fmode == 1 else V, mask))
            screen.blit(overlay[1], (0, 0))
        tname = ep.target_name(i)
        draw_targets(screen, sr, font, ep.vessel.targets, tname, False)
        lines = [
            (f"frame {i + 1}/{T}   t={a['t'][i]:.2f}s   x{speed}", (230, 230, 230)),
            (f"source {a['action_source'][i]}" + ("   [video]" if show_video else ""), (160, 160, 165)),
            ("              action   state", (160, 160, 165)),
            (f"insert [mm] {a['action'][i, 0]:7.1f}  {a['state'][i, 0]:7.1f}", (230, 230, 230)),
            (f"bend  [deg] {a['action'][i, 1]:+7.1f}  {a['state'][i, 1]:+7.1f}", (230, 230, 230)),
            (f"target {tname}  {a['target_dist'][i]:.1f} mm", (90, 220, 120)),
        ]
        panel.draw(screen, pygame.Rect(cam.width, 0, PANEL_W, cam.height), lines,
                   float(a["action"][i, 1]), float(a["state"][i, 1]), float(d["gt_bend_meas"][gi]),
                   bend_max, False, "TARGET REACHED" if a["success"][i] else None, None)
        pygame.display.flip()
        if not paused:
            i = min(i + speed, T - 1)
        clock.tick(meta["camera_fps"])
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="raw episode directory (or its meta.json)")
    ap.add_argument("--resim", action="store_true")
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    p = Path(args.episode)
    if p.suffix == ".npz" and not (p.parent / "meta.json").exists():
        raise SystemExit("this is an older single-file recording; replay supports the raw episode directories")
    ep = Episode(p)
    print({k: ep.meta.get(k) for k in ("model", "target", "n_frames", "duration", "reached_target",
                                       "time_to_target", "sources", "video_size")})
    if args.sync:
        print(json.dumps(ep.sync_report(), indent=2))
    elif args.resim:
        resim(ep)
    else:
        view(ep)


if __name__ == "__main__":
    main()
