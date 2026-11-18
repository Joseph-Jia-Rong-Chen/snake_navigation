#!/usr/bin/env python
"""Keyboard teleoperation + data collection for the 2D catheter simulator.

  W / S        insert / retract              (insertion length, + = deeper)
  Left / Right curve tip CCW / CW            (tip curvature, - = CCW, + = CW)
  Shift        fast;  Space  straighten tip
  Enter        start / stop recording        R  reset (saves a running recording)
  N / P        next / previous vessel model  T  next target
  V roadmap    C contact forces    H help    Esc quit

Example:
  python teleop.py --model vessel_01 --set catheter.EI_shaft=30 --set catheter.friction=0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import catheter_sim  # noqa: F401  (sets BLAS threads before numpy)
import numpy as np
import pygame

from catheter_sim.config import load_config
from catheter_sim.env import CatheterEnv
from catheter_sim.recorder import Recorder
from catheter_sim.render import Renderer
from catheter_sim.vessel import list_models

ROOT = Path(__file__).resolve().parent


def resolve_model(models: list[Path], spec: str | None) -> int:
    if spec is None:
        return 0
    if spec.isdigit() and int(spec) < len(models):
        return int(spec)
    for i, m in enumerate(models):
        if spec in (m.name, m.stem) or m.stem.endswith(spec) or m.stem.startswith(spec):
            return i
    sys.exit(f"model '{spec}' not found in {[m.stem for m in models]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model name/stem/index in --models-dir")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a parameter, e.g. catheter.tip_length=20")
    ap.add_argument("--record-images", action="store_true", help="also store fluoroscopy frames")
    ap.add_argument("--autorecord", action="store_true", help="start recording on every reset")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.record_images:
        cfg.teleop.record_images = True
    tp = cfg.teleop
    models = list_models(args.models_dir)
    if not models:
        sys.exit(f"no *.json models in {args.models_dir}")
    mi = resolve_model(models, args.model)

    pygame.init()
    screen = pygame.display.set_mode((tp.window_width, tp.window_height))
    clock = pygame.time.Clock()
    renderer = Renderer(screen)
    env = CatheterEnv(cfg, models[mi])
    renderer.set_vessel(env.vessel)
    data_dir = Path(tp.data_dir) if Path(tp.data_dir).is_absolute() else ROOT / tp.data_dir
    rec = Recorder(data_dir)
    dt = 1.0 / tp.fps
    status = ""
    status_t = 0.0

    def set_title():
        pygame.display.set_caption(f"Catheter teleop - {env.vessel.name}")

    def start_recording():
        tgt = env.target
        rec.start({
            "model": env.vessel.name,
            "model_path": str(env.model_path),
            "target": tgt.name if tgt else None,
            "target_pos": tgt.position.tolist() if tgt else None,
            "target_radius": tgt.radius if tgt else None,
            "dt": dt,
            "config": cfg.to_dict(),
            "image_size": tp.image_size if tp.record_images else None,
        }, init_state=env.cath.get_state())

    def stop_recording() -> str:
        path = rec.stop()
        return f"saved {path.relative_to(ROOT) if path and path.is_relative_to(ROOT) else path}" if path else "nothing recorded"

    def reset():
        nonlocal status, status_t
        if rec.active:
            status, status_t = stop_recording(), 3.0
        env.reset()
        if args.autorecord:
            start_recording()

    set_title()
    reset()
    obs = env.observe()
    running = True
    while running:
        straighten = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_RETURN:
                    if rec.active:
                        status, status_t = stop_recording(), 3.0
                    else:
                        start_recording()
                        status, status_t = "recording...", 1.5
                elif k == pygame.K_r:
                    reset()
                elif k in (pygame.K_n, pygame.K_p):
                    if rec.active:
                        status, status_t = stop_recording(), 3.0
                    mi = (mi + (1 if k == pygame.K_n else -1)) % len(models)
                    env.load_model(models[mi])
                    renderer.set_vessel(env.vessel)
                    set_title()
                    reset()
                elif k == pygame.K_t and env.vessel.targets:
                    env.target_idx = (env.target_idx + 1) % len(env.vessel.targets)
                    if rec.active:
                        rec.meta.update(target=env.target.name, target_pos=env.target.position.tolist())
                elif k == pygame.K_SPACE:
                    straighten = True
                elif k == pygame.K_v:
                    renderer.show_roadmap = not renderer.show_roadmap
                elif k == pygame.K_c:
                    renderer.show_contacts = not renderer.show_contacts
                elif k == pygame.K_h:
                    renderer.show_help = not renderer.show_help

        pressed = pygame.key.get_pressed()
        keys = [pressed[pygame.K_w], pressed[pygame.K_s], pressed[pygame.K_LEFT], pressed[pygame.K_RIGHT]]
        fast = pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT]
        mult = tp.fast_multiplier if fast else 1.0
        d_len = (keys[0] - keys[1]) * tp.insert_speed * mult * dt
        d_kap = (keys[3] - keys[2]) * tp.kappa_rate * mult * dt   # Right = CW = +
        if straighten:
            d_kap = -env.cath.kappa
        obs = env.step(d_len, d_kap, dt)

        cl = env.cath.spline(step=0.5)
        if rec.active:
            img = renderer.capture(cl, cfg.catheter.diameter, tp.image_size) if tp.record_images else None
            rec.add(obs, keys, fast, (d_len, d_kap), img)

        ci = env.cath.last_contact
        renderer.draw_scene(cl, cfg.catheter.diameter, target=env.target,
                            contacts=(ci.points[ci.active], ci.force[ci.active], ci.friction[ci.active]))
        tgt = env.target
        white, grey = (230, 230, 230), (160, 160, 165)
        lines = [
            (f"model   {env.vessel.name}", white),
            (f"target  {tgt.name if tgt else '-'}  ({obs['target_dist']:.1f} mm)", (90, 220, 120)),
            ("", white),
            (f"insertion   {obs['length']:7.1f} mm", white),
            (f"kappa cmd   {obs['kappa_cmd']:+7.4f} 1/mm", white),
            (f"kappa meas  {obs['kappa_meas']:+7.4f} 1/mm", grey),
            (f"tip         ({obs['tip_pos'][0]:6.1f}, {obs['tip_pos'][1]:6.1f})", grey),
            (f"tip angle   {np.degrees(obs['tip_angle']):7.1f} deg", grey),
            (f"contacts    {obs['n_contacts']:4d}   sum|F| {obs['contact_force_total']:.3f} N", grey),
            (f"insert force {obs['insertion_force']:+.3f} N", grey),
            (f"fps {clock.get_fps():4.0f}   newton it {env.cath.solver_iters}", (110, 110, 115)),
        ]
        if rec.active:
            lines.append((f"frames {len(rec.frames)}", (230, 80, 80)))
        if status_t > 0:
            lines.append((status, (240, 200, 90)))
            status_t -= dt
        banner = "TARGET REACHED" if obs["success"] else None
        renderer.draw_panel(lines, env.cath.kappa, cfg.catheter.kappa_max, obs["kappa_meas"], rec.active, banner)
        pygame.display.flip()
        clock.tick(tp.fps)

    if rec.active:
        print(stop_recording())
    pygame.quit()


if __name__ == "__main__":
    main()
