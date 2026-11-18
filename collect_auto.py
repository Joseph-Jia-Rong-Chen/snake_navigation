#!/usr/bin/env python
"""Generate episodes with the scripted demonstrator (same format as teleop).

  python collect_auto.py                       # 1 episode per target, all models
  python collect_auto.py --model 02_arch --episodes 5 --noise 0.3 --record-images

`--noise` randomises the controller gains and randomly drops/flips key presses,
so repeated episodes differ. Episodes go to data/<model>/ with meta.source="autopilot".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import catheter_sim  # noqa: F401
import numpy as np

from catheter_sim.autopilot import Autopilot
from catheter_sim.config import load_config
from catheter_sim.env import CatheterEnv
from catheter_sim.recorder import Recorder
from catheter_sim.vessel import list_models

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="restrict to one model (stem substring)")
    ap.add_argument("--target", help="restrict to one target name")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per target")
    ap.add_argument("--noise", type=float, default=0.0, help="0..1 action/gain randomisation")
    ap.add_argument("--max-time", type=float, default=90.0, help="seconds of sim time per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--record-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tp = cfg.teleop
    dt = 1.0 / tp.fps
    rng = np.random.default_rng(args.seed)
    rec = Recorder(ROOT / tp.data_dir)
    renderer = None
    if args.record_images:
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        from catheter_sim.render import Renderer
        pygame.init()
        renderer = Renderer(pygame.display.set_mode((tp.window_width, tp.window_height)))

    models = [m for m in list_models(ROOT / "models") if not args.model or args.model in m.stem]
    for mp in models:
        env = CatheterEnv(cfg, mp)
        if renderer:
            renderer.set_vessel(env.vessel)
        for ti, tgt in enumerate(env.vessel.targets):
            if args.target and tgt.name != args.target:
                continue
            for ep in range(args.episodes):
                env.target_idx = ti
                obs = env.reset()
                n = args.noise
                pilot = Autopilot(cfg, env.vessel.path_to(tgt),
                                  lookahead=6.0 * (1 + n * rng.uniform(-0.4, 0.4)),
                                  gain=0.35 * (1 + n * rng.uniform(-0.4, 0.4)))
                rec.start({"model": env.vessel.name, "model_path": str(mp), "target": tgt.name,
                           "target_pos": tgt.position.tolist(), "target_radius": tgt.radius,
                           "dt": dt, "config": cfg.to_dict(), "source": "autopilot",
                           "noise": n, "image_size": tp.image_size if renderer else None},
                          init_state=env.cath.get_state())
                for _ in range(int(args.max_time / dt)):
                    keys = pilot.act(obs, dt)
                    if n > 0 and rng.random() < 0.15 * n:          # imperfect operator
                        keys = [k if rng.random() > 0.5 else not k for k in keys]
                        keys[0], keys[1] = keys[0] and not keys[1], keys[1] and not keys[0]
                    a = ((keys[0] - keys[1]) * tp.insert_speed * dt,
                         (keys[3] - keys[2]) * tp.kappa_rate * dt)
                    obs = env.step(*a, dt)
                    img = renderer.capture(env.cath.spline(), cfg.catheter.diameter, tp.image_size) if renderer else None
                    rec.add(obs, keys, False, a, img)
                    if obs["success"]:
                        # keep a short hold so the success state is in the data
                        for _ in range(int(0.5 / dt)):
                            obs = env.step(0.0, 0.0, dt)
                            rec.add(obs, [False] * 4, False, (0.0, 0.0), None if not renderer else
                                    renderer.capture(env.cath.spline(), cfg.catheter.diameter, tp.image_size))
                        break
                path = rec.stop()
                print(f"{env.vessel.name:20s} {tgt.name:12s} ep{ep} "
                      f"{'OK ' if obs['success'] else 'FAIL'} t={obs['t']:5.1f}s -> {path}")


if __name__ == "__main__":
    main()
