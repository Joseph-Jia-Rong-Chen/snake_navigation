#!/usr/bin/env python
"""Launch the catheter system (the analogue of a ROS2 launch file).

  Teleop (keyboard), recording on Enter:
    python launch.py
    python launch.py --model vessel_03 --record          # record every episode (starts on reset)

  Policy in control (operator can take over with any motion key, G gives it back):
    python launch.py --commander policy --policy policies/example_policy.py:CreepPolicy
    python launch.py --commander policy --policy my_pkg.my_policy:Net --policy-arg ckpt=runs/best.pt

  Headless batches (no window; lockstep = as fast as possible, deterministic):
    python launch.py --commander autopilot --headless --record --episodes 3 --noise 0.3
    python launch.py --commander policy --policy ... --headless --episodes 5   # evaluation

Node graph:
  viewer (/ui/keys, events) -> teleop | policy | autopilot -> mux -> /robot/joint_target
  robot (clamp, lag, speed limit, physics) -> /robot/joint_state, /sim/ground_truth
  camera -> /camera/image, /camera/info      feature_map -> /nav/feature_map
  task -> /sim/world, /nav/target, /task/status      recorder -> data/raw/<model>/<episode>/{video.mp4,streams.npz,meta.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import catheter_sim  # noqa: F401  (sets BLAS threads before numpy)

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commander", choices=["teleop", "policy", "autopilot"], default="teleop")
    ap.add_argument("--policy", help="file.py:ClassName or module:ClassName (with --commander policy)")
    ap.add_argument("--policy-arg", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--model", help="vessel model name/stem/index (default: first)")
    ap.add_argument("--models", help="headless: comma-separated models to run (default: all, or --model)")
    ap.add_argument("--split", help="only vessel models of this split: train | val | test (comma list ok)")
    ap.add_argument("--target", help="target name or index")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--record", action="store_true", help="record every episode")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--record-images", action="store_true", help="also store lossless frames (video size) in streams.npz")
    ap.add_argument("--headless", action="store_true", help="no window (needs --commander policy/autopilot)")
    ap.add_argument("--episodes", type=int, default=None, help="episodes per target (enables the supervisor)")
    ap.add_argument("--max-time", type=float, default=90.0, help="supervisor: episode timeout [s sim]")
    ap.add_argument("--lockstep", action="store_true", help="don't pace to wall-clock time")
    ap.add_argument("--realtime", action="store_true", help="pace to wall-clock time even when headless")
    ap.add_argument("--noise", type=float, default=0.0, help="autopilot action noise 0..1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=None, help="stop after this much sim time [s]")
    return ap


def resolve_model(models: list[Path], spec: str | None) -> int:
    if spec is None:
        return 0
    if spec.isdigit() and int(spec) < len(models):
        return int(spec)
    for i, m in enumerate(models):
        if spec in (m.name, m.stem):
            return i
    sys.exit(f"model '{spec}' not found in {[m.stem for m in models]}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.headless and args.commander == "teleop":
        sys.exit("--headless needs --commander policy or autopilot")
    if args.commander == "policy" and not args.policy:
        sys.exit("--commander policy needs --policy file.py:ClassName")
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from catheter_sim.bus import Bus
    from catheter_sim.config import load_config
    from catheter_sim.nodes import (AutopilotNode, CameraNode, FeatureMapNode, MuxNode, PolicyNode,
                                    RecorderNode, RobotSimNode, SupervisorNode, TaskNode, TeleopNode)
    from catheter_sim.policy_api import load_policy, parse_kwargs
    from catheter_sim.vessel import list_models

    cfg = load_config(args.config, args.set)
    if args.no_video:
        cfg.recording.record_video = False
    if args.record_images:
        cfg.recording.record_images = True
    models = list_models(args.models_dir, args.split)
    if not models:
        sys.exit(f"no *.json models in {args.models_dir}" + (f" with split {args.split}" if args.split else ""))
    mi = resolve_model(models, args.model)
    realtime = (not args.headless and not args.lockstep) or args.realtime

    pygame.init()
    bus = Bus(cfg.sim.rate_hz, realtime=realtime)
    task = TaskNode(bus, models, mi, args.target)
    if not args.headless:
        from catheter_sim.nodes.viewer import ViewerNode
        ViewerNode(bus, cfg, args.commander)
        TeleopNode(bus, cfg)                       # always available: driving or takeover
    if args.episodes is not None or args.headless:
        sel = ([resolve_model(models, m) for m in args.models.split(",")] if args.models
               else [mi] if args.model else list(range(len(models))))
        targets = [args.target] if args.target else None
        SupervisorNode(bus, sel, targets, args.episodes or 1, args.max_time, args.record)
    if args.commander == "autopilot":
        AutopilotNode(bus, cfg, noise=args.noise, seed=args.seed)
    elif args.commander == "policy":
        policy = load_policy(args.policy, parse_kwargs(args.policy_arg))
        PolicyNode(bus, cfg, policy, asynchronous=realtime)
    MuxNode(bus, args.commander)
    RobotSimNode(bus, cfg)
    CameraNode(bus, cfg)
    FeatureMapNode(bus)
    data_dir = Path(cfg.recording.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    supervised = args.episodes is not None or args.headless
    RecorderNode(bus, cfg, data_dir, autorecord=args.record and not supervised,
                 meta={"commander": args.commander, "policy": args.policy, "realtime": realtime})
    task.start()
    bus.spin(max_time=args.duration)
    pygame.quit()


if __name__ == "__main__":
    main()
