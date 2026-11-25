#!/usr/bin/env python
"""Render annotated videos of recorded episodes (for viewing / presenting results).

The recorded video.mp4 stays the plain camera stream the policy saw; this draws
an annotated copy from the logged data instead:
  * the camera frame (upscaled to the camera resolution by default)
  * the active target (green circle + name), other targets faint
  * the tip path so far
  * a status bar: model / target, time, distance to target, who is driving
  * a final banner: TARGET REACHED / NOT REACHED

  python tools/render_video.py data/eval/run1                      # every episode below -> <episode>/overlay.mp4
  python tools/render_video.py data/eval/run1 --out videos/run1    # all into one folder, named <model>_<episode>.mp4
  python tools/render_video.py data/raw/vessel_21/<episode> --size video   # at the recorded video size
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import catheter_sim  # noqa: E402,F401
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from catheter_sim.dataset import Episode  # noqa: E402
from catheter_sim.recorder import VideoWriter, even_size, list_raw  # noqa: E402
from catheter_sim.render import array_to_surface, resize_image, surface_to_array  # noqa: E402

GREEN, FAINT, TRAIL = (40, 210, 90), (120, 160, 130), (255, 170, 40)


def render_episode(ep_dir: Path, out_path: Path, size: str = "camera", trail: bool = True) -> dict:
    ep = Episode(ep_dir)
    d, a, meta = ep.d, ep.a, ep.meta
    cam = ep.cam if size == "camera" else ep.vcam
    W, H = even_size(cam.width, cam.height)
    cam = cam.scaled(W, H)
    s = W / 1280
    font = pygame.font.SysFont("dejavusansmono,monospace", max(int(18 * s), 11))
    big = pygame.font.SysFont("dejavusans,sans", max(int(34 * s), 16), bold=True)
    small = pygame.font.SysFont("dejavusansmono,monospace", max(int(14 * s), 10))
    targets = {t.name: t for t in ep.vessel.targets}
    tips = cam.world_to_px(d["gt_tip_pos"]) if trail else None
    reached = bool(meta.get("reached_target"))
    driver = "-"
    writer = VideoWriter(out_path, (W, H), ep.fps, gray=False)
    try:
        for k, frame in enumerate(ep.iter_frames()):
            surf = array_to_surface(resize_image(frame, (W, H)))
            tname = ep.target_name(k)
            for t in targets.values():                         # targets
                p = cam.world_to_px(t.position)[0]
                active = t.name == tname
                pygame.draw.circle(surf, GREEN if active else FAINT, p, max(t.radius * cam.scale, 5),
                                   max(int(3 * s), 2) if active else 1)
                if active:
                    lab = font.render(t.name, True, GREEN)
                    off = np.array([12, -lab.get_height() / 2]) * np.array([s, 1])
                    if p[0] + off[0] + lab.get_width() > W:           # keep inside the frame
                        off[0] = -12 * s - lab.get_width()
                    surf.blit(lab, p + off)
            if tips is not None:                               # tip path up to this frame
                gi = int(a["gt_index"][k]) + 1
                pts = tips[:gi:3]
                if len(pts) >= 2:
                    pygame.draw.lines(surf, TRAIL, False, pts, max(int(2 * s), 1))
            bar = pygame.Surface((W, int(34 * s) + 6), pygame.SRCALPHA)   # status bar
            bar.fill((0, 0, 0, 150))
            surf.blit(bar, (0, 0))
            dist = a["target_dist"][k] if "target_dist" in a else np.nan
            if a["action_source"][k] != "hold":                # 'hold' = no new command this frame
                driver = a["action_source"][k]
            txt = (f"{meta['model']}  ->  {tname}    t = {a['t'][k]:5.1f} s    "
                   f"dist {dist:5.1f} mm    driver: {driver}")
            surf.blit(font.render(txt, True, (240, 240, 240)), (int(10 * s), int(8 * s)))
            if k == len(ep) - 1 or a["success"][k]:            # banner at the end / on success
                msg, col = ("TARGET REACHED", GREEN) if (reached and a["success"][k]) else \
                           (("NOT REACHED", (235, 70, 60)) if k == len(ep) - 1 else (None, None))
                if msg:
                    r = big.render(msg, True, col)
                    surf.blit(r, ((W - r.get_width()) // 2, H - r.get_height() - int(24 * s)))
            surf.blit(small.render(ep.dir.name, True, (200, 200, 200)), (int(10 * s), H - int(22 * s)))
            writer.write(surface_to_array(surf))
        last = writer.n
        # hold the final frame for 1 s so the result is readable
        for _ in range(int(ep.fps)):
            writer.write(surface_to_array(surf))
    finally:
        writer.close()
    return {"episode": ep.dir.name, "model": meta["model"], "target": meta["target"],
            "reached": reached, "frames": last, "out": str(out_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="raw episode directories or folders containing them")
    ap.add_argument("--out", help="write all videos into this folder (default: overlay.mp4 inside each episode)")
    ap.add_argument("--size", choices=["camera", "video"], default="camera",
                    help="camera = full camera resolution (default), video = recorded video size")
    ap.add_argument("--no-trail", action="store_true", help="don't draw the tip path")
    args = ap.parse_args(argv)
    pygame.init()
    eps = sorted({p for i in args.inputs for p in list_raw(i)})
    if not eps:
        raise SystemExit("no recorded episodes found")
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for p in eps:
        dst = (out_dir / f"{p.parent.name}_{p.name}.mp4") if out_dir else p / "overlay.mp4"
        r = render_episode(p, dst, args.size, not args.no_trail)
        ok += r["reached"]
        print(f"{r['model']:10s} {r['target']:11s} {'OK  ' if r['reached'] else 'FAIL'}  -> {dst}", flush=True)
    print(f"{len(eps)} videos, {ok}/{len(eps)} reached the target")


if __name__ == "__main__":
    main()
