#!/usr/bin/env python
"""Convert raw recorded episodes into a LeRobot dataset.

  python tools/convert_lerobot.py data/raw --out data/lerobot/catheter_nav_train --split train
  python tools/convert_lerobot.py data/raw --out data/lerobot/catheter_nav_val --split val
  python tools/convert_lerobot.py data/raw/vessel_02 --out data/lerobot/v02_small --scale 0.5 \\
      --only-success --sources teleop --overwrite

Inputs are raw episode directories (anything below the given paths containing
meta.json + streams.npz). Output size = recorded video size * --scale. A sync
report per episode is written to <out>/meta/sync_report.json. Schema and goal-map
encoding: catheter_sim/lerobot_export.py; alignment rules: catheter_sim/sync.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import catheter_sim  # noqa: E402,F401
from catheter_sim.lerobot_export import convert, select  # noqa: E402
from catheter_sim.recorder import list_raw  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="raw episode directories or folders containing them")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--repo-id", default="local/catheter_nav", help="dataset name (nothing is uploaded)")
    ap.add_argument("--scale", type=float, default=1.0, help="resize relative to the recorded video")
    ap.add_argument("--only-success", action="store_true", help="drop episodes that never reached the target")
    ap.add_argument("--sources", help="keep episodes whose commands all came from these, e.g. teleop,takeover")
    ap.add_argument("--split", help="keep episodes on vessels of this split: train | val | test (comma list ok)")
    ap.add_argument("--models", help="comma-separated vessel models to keep")
    ap.add_argument("--models-dir", default=str(ROOT / "models"), help="where model files (and their split) live")
    ap.add_argument("--action-align", choices=["last", "first"], default="last")
    ap.add_argument("--state-align", choices=["hold", "interp"], default="hold")
    ap.add_argument("--fill-gaps", action="store_true", help="repeat frames over timing gaps instead of skipping")
    ap.add_argument("--hold-end", type=float, default=2.0,
                    help="seconds of 'arrived, stay still' appended to episodes that end in the target "
                         "(last frame + state repeated, action = state); 0 disables")
    ap.add_argument("--vcodec", default="h264", help="LeRobot video codec: h264 | hevc | libsvtav1 | auto")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be converted")
    args = ap.parse_args(argv)

    eps = sorted({p for i in args.inputs for p in list_raw(i)})
    kept, skipped = select(eps, args.only_success,
                           set(args.sources.split(",")) if args.sources else None,
                           set(args.models.split(",")) if args.models else None,
                           set(args.split.split(",")) if args.split else None, Path(args.models_dir))
    print(f"{len(eps)} raw episodes, {len(kept)} selected")
    for name, why in skipped:
        print(f"  skip {name}: {why}")
    if args.dry_run or not kept:
        return
    rep = convert(kept, Path(args.out), args.repo_id, args.scale, args.action_align, args.state_align,
                  args.fill_gaps, args.vcodec, args.overwrite, hold_end=args.hold_end)
    for name, why in rep["skipped"]:
        print(f"  skip {name}: {why}")
    total = sum(e["lerobot_frames"] for e in rep["episodes"])
    print(f"wrote {len(rep['episodes'])} episodes, {total} frames, {rep['size'][0]}x{rep['size'][1]} "
          f"@ {rep['fps']} fps -> {args.out}")


if __name__ == "__main__":
    main()
