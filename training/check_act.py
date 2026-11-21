#!/usr/bin/env python
"""Check that a converted catheter dataset and LeRobot's ACT policy fit together.

Follows lerobot/examples/tutorial/act/act_training_example.py (adapted to the
import paths of lerobot 0.4.4 and a local dataset), then runs one inference step:

  1. dataset metadata -> policy input / output features
  2. ACTConfig + ACTPolicy + pre/post processors (normalisation from the dataset stats)
  3. dataset with ACT's action chunk (delta_timestamps), one batch through the
     preprocessor, forward + backward + optimizer step
  4. inference: one observation -> select_action -> postprocess -> [length_mm, bend_deg]

  python training/check_act.py --root data/lerobot/catheter_nav
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path(__file__).resolve().parent.parent


def require_dataset(root: str | None, flag: str) -> None:
    """Fail clearly instead of letting LeRobot fall back to the Hugging Face Hub."""
    if root and not (Path(root) / "meta" / "info.json").exists():
        raise SystemExit(f"{flag} {root}: no LeRobot dataset there (missing meta/info.json).\n"
                         f"Create it first, e.g.  python tools/convert_lerobot.py data/raw --out {root} [--split ...]")


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def show(title: str, d: dict) -> None:
    print(f"\n== {title}")
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"   {k:34s} {str(tuple(v.shape)):22s} {str(v.dtype):15s} "
                  f"[{v.float().min():+.3f}, {v.float().max():+.3f}]")
        else:
            print(f"   {k:34s} {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data/lerobot/catheter_nav"))
    ap.add_argument("--repo-id", default="local/catheter_nav")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    require_dataset(args.root, '--root')
    device = torch.device(args.device)

    # 1. features from the dataset
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    print(f"dataset: {meta.total_episodes} episodes, {meta.total_frames} frames, {meta.fps} fps")
    show("policy inputs", {k: f"{ft.type.name:7s} {ft.shape}" for k, ft in input_features.items()})
    show("policy outputs", {k: f"{ft.type.name:7s} {ft.shape}" for k, ft in output_features.items()})

    # 2. policy + processors
    cfg = ACTConfig(input_features=input_features, output_features=output_features, device=args.device)
    policy = ACTPolicy(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    policy.train().to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nACT: {n_params / 1e6:.1f} M parameters, chunk_size {cfg.chunk_size}, "
          f"n_action_steps {cfg.n_action_steps}, backbone {cfg.vision_backbone}, "
          f"image features {list(cfg.image_features)}")

    # 3. dataset with action chunks, one training step
    # Only the action needs a time window (ACT's chunk). The upstream example also passes
    # [0] for every image; in lerobot 0.4.4 that adds a time axis to `image` features
    # (B, 1, C, H, W) but not to `video` ones, which ACT's backbone rejects.
    delta = {"action": make_delta_timestamps(cfg.action_delta_indices, meta.fps)}
    if cfg.observation_delta_indices is not None:
        delta |= {k: make_delta_timestamps(cfg.observation_delta_indices, meta.fps) for k in cfg.image_features}
    ds = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=2, pin_memory=device.type != "cpu")
    batch = next(iter(loader))
    show("raw batch from the dataset", batch)
    batch = pre(batch)
    show("after the preprocessor (normalised, on device)", batch)
    for k in cfg.image_features:
        m = batch[k].abs().max().item()
        if m > 50:
            raise SystemExit(f"\n{k}: normalised values reach {m:.0f}; the image std in meta/stats.json is "
                             "too small. Re-convert with tools/convert_lerobot.py (it recomputes image stats).")

    opt = cfg.get_optimizer_preset().build(policy.parameters())
    t = time.perf_counter()
    loss, info = policy.forward(batch)
    loss.backward()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"\n== training step OK: loss {loss.item():.3f} {({k: round(v, 3) for k, v in info.items()})}, "
          f"{time.perf_counter() - t:.2f} s")

    # 4. inference on one observation, as at deployment
    policy.eval()
    policy.reset()
    sample = LeRobotDataset(args.repo_id, root=args.root)[0]
    obs = {k: v.unsqueeze(0) for k, v in sample.items() if k.startswith("observation.")}
    obs["task"] = [sample["task"]]
    with torch.no_grad():
        action = post(policy.select_action(pre(obs)))
    print(f"\n== inference OK: action {tuple(action.shape)} = {action.squeeze(0).tolist()} "
          f"(untrained; dataset label {sample['action'].tolist()})")
    print("\nCOMPATIBLE: dataset features, normalisation stats, action chunks and ACT all line up.")


if __name__ == "__main__":
    main()
