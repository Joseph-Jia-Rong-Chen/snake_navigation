#!/usr/bin/env python
"""Train LeRobot's ACT policy on a converted catheter dataset.

Based on lerobot/examples/tutorial/act/act_training_example.py, adapted to
lerobot 0.4.4 and this project:

  * local dataset (tools/convert_lerobot.py output), nothing is downloaded or uploaded
  * RELATIVE actions: for the observation at frame t the model predicts the chunk
        rel[t, i] = action[t + i] - observation.state[t],   i = 0 .. chunk_size-1
    i.e. joint-target offsets from where the robot is when the frame is taken.
    At deployment the wrapper adds the state at prediction time back
    (training/act_policy.py). The action normalisation stats are recomputed for
    these offsets (the dataset's stats describe absolute targets).
  * optional validation on a second dataset (e.g. the val-split vessels)
  * checkpoints that the simulator wrapper can load directly

  python training/training.py --root data/lerobot/catheter_nav --out outputs/act_catheter
  python training/training.py --root data/lerobot/catheter_nav_train --val-root data/lerobot/catheter_nav_val \\
      --steps 20000 --batch-size 16 --chunk-size 60 --n-action-steps 60

Checkpoint layout (each is loadable by act_policy.py):
  <out>/step_000500/  <out>/last/  <out>/best/  (best = lowest validation L1, with --val-root)
      config.json, model.safetensors, policy_*processor*.json + stats, training_meta.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path(__file__).resolve().parent.parent
STATE, ACTION = "observation.state", "action"


def require_dataset(root: str | None, flag: str) -> None:
    """Fail clearly instead of letting LeRobot fall back to the Hugging Face Hub."""
    if root and not (Path(root) / "meta" / "info.json").exists():
        raise SystemExit(f"{flag} {root}: no LeRobot dataset there (missing meta/info.json).\n"
                         f"Create it first, e.g.  python tools/convert_lerobot.py data/raw --out {root} [--split ...]")


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


# --------------------------------------------------------------------- relative actions
def to_relative(batch: dict) -> dict:
    """Absolute action chunk (B, chunk, D) -> offsets from the state at the chunk's frame."""
    batch[ACTION] = batch[ACTION] - batch[STATE].unsqueeze(1)
    return batch


def relative_action_stats(root: Path, chunk_size: int) -> dict[str, np.ndarray]:
    """mean / std / min / max of action[t+i] - state[t] over every frame t and every
    chunk offset i that stays inside the episode (padded steps excluded)."""
    files = sorted((Path(root) / "data").rglob("*.parquet"))
    t = pq.read_table(files[0]) if len(files) == 1 else __import__("pyarrow").concat_tables(
        [pq.read_table(f) for f in files])
    ep = np.asarray(t.column("episode_index").to_numpy())
    A = np.stack(t.column(ACTION).to_numpy(zero_copy_only=False)).astype(np.float64)
    S = np.stack(t.column(STATE).to_numpy(zero_copy_only=False)).astype(np.float64)
    n, s, ss = 0, np.zeros(A.shape[1]), np.zeros(A.shape[1])
    lo, hi = np.full(A.shape[1], np.inf), np.full(A.shape[1], -np.inf)
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        a, st = A[idx], S[idx]
        for i in range(min(chunk_size, len(idx))):
            d = a[i:] - st[: len(idx) - i]
            n += len(d)
            s += d.sum(0)
            ss += (d * d).sum(0)
            lo, hi = np.minimum(lo, d.min(0)), np.maximum(hi, d.max(0))
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "min": lo.astype(np.float32), "max": hi.astype(np.float32), "count": np.array([n])}


# ----------------------------------------------------------------------------- helpers
def save_checkpoint(path: Path, policy, pre, post, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    (path / "training_meta.json").write_text(json.dumps(meta, indent=2))


@torch.no_grad()
def validate(policy, pre, loader, device, act_std: torch.Tensor, max_batches: int) -> dict:
    """Masked L1 of the predicted chunk (eval mode, no VAE sampling), in normalised
    units and in mm / deg."""
    policy.eval()
    tot, tot_phys, n = 0.0, torch.zeros(act_std.shape[-1]), 0
    for k, batch in enumerate(loader):
        if k >= max_batches:
            break
        batch = pre(to_relative(batch))
        pred = policy.predict_action_chunk(batch)
        mask = (~batch["action_is_pad"]).unsqueeze(-1).float()
        err = (pred - batch[ACTION]).abs() * mask
        tot += (err.sum() / (mask.sum() * err.shape[-1])).item()
        tot_phys += ((err.sum((0, 1)) / mask.sum()).cpu() * act_std)
        n += 1
    policy.train()
    phys = (tot_phys / max(n, 1)).tolist()
    return {"l1": tot / max(n, 1), "l1_length_mm": phys[0], "l1_bend_deg": phys[1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data/lerobot/catheter_nav"), help="training dataset")
    ap.add_argument("--repo-id", default="local/catheter_nav")
    ap.add_argument("--val-root", help="validation dataset (e.g. converted with --split val)")
    ap.add_argument("--out", default=str(ROOT / "outputs/act_catheter"))
    ap.add_argument("--steps", type=int, default=5000, help="training length in optimizer steps (batches)")
    ap.add_argument("--epochs", type=float, help="training length in epochs (passes over the data); overrides --steps")
    ap.add_argument("--patience", type=int, help="early stop after this many validations without improvement "
                                                   "(needs --val-root)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--chunk-size", type=int, default=60, help="predicted actions per observation, in frames (60 = 2 s at 30 fps)")
    ap.add_argument("--n-action-steps", type=int, default=60, help="actions executed before re-predicting (= chunk size: run the whole chunk)")
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--absolute", action="store_true", help="train on absolute targets instead of relative")
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--eval-freq", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-freq", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    require_dataset(args.root, '--root')
    require_dataset(args.val_root, '--val-root')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    relative = not args.absolute
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- features, stats, policy, processors
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    cfg = ACTConfig(input_features=input_features, output_features=output_features, device=args.device,
                    chunk_size=args.chunk_size, n_action_steps=args.n_action_steps, kl_weight=args.kl_weight,
                    optimizer_lr=args.lr, optimizer_lr_backbone=args.lr_backbone)
    stats = {k: dict(v) for k, v in meta.stats.items()}
    if relative:
        stats[ACTION] = relative_action_stats(Path(args.root), cfg.chunk_size)
    policy = ACTPolicy(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.train().to(device)
    act_std = torch.as_tensor(np.asarray(stats[ACTION]["std"]), dtype=torch.float32)

    img_key = next(iter(cfg.image_features))
    H, W = meta.features[img_key]["shape"][:2]
    train_meta = {
        "dataset_root": str(Path(args.root).resolve()), "repo_id": args.repo_id,
        "val_root": str(Path(args.val_root).resolve()) if args.val_root else None,
        "fps": meta.fps, "image_size": [W, H], "relative_action": relative,
        "action_names": meta.features[ACTION]["names"], "state_names": meta.features[STATE]["names"],
        "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
        "action_stats": {k: np.asarray(v).tolist() for k, v in stats[ACTION].items()},
        "args": vars(args),
    }
    print(f"dataset {args.root}: {meta.total_episodes} episodes, {meta.total_frames} frames, "
          f"{W}x{H} @ {meta.fps} fps")
    print(f"ACT {sum(p.numel() for p in policy.parameters()) / 1e6:.1f} M params | chunk {cfg.chunk_size} "
          f"({cfg.chunk_size / meta.fps:.2f} s), execute {cfg.n_action_steps} | "
          f"{'relative' if relative else 'absolute'} actions, std {act_std.tolist()}")

    # ---- data
    delta = {ACTION: make_delta_timestamps(cfg.action_delta_indices, meta.fps)}
    if cfg.observation_delta_indices is not None:     # not for ACT (see check_act.py)
        delta |= {k: make_delta_timestamps(cfg.observation_delta_indices, meta.fps) for k in cfg.image_features}

    def loader(root, shuffle):
        ds = LeRobotDataset(args.repo_id, root=root, delta_timestamps=delta)
        return ds, torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=True,
                                               num_workers=args.workers, pin_memory=device.type != "cpu",
                                               persistent_workers=args.workers > 0)
    train_ds, train_dl = loader(args.root, True)
    val_dl = loader(args.val_root, False)[1] if args.val_root else None
    steps_per_epoch = len(train_dl)                    # = frames // batch_size
    if args.epochs is not None:
        args.steps = max(int(round(args.epochs * steps_per_epoch)), 1)
    if args.patience is not None and val_dl is None:
        raise SystemExit("--patience needs a validation set (--val-root)")
    print(f"{steps_per_epoch} steps per epoch (batch {args.batch_size}) -> {args.steps} steps "
          f"= {args.steps / steps_per_epoch:.1f} epochs"
          + (f", early stop after {args.patience} validations without improvement" if args.patience else ""))

    # ---- train
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    step, best, t0, done, stale, reason = 0, float("inf"), time.perf_counter(), False, 0, "max steps"
    hist: list[dict] = []
    try:
        while not done:
            for batch in train_dl:
                batch = pre(to_relative(batch) if relative else batch)
                loss, info = policy.forward(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_freq == 0 or step == 1:
                    rate = step / (time.perf_counter() - t0)
                    print(f"step {step:6d}  epoch {step / steps_per_epoch:5.2f}  loss {loss.item():.4f}  "
                          f"l1 {info['l1_loss']:.4f}  kld {info.get('kld_loss', 0):.4f}  ({rate:.1f} it/s)", flush=True)
                    hist.append({"step": step, "epoch": step / steps_per_epoch, "loss": loss.item(), **info})
                if val_dl is not None and (step % args.eval_freq == 0 or step == args.steps):
                    v = validate(policy, pre, val_dl, device, act_std, args.eval_batches)
                    improved = v["l1"] < best
                    stale = 0 if improved else stale + 1
                    print(f"   val  l1 {v['l1']:.4f}  = {v['l1_length_mm']:.2f} mm, {v['l1_bend_deg']:.2f} deg"
                          + ("  (best)" if improved else f"  (no improvement {stale}x)"), flush=True)
                    hist.append({"step": step, **{f"val_{k}": x for k, x in v.items()}})
                    if improved:
                        best = v["l1"]
                        save_checkpoint(out / "best", policy, pre, post, {**train_meta, "step": step, "val": v})
                    if args.patience is not None and stale >= args.patience:
                        done, reason = True, f"early stop: no validation improvement in {stale} evaluations"
                        break
                if step % args.save_freq == 0:
                    save_checkpoint(out / f"step_{step:06d}", policy, pre, post, {**train_meta, "step": step})
                if step >= args.steps:
                    done = True
                    break
    except KeyboardInterrupt:
        reason = "stopped with Ctrl+C"

    save_checkpoint(out / "last", policy, pre, post, {**train_meta, "step": step, "stop_reason": reason})
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"{reason}: {step} steps ({step / steps_per_epoch:.1f} epochs) in {(time.perf_counter() - t0) / 60:.1f} min"
          f" -> {out / 'last'}" + (f"  (best val l1 {best:.4f} -> {out / 'best'})" if best < float("inf") else ""))


if __name__ == "__main__":
    main()
