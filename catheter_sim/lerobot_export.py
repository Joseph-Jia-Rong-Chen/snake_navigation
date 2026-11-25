"""Raw episodes -> LeRobot dataset (lerobot 0.4.x), and the matching encoders.

Schema (H, W = the export size; LeRobot needs 3-channel images, so the
grayscale frame and each goal map are replicated into 3 identical channels):

  observation.state             float32 (2,)      [length_mm, bend_deg]  joints when the frame was taken
  action                        float32 (2,)      [length_mm, bend_deg]  target commanded in response
  observation.images.cam        video (H, W, 3)   grayscale camera frame
  observation.images.x_goal_map image (H, W, 3)   U encoded as uint8
  observation.images.y_goal_map image (H, W, 3)   V encoded as uint8
  task                          "navigate to <target> in <model>"

Goal map encoding:  q = round((u + 1) / 2 * 255),  u = q / 127.5 - 1
(-1 -> 0, 0 -> 128, +1 -> 255; outside the vessel u = 1 -> 255). Use
decode_goal_map() in training code to get U / V back in [-1, 1].

At deployment use observation_to_frame(obs) on a PolicyNode observation to get
exactly the arrays the dataset stores.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .dataset import Episode

STATE_NAMES = ["length_mm", "bend_deg"]


def encode_goal_map(u: np.ndarray) -> np.ndarray:
    """(H, W) float in [-1, 1] -> (H, W, 3) uint8."""
    q = np.round((np.asarray(u, np.float32) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np.repeat(q[..., None], 3, axis=2)


def decode_goal_map(img) -> np.ndarray:
    """(H, W, 3) or (3, H, W) uint8 (or float in [0, 1] as LeRobot returns it) -> (H, W) float32 in [-1, 1]."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = a[0]
    elif a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    if a.max() <= 1.0 and np.asarray(img).dtype != np.uint8:
        a = a * 255.0
    return a / 127.5 - 1.0


def gray3(img: np.ndarray) -> np.ndarray:
    """Grayscale (H, W) -> (H, W, 3) with identical channels (RGB is converted first)."""
    if img.ndim == 3:
        from .render import to_gray
        img = to_gray(img)
    return np.repeat(img[..., None], 3, axis=2)


def features(width: int, height: int) -> dict:
    img = {"shape": (height, width, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": STATE_NAMES},
        "observation.images.cam": {"dtype": "video", **img},
        "observation.images.x_goal_map": {"dtype": "image", **img},
        "observation.images.y_goal_map": {"dtype": "image", **img},
    }


def make_frame(image, U, V, state, action=None) -> dict:
    f = {"observation.state": np.asarray(state, np.float32),
         "observation.images.cam": gray3(image),
         "observation.images.x_goal_map": encode_goal_map(U),
         "observation.images.y_goal_map": encode_goal_map(V)}
    if action is not None:
        f["action"] = np.asarray(action, np.float32)
    return f


def observation_to_frame(obs: dict) -> dict:
    """PolicyNode observation -> the dataset's observation arrays (no action)."""
    return make_frame(obs["image"], obs["feature_u"], obs["feature_v"], obs["joint"])


def task_text(model: str, target: str) -> str:
    return f"navigate to {target} in {model}"


def episode_split(meta: dict, models_dir: Path | None = None) -> str:
    """Split of the episode's vessel: the current model file decides (so a model can
    be moved between splits later); falls back to what was recorded."""
    from .vessel import model_split
    for cand in ([Path(models_dir) / f"{meta['model']}.json"] if models_dir else []) + [Path(meta["model_path"])]:
        if cand.exists():
            return model_split(cand)
    return meta.get("split", "train")


def select(episodes: list[Path], only_success: bool = False, sources: set[str] | None = None,
           models: set[str] | None = None, splits: set[str] | None = None,
           models_dir: Path | None = None) -> tuple[list[Path], list[tuple[str, str]]]:
    """Episode-level filters on meta.json. Returns (kept, [(name, reason)])."""
    kept, skipped = [], []
    for p in episodes:
        m = json.loads((p / "meta.json").read_text())
        sp = episode_split(m, models_dir)
        if splits and sp not in splits:
            skipped.append((p.name, f"split {sp}"))
        elif models and m["model"] not in models:
            skipped.append((p.name, f"model {m['model']}"))
        elif only_success and not m.get("reached_target"):
            skipped.append((p.name, "target not reached"))
        elif sources and not set(m.get("sources", [])) <= sources:
            skipped.append((p.name, f"sources {m.get('sources')}"))
        elif not m.get("video"):
            skipped.append((p.name, "no video"))
        else:
            kept.append(p)
    return kept, skipped


def convert(episodes: list[Path], out: Path, repo_id: str, scale: float = 1.0,
            action_align: str = "last", state_align: str = "hold", fill_gaps: bool = False,
            vcodec: str = "h264", overwrite: bool = False, image_threads: int = 4,
            hold_end: float = 2.0) -> dict:
    """hold_end: seconds of "arrived, stay still" appended to every episode whose last
    frame is inside the target: the last frame / maps / joint state repeated, with
    action = that joint state (relative offset 0). Raw recordings are not changed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    eps = [Episode(p, scale=scale, action_align=action_align, state_align=state_align) for p in episodes]
    if not eps:
        raise ValueError("no episodes to convert")
    size, fps = eps[0].size, eps[0].fps
    ds = LeRobotDataset.create(repo_id=repo_id, root=out, fps=fps, features=features(*size),
                               robot_type="catheter_2dof", use_videos=True, vcodec=vcodec,
                               image_writer_threads=image_threads)
    n_hold = int(round(hold_end * fps))
    rep = {"size": list(size), "fps": fps, "action_align": action_align, "state_align": state_align,
           "hold_end_s": hold_end, "hold_end_frames": n_hold, "episodes": [], "skipped": []}
    for ep in eps:
        name = ep.dir.name
        r = {"episode": name, "path": str(ep.dir), **ep.sync_report()}
        if ep.size != size or ep.fps != fps:
            rep["skipped"].append((name, f"size {ep.size} / fps {ep.fps} differs from {size} / {fps}"))
            continue
        if len(ep.a["gaps"]) and not fill_gaps:
            rep["skipped"].append((name, f"{len(ep.a['gaps'])} timing gaps (use --fill-gaps)"))
            continue
        model = ep.meta["model"]
        prev = None
        n = 0
        for k, s in ep.samples():
            frame = make_frame(s["image"], s["feature_u"], s["feature_v"], s["state"], s["action"])
            frame["task"] = task_text(model, s["target_name"])
            repeats = 1
            if fill_gaps and k > 0 and (k - 1) in ep.a["gaps"]:
                dt = ep.a["stamp"][k] - ep.a["stamp"][k - 1]
                repeats = max(int(round(dt * fps)), 1)
                for _ in range(repeats - 1):                 # repeat the previous frame
                    ds.add_frame(dict(prev))
                    n += 1
            prev = {**frame}                                 # add_frame pops "task"
            last = s
            ds.add_frame(frame)
            n += 1
        arrived = bool(ep.a["success"][-1]) if "success" in ep.a else False
        held = n_hold if (arrived and n_hold > 0) else 0
        if held:                                             # arrived: stay still for hold_end s
            hold = make_frame(last["image"], last["feature_u"], last["feature_v"], last["state"], last["state"])
            hold["task"] = prev["task"]
            for _ in range(held):
                ds.add_frame(dict(hold))
            n += held
        ds.save_episode()
        r["lerobot_episode_index"] = len(rep["episodes"])
        r["lerobot_frames"] = n
        r["hold_frames_added"] = held
        rep["episodes"].append(r)
        print(f"  {name}: {n} frames ({held} hold)  sources {r['action_sources']}  gaps {len(r['gaps'])}",
              flush=True)
    ds.finalize()
    rep["image_stats"] = fix_image_stats(out, repo_id)
    (out / "meta" / "sync_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def fix_image_stats(root: Path, repo_id: str = "local/catheter_nav", stride: int = 4) -> dict:
    """Recompute per-channel mean / std of the image features in float64 and write
    them into meta/stats.json.

    lerobot 0.4.4 computes variance as E[x^2] - E[x]^2 in float32 over millions of
    pixels; for these mostly flat fluoroscopy-like images (spread << mean) that
    cancels to ~0 and the stored std comes out 20-70x too small, so normalised
    pixels blow up to hundreds. Uses every `stride`-th frame."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    ds = LeRobotDataset(repo_id, root=root)
    keys = [k for k, f in ds.meta.features.items() if f["dtype"] in ("image", "video")]
    acc = {k: [0.0, np.zeros(3), np.zeros(3)] for k in keys}          # n, sum, sum of squares
    for i in range(0, len(ds), stride):
        item = ds[i]
        for k in keys:
            x = item[k].double().numpy().reshape(3, -1)                 # (C, H*W) in [0, 1]
            acc[k][0] += x.shape[1]
            acc[k][1] += x.sum(1)
            acc[k][2] += (x * x).sum(1)
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    out = {}
    for k, (n, s, ss) in acc.items():
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
        old = float(np.asarray(stats[k]["std"]).ravel()[0])
        stats[k]["mean"] = mean.reshape(3, 1, 1).tolist()
        stats[k]["std"] = std.reshape(3, 1, 1).tolist()
        out[k] = {"mean": round(float(mean[0]), 4), "std": round(float(std[0]), 4), "lerobot_std": round(old, 5)}
    path.write_text(json.dumps(stats, indent=4))
    return out
