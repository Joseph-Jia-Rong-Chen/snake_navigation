"""Run a trained ACT checkpoint (training/training.py) in the simulator.

    python launch.py --commander policy --policy training/act_policy.py:ACTSimPolicy \\
        --policy-arg checkpoint=outputs/act_catheter/last --model vessel_21
    python launch.py --commander policy --policy training/act_policy.py:ACTSimPolicy \\
        --policy-arg checkpoint=outputs/act_catheter/best --headless --split val --episodes 2

Every call of act() (at the dataset fps, 30 Hz) receives the live observation
from PolicyNode: grayscale frame, U / V feature maps and joints. They are encoded
exactly like the dataset rows (catheter_sim.lerobot_export.observation_to_frame:
gray -> 3 channels, goal maps -> uint8) and normalised by the checkpoint's own
preprocessor.

Action chunking: when the queue is empty the model predicts a chunk of
`chunk_size` actions from the current observation; the first `n_action_steps`
are queued and executed one per frame, then it predicts again. With relative
actions the chunk is converted to absolute joint targets using the joint state
AT PREDICTION TIME (the state the offsets were trained against), not the state
when each queued action is sent.

You can operate the operator takeover as usual (any motion key; G resumes);
after a resume the queue is cleared so stale actions are not replayed.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catheter_sim.lerobot_export import observation_to_frame, task_text  # noqa: E402
from catheter_sim.policy_api import Policy  # noqa: E402


class ACTSimPolicy(Policy):
    def __init__(self, checkpoint: str = "outputs/act_catheter/last", device: str | None = None,
                 n_action_steps: int | None = None, verbose: bool = True):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        ckpt = Path(checkpoint)
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt
        if not (ckpt / "config.json").exists():
            found = sorted(str(p.parent.relative_to(ROOT)) for p in ckpt.parent.glob("*/config.json")) \
                if ckpt.parent.exists() else []
            hint = (f"available: {', '.join(found)}" if found else "train with training/training.py")
            if ckpt.name == "best":
                hint = "best/ is only written when training with --val-root; " + hint
            raise SystemExit(f"{ckpt} is not an ACT checkpoint (no config.json). {hint}")
        self.meta = json.loads((ckpt / "training_meta.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ACTPolicy.from_pretrained(ckpt)
        self.model.config.device = str(self.device)
        self.model.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.model.config, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        self.relative = self.meta["relative_action"]
        self.n_steps = int(n_action_steps or self.meta["n_action_steps"])
        self.rate_hz = float(self.meta["fps"])                  # one action per dataset frame
        self.input_size = tuple(self.meta["image_size"])       # (W, H) the model was trained on
        self.verbose = verbose
        self.queue: deque = deque()
        self.task = ""
        self.last_infer_ms = 0.0
        if verbose:
            print(f"ACT {ckpt}: {'relative' if self.relative else 'absolute'} actions, chunk "
                  f"{self.meta['chunk_size']}, execute {self.n_steps}, input {self.input_size} @ {self.rate_hz:g} Hz, "
                  f"{self.device}")

    def reset(self, info: dict) -> None:
        self.queue.clear()
        self.task = task_text(info.get("model") or "", info.get("target") or "")

    def on_event(self, kind: str) -> None:
        if kind in ("takeover", "resume"):
            self.queue.clear()                   # never replay actions planned before the operator moved it

    # ------------------------------------------------------------------
    def _batch(self, obs: dict) -> dict:
        """PolicyNode observation -> the tensors a LeRobotDataset item would contain."""
        f = observation_to_frame(obs)                       # uint8 (H, W, 3) images, float32 state
        b = {k: torch.from_numpy(v).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
             for k, v in f.items() if k.startswith("observation.images.")}
        b["observation.state"] = torch.from_numpy(f["observation.state"]).unsqueeze(0)
        b["task"] = [self.task or task_text("", obs.get("target_name") or "")]
        return b

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        """(chunk_size, 2) absolute joint targets [length_mm, bend_deg] for this observation."""
        t = time.perf_counter()
        chunk = self.model.predict_action_chunk(self.pre(self._batch(obs)))      # normalised (1, C, 2)
        chunk = self.post(chunk)[0].cpu().numpy().astype(np.float64)              # mm / deg
        if self.relative:
            chunk = chunk + np.asarray(obs["joint"], np.float64)[None, :]
        self.last_infer_ms = (time.perf_counter() - t) * 1e3
        return chunk

    def act(self, obs: dict):
        if not self.queue:
            self.queue.extend(self.predict_chunk(obs)[: self.n_steps])
        a = self.queue.popleft()
        return float(a[0]), float(a[1])
