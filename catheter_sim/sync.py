"""Align the raw streams of an episode to the camera timeline.

The camera frames are the master clock (one dataset row per video frame). For
frame k with capture stamp t_k and sequence number s_k (see msgs.py: stamp =
when the data was valid, seq = total order of everything on the bus):

  state   "hold"   (default) the latest /robot/joint_state with seq < s_k: the
                   newest measurement that existed when the frame was taken.
                   This is causal, i.e. exactly what a policy has at deployment.
          "interp" linear interpolation of the state stream at t_k (for real
                   hardware whose sensors are not synchronised with the camera)
  action  "last"   (default) the last /robot/joint_target with s_k < seq < s_{k+1}:
                   the command in effect when the next frame is taken, i.e. the
                   response to frame k. Within one tick the order is
                   commanders (20) -> mux (30) -> robot (40) -> camera (50), so a
                   command with the same stamp as frame k+1 but a smaller seq was
                   issued (and executed) before frame k+1 was captured.
          "first"  the first command after frame k
          If no command falls in the window (slow policy), the previous action
          is held, because the robot holds its target too (source "hold").
  maps    the latest feature map with seq < s_k (switches at the right frame when
          the target changes mid-episode)
  gt / status   latest sample with seq < s_k (sim-only labels, replay)

Timing check: frame stamps must be spaced exactly 1/fps; anything else is a
dropped or duplicated frame and is reported in `gaps`.
"""
from __future__ import annotations

import numpy as np


def latest_before(stream_seq: np.ndarray, query_seq: np.ndarray) -> np.ndarray:
    """Index of the last stream element with seq < query (-1 if none). Stream seqs are increasing."""
    return np.searchsorted(stream_seq, query_seq, side="left") - 1


def align(d: dict, fps: float, action_align: str = "last", state_align: str = "hold") -> dict:
    fs, ft = d["frame_seq"], d["frame_t"]
    T = len(fs)
    period = 1.0 / fps
    out: dict = {"t": ft - ft[0], "stamp": ft.copy(), "seq": fs.copy()}

    # ---- observation.state
    ss, st = d["state_seq"], d["state_t"]
    si = latest_before(ss, fs)
    out["state_missing"] = int((si < 0).sum())
    si = np.clip(si, 0, len(ss) - 1)
    L, B = d["state_length"], d["state_bend"]
    if state_align == "interp":
        out["state"] = np.stack([np.interp(ft, st, L), np.interp(ft, st, B)], 1)
    elif state_align == "hold":
        out["state"] = np.stack([L[si], B[si]], 1)
    else:
        raise ValueError(f"state_align must be hold or interp, not {state_align!r}")
    out["state_vel"] = np.stack([d["state_length_vel"][si], d["state_bend_vel"][si]], 1)
    out["state_target"] = np.stack([d["state_target_length"][si], d["state_target_bend"][si]], 1)
    out["state_lag"] = ft - st[si]                       # age of the state at the frame [s]

    # ---- action
    cs, ct = d["cmd_seq"], d["cmd_t"]
    # window upper bound: next frame's seq; for the last frame, commands up to one period later
    tail = cs[ct <= ft[-1] + period + 1e-9]
    last_ub = (tail[-1] + 1) if len(tail) else fs[-1] + 1
    ub = np.append(fs[1:], max(last_ub, fs[-1] + 1))
    lo = latest_before(cs, fs)                           # last command before frame k
    hi = latest_before(cs, ub)                           # last command before the window end
    n_in = hi - lo                                       # commands inside (s_k, ub)
    if action_align == "last":
        ci = hi
    elif action_align == "first":
        ci = lo + 1
    else:
        raise ValueError(f"action_align must be last or first, not {action_align!r}")
    action = np.zeros((T, 2))
    source = np.empty(T, dtype=object)
    for k in range(T):
        if n_in[k] > 0:
            action[k] = d["cmd_length"][ci[k]], d["cmd_bend"][ci[k]]
            source[k] = str(d["cmd_source"][ci[k]])
        elif k > 0:
            action[k], source[k] = action[k - 1], "hold"
        elif lo[0] >= 0:                                 # command issued before the first frame
            action[k], source[k] = (d["cmd_length"][lo[0]], d["cmd_bend"][lo[0]]), "hold"
        else:                                            # nothing commanded yet: robot holds
            action[k], source[k] = out["state_target"][k], "hold"
    out["action"], out["action_source"] = action, source.astype(str)
    out["cmds_per_frame"] = n_in

    # ---- feature maps, sim labels
    out["map_index"] = (np.clip(latest_before(d["fmap_seq"], fs), 0, None)
                        if "fmap_seq" in d else np.full(T, -1))
    if len(d.get("gt_seq", [])):
        out["gt_index"] = np.clip(latest_before(d["gt_seq"], fs), 0, None)
    if len(d.get("status_seq", [])):
        k = np.clip(latest_before(d["status_seq"], fs), 0, None)
        out["success"] = d["status_success"][k].astype(bool)
        out["target_dist"] = d["status_target_dist"][k]

    # ---- timing
    dt = np.diff(ft)
    out["gaps"] = np.nonzero(np.abs(dt - period) > 1e-6)[0]          # frame k -> k+1 not one period
    return out


def report(a: dict) -> dict:
    """Summary of an alignment (written to the conversion sync report)."""
    lag = a["state_lag"]
    src, cnt = np.unique(a["action_source"], return_counts=True)
    return {
        "frames": int(len(a["t"])),
        "gaps": [int(g) for g in a["gaps"]],
        "state_missing": a["state_missing"],
        "state_lag_ms": {"mean": round(float(lag.mean()) * 1e3, 3), "max": round(float(lag.max()) * 1e3, 3)},
        "cmds_per_frame": {"min": int(a["cmds_per_frame"].min()), "max": int(a["cmds_per_frame"].max()),
                           "held_frames": int((a["cmds_per_frame"] == 0).sum())},
        "action_sources": {str(s): int(c) for s, c in zip(src, cnt)},
    }
