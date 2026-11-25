# Pipeline cheat sheet

Run every command from the project root (`~/dev/snake`) with `.venv/bin/python`
(or activate the venv and use `python3`). Each section gives the command you
normally run, then the options worth knowing. Every script also takes `--help`.

```
0 setup → 1 vessels & splits → 2 collect → 3 inspect → 4 convert → 5 check → 6 train → 7 evaluate → 8 videos → 9 improve
```

Current splits: **train** `vessel_02, 03, 10–20, 23, 24` · **val** `vessel_21, 22` · **test** `vessel_04–09`

---

## 0. Setup (once)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python lerobot==0.4.4      # for steps 4–8
```
`ffmpeg` must be on the PATH.

Settings in **`config.json`** must stay the same for every episode that goes
into one dataset:

| setting | current | meaning |
|---|---|---|
| `camera.width` / `height` / `fps` | 1280 / 720 / 30 | camera image = the scene area of the window |
| `recording.video_scale` | 0.5 | recorded video size = 640×360 |
| `recording.stop_on_success` | false | true = a recording stops automatically when the tip reaches the target |
| `actuator.delay` / `time_constant` | 0.08 / 0.08 s | robot latency and lag |
| `teleop.insert_speed` / `bend_rate` | 15 mm/s / 172 °/s | how fast the keys move the target |

To change a setting for one run only: `--set section.key=value`, e.g. `--set recording.stop_on_success=true`.

---

## 1. Vessels and splits

```bash
.venv/bin/python tools/make_models.py                      # regenerate all vessel models (vessel_02–24)
.venv/bin/python tools/make_models.py vessel_07 vessel_08  # only these
```
- **Where the split lives:** each `models/vessel_NN.json` has `"split": "train" | "val" | "test"`.
- **To change splits:** edit `SPLITS` in `tools/make_models.py` and regenerate, or edit the `"split"` field directly.
- **Recordings follow automatically:** each recording uses its vessel's *current* split at conversion time, so recordings never need changing.

---

## 2. Collect data → `data/raw/<vessel>/<time>_<target>/`

**Recording is manual:** **Enter** starts a recording and **Enter** ends it. Nothing else does:
- **Handing control** back and forth with a policy doesn't split it.
- **Reaching the target** doesn't end it; stay as long as you like, then press Enter.
- **Exceptions:** R (reset), N / P (change vessel) and Esc (quit) save and close the recording, because the scene changes.

### Teleop (you drive)
```bash
.venv/bin/python launch.py --split train
```

| option | what it does | example |
|---|---|---|
| `--model NAME` | start on one vessel | `--model vessel_12` |
| `--split S` | **N** / **P** only cycle through this split's vessels | `--split train` |
| `--target NAME` | start on this target | `--target left_up` |

**Keys:**
- **Driving:** W / S insert / retract, ← / → bend, Shift fast, Space straighten.
- **Session:** Enter start / stop recording, R reset, T next target, N / P vessel, F feature map, H help, Esc quit.


### Teleop with a policy on standby (you choose when it drives)
```bash
.venv/bin/python launch.py --split train --policy training/act_policy.py:ACTSimPolicy \
    --policy-arg checkpoint=outputs/act_catheter/best
```
- **You drive first.**
- **G** hands control to the policy.
- **Any motion key** (W / S / ← / →) takes control back, as often as you like.
- **The panel shows who's driving:** `driver you (G: policy)` or `driver policy (keys: take over)`.

| option | what it does | example |
|---|---|---|
| `--policy FILE:CLASS` | which policy is on standby | `training/act_policy.py:ACTSimPolicy` |
| `--policy-arg checkpoint=DIR` | which trained model it loads | `checkpoint=outputs/act_v2/best` |
| `--policy-arg n_action_steps=N` | ACT: execute fewer actions per chunk | `n_action_steps=30` |
| `--commander policy` | let the **policy** drive first instead (you take over, G hands back) | |

**Per-frame labels:** each frame records who was driving: `teleop` (you) or `policy`. In a `--commander policy` session, your driving is labelled `takeover`.
- **Check a recording:** `replay.py <episode> --sync` shows the mix.
- **Watch out:** frames the policy drove are converted as training labels too, unless you keep such recordings out of the dataset (e.g. `--sources teleop` in step 4, which drops whole mixed recordings).

### Autopilot demonstrations (headless, fast)
```bash
.venv/bin/python collect_auto.py --split train --episodes 3 --noise 0.3
```
Automatic batch: one recording per attempt. Each stops 0.5 s after the target is reached, or on timeout.

| option | what it does | example |
|---|---|---|
| `--episodes N` | attempts per target | `--episodes 3` |
| `--noise 0..1` | imperfect-operator randomness; varies the demonstrations | `--noise 0.3` |
| `--seed N` | different random runs | `--seed 1` |
| `--models A,B` | instead of `--split` | `--models vessel_10,vessel_11` |

---

## 3. Inspect recordings

```bash
.venv/bin/python replay.py data/raw/vessel_12/20261121_101500_left_up            # watch it
.venv/bin/python replay.py data/raw/vessel_12/20261121_101500_left_up --sync     # timing / alignment report
.venv/bin/python replay.py data/raw/vessel_12/20261121_101500_left_up --resim    # physics re-run, should be 0 error
```
- **Replay keys:** Space pause, ← / → step, + / − speed, F feature map, I recorded video, V roadmap.
- **A good `--sync` report:** `gaps: []` and `state_lag_ms` 0. One `hold` frame at the end is normal.

---

## 4. Convert to LeRobot datasets → `data/lerobot/<name>/`

```bash
.venv/bin/python tools/convert_lerobot.py data/raw --out data/lerobot/catheter_nav_train --split train --overwrite
.venv/bin/python tools/convert_lerobot.py data/raw --out data/lerobot/catheter_nav_val   --split val   --overwrite
```

| option | what it does | example |
|---|---|---|
| `--split S` | only episodes on this split's vessels | `--split train` |
| `--overwrite` | replace an existing dataset (needed after recording more) | |
| `--dry-run` | only list what would be included | |
| `--only-success` | drop episodes that never reached the target | |
| `--sources A,B` | only episodes driven entirely by these | `--sources teleop` |
| `--models A,B` | only these vessels | `--models vessel_10,vessel_12` |
| `--scale X` | resize vs. the 640×360 video; maps are recomputed | `--scale 0.5` → 320×180, ~4× faster training |
| `--hold-end S` | seconds of "arrived, stay still" added to episodes that end in the target (default 2) | `--hold-end 0` to disable |

**Output:**
- **Data:** `observation.state`, `action`, `observation.images.cam`, `x_goal_map`, `y_goal_map`, `task`.
- **Report:** a sync report in `meta/sync_report.json`.

**Your raw recordings are never modified.** Re-convert any time.

---

## 5. Check the dataset works with ACT (seconds)

```bash
.venv/bin/python training/check_act.py --root data/lerobot/catheter_nav_train
```
It should end with `COMPATIBLE`. Run it after changing the converter or the dataset settings.

---

## 6. Train ACT → `outputs/<name>/`

```bash
.venv/bin/python training/training.py --root data/lerobot/catheter_nav_train \
    --val-root data/lerobot/catheter_nav_val --out outputs/act_catheter --epochs 30 --patience 5
```

| option | what it does | example |
|---|---|---|
| `--root` / `--val-root` | training / validation dataset | |
| `--out DIR` | where checkpoints go; use a new name per experiment | `--out outputs/act_v2` |
| `--epochs N` | length in passes over the data (overrides `--steps`) | `--epochs 30` |
| `--steps N` | length in weight updates (default 5000) | `--steps 20000` |
| `--patience N` | stop after N validations without improvement | `--patience 5` |
| `--chunk-size N` | actions predicted per frame (default 60 = 2 s) | `--chunk-size 60` |
| `--n-action-steps N` | actions executed before predicting again (default 60 = whole chunk) | `--n-action-steps 30` |
| `--batch-size N` / `--lr X` | batch size (16) / learning rate (1e-5) | `--batch-size 8` if GPU memory is short |
| `--absolute` | absolute joint targets instead of relative (default: relative to the current joints) | |
| `--eval-freq N` / `--save-freq N` | validate / save every N steps (500 / 1000) | |

- **Progress:**
  - every 50 steps: `step … epoch … loss … l1 … kld`
  - every 500 steps: `val l1 … = … mm, … deg`, marked `(best)` or `(no improvement Nx)`
- **Checkpoints:** `best/` (lowest val error; only with `--val-root`), `last/`, `step_NNNNNN/`.
- **Stopping:** Ctrl+C stops cleanly and saves `last/`.
- **Running in the background:**
  ```bash
  nohup .venv/bin/python training/training.py ... > train.log 2>&1 &
  tail -f train.log
  ```
- **GPU memory:** only one training run fits on the GPU at a time.

---

## 7. Evaluate in the simulator

### Watch it drive (window)
```bash
.venv/bin/python launch.py --commander policy --policy training/act_policy.py:ACTSimPolicy \
    --policy-arg checkpoint=outputs/act_catheter/best --model vessel_21
```
- T next target, R reset, N / P vessel.
- Any motion key takes over; G hands control back.

### Score it
```bash
.venv/bin/python launch.py --commander policy --policy training/act_policy.py:ACTSimPolicy \
    --policy-arg checkpoint=outputs/act_catheter/best --headless --split val --episodes 1
```

| option | what it does | example |
|---|---|---|
| `--policy-arg checkpoint=DIR` | which model | `checkpoint=outputs/act_catheter/last` |
| `--policy-arg n_action_steps=N` | execute fewer actions per chunk at test time (no retraining) | `n_action_steps=30` |
| `--split val` / `--split test` | which vessels: val while tuning, **test only once at the end** | |
| `--episodes N` | attempts per target. Headless runs are deterministic, so 1 is enough. | `--episodes 1` |
| `--max-time S` | timeout per attempt (default 90 s) | `--max-time 60` |
| `--headless` | no window, runs as fast as possible | |
| `--realtime` | real-time pacing even when headless (realistic inference timing) | |

**Output:** an `OK` / `FAIL` line per target (with time and final distance), then a success rate.

---

## 8. Record the evaluation as videos (with target overlay)

```bash
.venv/bin/python launch.py --commander policy --policy training/act_policy.py:ACTSimPolicy --policy-arg checkpoint=outputs/act_catheter/best --split val --episodes 1 --headless --record --set recording.data_dir=data/eval/run1 && .venv/bin/python tools/render_video.py data/eval/run1 --out videos/run1
```

| option (`render_video.py`) | what it does | example |
|---|---|---|
| `--out DIR` | all videos into one folder | `--out videos/run1` |
| *(no `--out`)* | `overlay.mp4` inside each episode folder | |
| `--size video` | render at 640×360 instead of 1280×720 | |
| `--no-trail` | hide the tip path | |

- **Separate folder:** `--set recording.data_dir=data/eval/<name>` keeps evaluation runs **out of `data/raw`**, so they never end up in training data.
- **New name per run:** use a new `<name>` per evaluation (`run1`, `run2`, …).
- **Any recording works:** `render_video.py` also renders your teleop recordings, e.g. `data/raw/vessel_12`.

---

## 9. Improve the policy (next round)

1. **Find the failures:** run step 8 on val and watch the `FAIL` videos. Also score a few train vessels: failing there means under-trained, not a generalisation problem.
2. **Collect more data on *train* vessels only:**
   - targeted teleop demos of the hard situations (step 2, teleop), or
   - corrections: let the standby policy drive (G), take over when it goes wrong (step 2, teleop with a policy on standby)
3. **Rebuild both datasets with `--overwrite`** (step 4).
4. **Retrain on all the data, old and new,** into a new `--out` (step 6).
5. **Compare on val** (step 7). When you're done tuning, score the final model **once** with `--split test`.

---

## Where things live

| path | what |
|---|---|
| `config.json` | all simulator / camera / recording settings |
| `models/vessel_NN.json` | vessel geometry, targets, split |
| `data/raw/<vessel>/<episode>/` | raw recordings: `video.mp4`, `streams.npz`, `meta.json` |
| `data/lerobot/<name>/` | converted LeRobot datasets |
| `data/eval/<run>/` | evaluation recordings |
| `outputs/<name>/{best,last,step_N}/` | trained checkpoints |
| `videos/<run>/` | rendered overlay videos |
| `README.md` | full documentation and code architecture |
