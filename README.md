# 2D steerable-catheter navigation simulator

A simulated 2-DoF steerable catheter (insertion length + tip bend angle) in 2D
vessel phantoms. It's built as a set of ROS2-style nodes on a small in-process
pub/sub bus (no ROS install). Teleoperation, data collection and learned
policies all drive the robot through the same topics.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # + system ffmpeg
.venv/bin/python launch.py                                   # teleop (keyboard)
.venv/bin/python launch.py --commander policy --policy policies/example_policy.py:FeatureMapPolicy
.venv/bin/python launch.py --commander autopilot --headless --record --episodes 3 --noise 0.3
```

`teleop.py` and `collect_auto.py` are shortcuts for the first and last of these.

## Architecture

```
 viewer ──/ui/keys──> teleop ─┐
                     policy ──┼─> mux ──/robot/joint_target──> robot ──/robot/joint_state──>
                  autopilot ──┘                                  │      /sim/ground_truth
                                                                 v
              /camera/image <── camera     /nav/feature_map <── feature_map     task ──> /sim/world,
                                                                                        /nav/target
 recorder <── everything (except the UI)                                               /task/status
```

| node | role |
|---|---|
| `robot` | Takes joint targets. Clamps them to the joint limits, applies a first-order **lag (80 ms)** and a hard **speed limit** (40 mm/s, 360°/s), then solves the catheter physics. A huge target jump therefore moves smoothly. Publishes the current joints on `/robot/joint_state`. |
| `camera` | Publishes the plain frame (vessel + black catheter + red tip, no UI) on `/camera/image` at 30 fps, plus the latched `/camera/info` (size, world↔pixel transform). |
| `feature_map` | Publishes the latched U/V target maps on `/nav/feature_map`. |
| `teleop` / `policy` / `autopilot` | Commanders. Each publishes candidate targets. |
| `mux` | Forwards one commander to the robot. With a policy or the autopilot in charge, **any motion key takes over** and **G** hands control back. |
| `task` | Loads the vessel model, sets the target, reports distance and success. |
| `recorder` | Writes episodes. |
| `viewer` | The window: keyboard, HUD and overlays. It only *subscribes* to images, so nothing it draws reaches the data. |
| `supervisor` | Headless episode batches (data generation, policy evaluation). |

The bus runs at 60 Hz on a sim clock. In **realtime** mode (the default with a
window), sim time tracks wall time and a policy runs in a worker thread: the
robot keeps moving while it thinks, as on hardware. In **lockstep** mode
(`--lockstep`, the default when `--headless`), each tick waits for every node,
so runs are deterministic and faster than real time.

For a physical setup, swap `robot` and `camera` for a hardware driver and a
frame grabber that publish the same topics. Nothing else changes.

### Topics

| topic | message | notes |
|---|---|---|
| `/robot/joint_target` | `JointCommand` length [mm], bend [deg], source | the action |
| `/robot/joint_state` | `JointState` length, bend, velocities, tracked target | the observation |
| `/camera/image` | `Image` (H, W, 3) uint8 | 30 fps |
| `/camera/info` | `CameraInfo` | latched |
| `/nav/feature_map` | `FeatureMap` U, V, mask, c, target pixel | latched; republished when the target changes |
| `/nav/target`, `/sim/world`, `/task/catalog` | | latched |
| `/sim/ground_truth` | tip, shape, forces | sim-only labels, never a policy input |
| `/task/status`, `/mux/active`, `/recorder/status`, `/ui/keys`, `/episode/events` | | |

## Controls (window)

| key | action |
|---|---|
| W / S | target insertion length + / − |
| ← / → | target bend CCW / CW (− CCW, + CW) |
| Shift / Space | fast / straighten tip |
| Enter | start / stop recording |
| R / T / N / P | reset, next target, next / previous vessel |
| G | give control back to the policy after a takeover |
| F | feature map overlay: U → V → off |
| I | show the raw `/camera/image` stream |
| V / C / H / Esc | roadmap, contact forces, help, quit |

## Feature maps

For target centre (x_a, y_a), in pixel-index coordinates of the camera image:

```
U[j, i] = (i − x_a) / c · m[j, i] + (1 − m[j, i])
V[j, i] = (j − y_a) / c · m[j, i] + (1 − m[j, i])
```

* i is the pixel column (x, + right) and j the row (y, + down).
* m is the vessel mask: 1 if the pixel centre is in the lumen, 0 otherwise.
* c = max(W, H) − 1.
* Inside the vessel, U and V lie in [−1, 1]. Outside, they are exactly 1.
* Implemented in `catheter_sim/featuremap.py`.
* The policy node computes the maps directly at the policy's `input_size`
  instead of resizing them, so the vessel edge stays exact.

## Recording

**Enter** toggles recording (`--record` records every episode automatically).
Each episode writes `data/<model>/<timestamp>_<target>.{mp4,npz,json}`:

* **`.mp4`**: the plain camera stream at 30 fps. Row *k* of the `.npz` is video frame *k*.
* **`.npz`**: per camera frame:
  * `joint_*`: the observation
  * `action_*`: the target in effect when the frame was taken
  * `next_action_*`: the first target commanded after the frame. **This is the action label.**
  * `action_source`: teleop / policy / takeover / autopilot / hold
  * `keys`, `feature_map_index`, and sim-only labels

  Also stored:
  * once per target: `feature_U`, `feature_V`, `vessel_mask`, `feature_target_*`
  * a 60 Hz control stream (`ctrl_*`) plus the initial state (`init_*`), for exact re-simulation
* **`.json`**: model, targets, camera info, config, success, time to target,
  commander and policy.

The full schema is in `catheter_sim/recorder.py`.

```bash
python replay.py data/vessel_02/<episode>.npz           # view (F = feature maps)
python replay.py data/vessel_02/<episode>.npz --resim   # re-run from the logged targets (exact)
```

Training samples match what a deployed policy receives:

```python
from catheter_sim.dataset import Episode, list_episodes
ep = Episode(list_episodes()[0], size=(256, 256))
s = ep[10]   # image, feature_u, feature_v, joint, joint_vel, last_action, action, source
```

The `.mp4` is H.264 at crf 18 (near-lossless). Use `--record-images` for
bit-exact frames as well.

## Deploying a policy

```python
from catheter_sim.policy_api import Policy

class MyPolicy(Policy):
    rate_hz = 10
    input_size = (256, 256)          # image + feature maps delivered at this size
    def __init__(self, ckpt): ...
    def reset(self, info): ...       # new episode
    def act(self, obs):              # obs: image, feature_u, feature_v, joint, joint_vel, last_action, t
        return target_length_mm, target_bend_deg
```

```bash
python launch.py --commander policy --policy my_policy.py:MyPolicy --policy-arg ckpt=best.pt           # watch + take over
python launch.py --commander policy --policy my_policy.py:MyPolicy --headless --episodes 5 --models vessel_02,vessel_03   # evaluate
```

Return absolute targets. For a delta policy, return `obs["last_action"] + delta`.
`policies/example_policy.py` has three examples:

* `CreepPolicy`: a minimal template.
* `FeatureMapPolicy`: steers using only the red tip and the feature maps. It's a
  demo of the inputs, not a good controller.
* `ReplayPolicy`: replays a recording's actions open loop.

## Catheter model

* **Spline rod.** The centreline is a cubic spline through knots spaced every
  `segment_length` mm. The knots are stored as segment angles, so the rod is
  inextensible by construction. The bending energy is the discrete form of
  `∫ EI/2 (κ − κ₀)² ds`.
* **Base.** The base is fixed at the vessel entry, but its orientation is free,
  held by a sheath spring (`base_rot_stiffness`).
* **Steerable tip.** The distal `tip_length` mm is a constant-curvature arc
  whose rest curvature is set by the bend joint: bend angle = κ × `tip_length`.
* **Wall contact.** Penalty contact against the vessel's distance field, with
  incremental Coulomb friction (`friction`).
* **Solver.** Newton with line search, including negative-curvature handling,
  so the rod buckles and prolapses.

| parameter | default | note |
|---|---|---|
| `catheter.EI_shaft` / `EI_tip` | 20 / 6 N·mm² | 5 Fr Pebax-like |
| `catheter.tip_length` / `kappa_max` | 20 mm / 0.16 1/mm | up to ≈183° of bend |
| `catheter.friction` | 0.05 | |
| `actuator.time_constant` | 0.08 s | lag |
| `actuator.delay` | 0 s | optional transport delay |
| `actuator.max_insert_speed` / `max_bend_rate` | 40 mm/s / 360 °/s | speed limit |
| `camera.width` / `height` / `fps` | 900 / 900 / 30 | square, so resizing never distorts |
| `teleop.insert_speed` / `bend_rate` | 15 mm/s / 172 °/s | how fast keys move the target |
| `recording.*` | | data dir, video, lossless images |

Override any parameter in `config.json` or with `--set section.key=value`.

## Vessel models (`models/*.json`)

Every `*.json` in `models/` is loaded. `tools/make_models.py` regenerates the
examples:

* `vessel_01`: the arch
* `vessel_02`–`vessel_06`: seeded Y-bifurcation variants (`Y_VARIANTS`)

A model is a list of vessels. Each vessel has a `name`, a `parent`,
`control_points` (in mm, y up) and a `radius`. To make a bifurcation, start a
branch on its parent's centreline. Targets default to the closed distal ends.
