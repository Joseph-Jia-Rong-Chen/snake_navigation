# 2D steerable-catheter navigation simulator

Keyboard teleoperation and data collection for a 2-DoF steerable catheter
(insertion length + tip curvature) inside 2D vessel phantoms.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python teleop.py                      # first model in models/
.venv/bin/python teleop.py --model vessel_01      # pick a model
.venv/bin/python teleop.py --set catheter.EI_shaft=30 --set catheter.friction=0.1
```

## Controls

| key | action |
|---|---|
| W / S | insert / retract (insertion length, + = deeper) |
| ← / → | tip curvature CCW / CW (κ < 0 CCW, κ > 0 CW) |
| Shift | fast (×3) |
| Space | straighten tip (κ → 0) |
| Enter | start / stop recording |
| R | reset episode (saves a running recording) |
| N / P | next / previous vessel model |
| T | cycle target |
| V | toggle vessel roadmap (off = no contrast, only the catheter is visible) |
| C | show contact (yellow) and friction (cyan) forces |
| H | toggle help · Esc quit |

The catheter is drawn black (radiopaque) and the tip has a red fiducial.

## Catheter model

* **Spline rod.** The centreline is a cubic spline through knots spaced every
  `segment_length` mm. The knots are the degrees of freedom, stored as
  segment angles, so the rod is inextensible by construction. The bending
  energy is the discrete form of `∫ EI/2 (κ − κ₀)² ds`.
* **Base.** The base is fixed at the vessel entry. Its orientation is free,
  held by a rotational spring `base_rot_stiffness` that models a sheath (set
  it to 0 for a free pivot). Insertion is kinematic: material is fed through
  the entry point.
* **Steerable tip.** The distal `tip_length` mm is a constant-curvature arc
  whose rest curvature is the commanded κ. The rest of the shaft has κ₀ = 0,
  so the rest shape is piecewise constant. The tip can have its own stiffness
  (`EI_tip`).
* **Wall contact.** The vessel lumen is a signed distance field. Contact uses
  a penalty (`wall_stiffness`), and Coulomb friction (`friction`) is applied
  incrementally with a lagged normal force (IPC-style).
* **Solver.** Every frame is a quasi-static energy minimisation, solved with
  Newton plus line search, with negative-curvature handling so the rod
  buckles and prolapses when pushed against an obstacle. A step takes about
  2–10 ms.

### Parameters (`config.json`, or `--set section.key=value`)

| parameter | default | note |
|---|---|---|
| `catheter.EI_shaft` | 20 N·mm² | 5 Fr Pebax-like shaft (E ≈ 50 MPa, d = 1.67 mm) |
| `catheter.EI_tip` | 6 N·mm² | softer steerable section |
| `catheter.tip_length` | 20 mm | bending-section arc length |
| `catheter.kappa_max` | 0.16 1/mm | min radius 6.25 mm, ≈180° max deflection |
| `catheter.friction` | 0.05 | hydrophilic-coated catheter on a smooth wall |
| `catheter.diameter` | 1.67 mm | 5 Fr |
| `catheter.base_rot_stiffness` | 10 N·mm/rad | sheath at the entry |
| `catheter.wall_stiffness` | 40 N/mm | penalty stiffness (higher means less penetration but stiffer solves) |
| `teleop.insert_speed` | 15 mm/s | |
| `teleop.kappa_rate` | 0.15 1/mm/s | |

Only stiffness ratios change the shape, because the walls are almost rigid
and insertion is kinematic. Absolute `EI` scales the reported forces.

The tip defaults were checked with the scripted demonstrator, which follows
the centreline with pure-pursuit steering:

| tip setting | example targets reached |
|---|---|
| 20 mm / 0.16 (default) | all targets in every model |
| 15 mm / 0.125 | misses hairpin take-offs (branches that turn back > ~110°) |

## Vessel models (`models/*.json`)

| model | description |
|---|---|
| `vessel_01` | arch with three upward take-offs, two of which bifurcate |
| `vessel_02` | wide symmetric Y (~50° take-offs); both branches split at their ends |
| `vessel_03` | asymmetric Y: shallow left / steep right take-off; side branch on the left |
| `vessel_04` | tortuous trunk before the split; mixed end / side sub-branches |
| `vessel_05` | long trunk, small-calibre branches, side branches on both sides |
| `vessel_06` | trunk entering at an angle; one sub-branch points backward |

A model is a list of vessels. Each vessel has a `name`, a `parent`,
`control_points` (in mm, y up) and a `radius` (one value, or one per control
point). Centrelines are cubic-spline interpolated. To make a bifurcation,
start a branch on its parent's centreline. The catheter enters at the first
point of `vessels[0]`.

Targets are optional. By default, every closed distal end is a target. Add new
models directly as JSON, or edit and re-run `tools/make_models.py`.
`vessel_02`–`vessel_06` come from a seeded generator (`Y_VARIANTS`): change
the knobs or seeds to get new variants. Every
`*.json` in `models/` shows up in the app (N / P keys).

## Data collection

Press **Enter** to start or stop recording. Episodes are saved to
`data/<model>/<timestamp>_<target>.npz`, with a `.json` metadata sidecar
(model, target, full config, success, time-to-target).

Per-frame arrays:

* `keys` [W, S, L, R] and `action` (Δlength, Δκ)
* `length`, `kappa_cmd`, `kappa_meas`
* `tip_pos`, `tip_angle`, `base_angle`
* `shape`: the spline resampled to 200 points, base to tip
* contact and insertion forces
* `target_dist`, `success`
* `images`: fluoroscopy frames (256×256 RGB), only with `--record-images`

The full schema is in `catheter_sim/recorder.py`.

Recordings also store the initial solver state, so they re-simulate exactly:

```bash
python replay.py data/vessel_01/<episode>.npz           # visual replay (Space/←/→/+/-)
python replay.py data/vessel_01/<episode>.npz --resim   # re-run physics from logged actions
python collect_auto.py --episodes 5 --noise 0.3        # scripted demonstrations, same format
```

For scripted control, policies or RL, use `catheter_sim.env.CatheterEnv`:

```python
from catheter_sim.config import load_config
from catheter_sim.env import CatheterEnv
env = CatheterEnv(load_config("config.json"), "models/vessel_01.json")
obs = env.reset()
obs = env.step(d_length=0.25, d_kappa=0.0)   # per-step increments [mm], [1/mm]
```
