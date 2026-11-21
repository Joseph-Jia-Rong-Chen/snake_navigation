"""Generate the example vessel phantoms in ./models.

Each model is plain JSON so you can hand-edit or author your own:
  - vessels[0] is the trunk; the catheter enters at its first control point
  - every vessel: name, parent, control_points [[x,y],...] (mm, y up), radius
    (scalar or one per control point). Start a branch on its parent's
    centreline to create a bifurcation.
  - targets: optional; default = closed distal ends of the tree.

Models are named vessel_NN: vessel_01 is the arch (MODELS), vessel_02-24 are
Y-bifurcation variants (see Y_VARIANTS; edit the knobs or seeds and re-run to
get different ones). Every model carries a "split" (train / val / test, see
SPLITS) so training, validation and test vessels never mix.

Run:  python tools/make_models.py                    # all Y variants
      python tools/make_models.py vessel_07 vessel_01 # only these
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "models"


def v(name, parent, pts, radius):
    return {"name": name, "parent": parent, "control_points": pts, "radius": radius}


MODELS = {
    # Arch-type anatomy: enter from below on the right, go up and over an arch;
    # three branches leave the top of the arch (sharp take-off angles).
    "vessel_01": {
        "description": "Arch with three upward take-offs that bifurcate; requires a tight tip curve",
        "wall_thickness": 1.5,
        "blend_radius": 2.5,
        "vessels": [
            v("arch", None, [[50, 0], [50, 50], [48, 90], [36, 118], [10, 132], [-18, 128],
                             [-40, 110], [-48, 80], [-48, 30]], 7.0),
            v("branch_a", "arch", [[32, 121], [40, 145], [36, 170], [44, 190]], [4.5, 4.2, 4.0, 3.8]),
            v("branch_a2", "branch_a", [[38, 160], [58, 170], [72, 188]], [3.2, 3.0, 2.8]),
            v("branch_b", "arch", [[6, 132], [4, 158], [-4, 180], [0, 205]], [4.0, 3.8, 3.6, 3.4]),
            v("branch_b2", "branch_b", [[2, 168], [-18, 180], [-26, 200]], [3.0, 2.8, 2.6]),
            v("branch_c", "arch", [[-22, 126], [-40, 146], [-48, 170], [-70, 186]], [4.0, 3.8, 3.6, 3.4]),
        ],
    },
}


# ---------------------------------------------------------------------------
# Y-bifurcation variants (vessel_02..06), generated from a
# seed + "style" knobs. A candidate is rejected (and the next seed tried) if
# any two non-adjacent vessels overlap or run too close together.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

Y_VARIANTS = [
    # name, description, knobs
    ("vessel_02", "Wide symmetric Y (~50 deg take-offs), both branches split at their ends",
     dict(seed=4, split_angle=((45, 55), (45, 55)), trunk_wiggle=4, styles=("end", "end"))),
    ("vessel_03", "Asymmetric Y: shallow left / steep right take-off; left has a side branch",
     dict(seed=5, split_angle=((20, 28), (55, 65)), trunk_wiggle=6, styles=("side", "end"))),
    ("vessel_04", "Tortuous trunk before the split; mixed end / side sub-branches",
     dict(seed=6, split_angle=((30, 45), (30, 45)), trunk_wiggle=22, trunk_len=(120, 135), styles=("end", "side"))),
    ("vessel_05", "Long trunk with small-calibre branches and side branches on both sides",
     dict(seed=7, split_angle=((35, 45), (35, 45)), trunk_wiggle=8, trunk_len=(130, 140),
          radius_scale=0.8, styles=("side", "side"))),
    ("vessel_06", "Trunk entering at an angle; rotated split with a backward-pointing sub-branch",
     dict(seed=8, split_angle=((30, 40), (40, 55)), trunk_wiggle=8, trunk_heading=68,
          styles=("end", "side"), back_branch=True)),
    # ---- vessel_07..24
    ("vessel_07", "Narrow symmetric Y (~26 deg take-offs), both branches split at their ends",
     dict(seed=9, split_angle=((22, 30), (22, 30)), trunk_wiggle=5, styles=("end", "end"))),
    ("vessel_08", "Very wide Y (~65 deg take-offs) with side branches on both sides",
     dict(seed=10, split_angle=((60, 70), (60, 70)), trunk_wiggle=6, styles=("side", "side"))),
    ("vessel_09", "Asymmetric Y: steep left / shallow right take-off; right has a side branch",
     dict(seed=11, split_angle=((55, 65), (20, 28)), trunk_wiggle=6, styles=("end", "side"))),
    ("vessel_10", "Tortuous trunk, symmetric split, both branches split at their ends",
     dict(seed=12, split_angle=((35, 45), (35, 45)), trunk_wiggle=20, trunk_len=(120, 135),
          styles=("end", "end"))),
    ("vessel_11", "Trunk leaning left; left side branch, right end split",
     dict(seed=13, split_angle=((30, 40), (40, 55)), trunk_wiggle=8, trunk_heading=112,
          styles=("side", "end"))),
    ("vessel_12", "Large-calibre vessels, short trunk",
     dict(seed=14, split_angle=((35, 50), (35, 50)), trunk_wiggle=6, trunk_len=(90, 100),
          radius_scale=1.15, styles=("end", "side"))),
    ("vessel_13", "Small-calibre vessels, wide split",
     dict(seed=15, split_angle=((45, 55), (45, 55)), trunk_wiggle=7, radius_scale=0.75,
          styles=("side", "end"))),
    ("vessel_14", "Moderately tortuous trunk; backward-pointing right side branch",
     dict(seed=16, split_angle=((35, 45), (35, 45)), trunk_wiggle=12, styles=("end", "side"),
          back_branch=True)),
    ("vessel_15", "Trunk leaning right; shallow left / steep right take-off",
     dict(seed=17, split_angle=((25, 35), (50, 60)), trunk_wiggle=6, trunk_heading=70,
          styles=("end", "end"))),
    ("vessel_16", "Long tortuous thin trunk; side branches on both sides",
     dict(seed=18, split_angle=((35, 45), (35, 45)), trunk_wiggle=16, trunk_len=(130, 140),
          radius_scale=0.85, styles=("side", "side"))),
    ("vessel_17", "Shallow split (~21 deg) after a long trunk",
     dict(seed=19, split_angle=((18, 24), (18, 24)), trunk_wiggle=5, trunk_len=(110, 125),
          styles=("end", "side"))),
    ("vessel_18", "Asymmetric split; backward-pointing right side branch",
     dict(seed=20, split_angle=((40, 50), (25, 35)), trunk_wiggle=10, styles=("side", "side"),
          back_branch=True)),
    ("vessel_19", "Moderate symmetric Y; right side branch",
     dict(seed=21, split_angle=((38, 48), (38, 48)), trunk_wiggle=7, styles=("end", "side"))),
    ("vessel_20", "Tortuous trunk leaning left; left side branch, right end split",
     dict(seed=22, split_angle=((30, 45), (45, 60)), trunk_wiggle=15, trunk_heading=105,
          styles=("side", "end"))),
    ("vessel_21", "Thin vessels; backward-pointing right side branch",
     dict(seed=23, split_angle=((40, 50), (35, 45)), trunk_wiggle=8, radius_scale=0.8,
          styles=("end", "side"), back_branch=True)),
    ("vessel_22", "Tortuous trunk; shallow left / steep right take-off, end splits",
     dict(seed=24, split_angle=((25, 35), (55, 70)), trunk_wiggle=12, styles=("end", "end"))),
    ("vessel_23", "Large calibre, trunk leaning right; side branches on both sides",
     dict(seed=25, split_angle=((35, 50), (35, 50)), trunk_wiggle=7, trunk_heading=75,
          radius_scale=1.1, styles=("side", "side"))),
    ("vessel_24", "Narrow split after a long trunk; backward-pointing right side branch",
     dict(seed=26, split_angle=((22, 30), (30, 40)), trunk_wiggle=9, trunk_len=(125, 140),
          styles=("side", "side"), back_branch=True)),
]

# Held-out vessels: never used for training. "val" for model selection /
# hyperparameter tuning, "test" for the final evaluation. Everything else is "train".
SPLITS = {"vessel_21": "val", "vessel_22": "val",
          **{f"vessel_{n:02d}": "test" for n in range(4, 10)}}


def _arc(start, heading, length, turn, n, wiggle, rng):
    """Control points along a path whose heading changes by `turn` (deg) overall
    plus random wiggle; returns (points, end_heading)."""
    pts, h, p = [list(map(float, start))], heading, np.array(start, float)
    for k in range(n - 1):
        h_seg = h + turn / (n - 1) * 0.5 + rng.uniform(-wiggle, wiggle)
        p = p + length / (n - 1) * np.array([np.cos(np.radians(h_seg)), np.sin(np.radians(h_seg))])
        pts.append([round(p[0], 1), round(p[1], 1)])
        h = h + turn / (n - 1)
    return pts, h


def _taper(r0, r1, n):
    return [round(float(r), 2) for r in np.linspace(r0, r1, n)]


def _y_candidate(rng, k):
    U = rng.uniform
    rs = k.get("radius_scale", 1.0)
    vs = []
    trunk, h = _arc([0, 0], k.get("trunk_heading", 90) + U(-5, 5), U(*k.get("trunk_len", (100, 120))),
                    U(-15, 15), 6, k["trunk_wiggle"], rng)
    rt = U(4.7, 5.3) * rs
    vs.append(v("trunk", None, trunk, _taper(rt, rt * 0.92, 6)))
    B = trunk[-1]
    for side, sgn in (("left", 1), ("right", -1)):
        ang = U(*k["split_angle"][0 if side == "left" else 1])
        rb = rt * U(0.76, 0.84)
        br, hb = _arc(B, h + sgn * ang, U(50, 68), sgn * U(-5, 20), 4, 4, rng)
        vs.append(v(side, "trunk", br, _taper(rb, rb * 0.92, 4)))
        rc = rb * U(0.75, 0.85)
        style = k["styles"][0 if side == "left" else 1]
        if style == "end":
            a_in, a_out = U(20, 40), U(30, 55)
            up, _ = _arc(br[-1], hb - sgn * a_in, U(35, 50), -sgn * U(0, 25), 4, 5, rng)
            out, _ = _arc(br[-1], hb + sgn * a_out, U(35, 50), sgn * U(-15, 20), 4, 5, rng)
            vs.append(v(f"{side}_up", side, up, _taper(rc, rc * 0.9, 4)))
            vs.append(v(f"{side}_out", side, out, _taper(rc * 0.95, rc * 0.85, 4)))
        else:  # side branch from the middle + continuation from the end
            mid = br[1] if U() < 0.5 else br[2]
            a = U(95, 115) if (k.get("back_branch") and side == "right") else U(45, 75)
            sideb, _ = _arc(mid, hb + sgn * a, U(32, 45), sgn * U(-10, 20), 4, 5, rng)
            cont, _ = _arc(br[-1], hb + U(-10, 10), U(35, 50), sgn * U(-20, 20), 4, 5, rng)
            vs.append(v(f"{side}_side", side, sideb, _taper(rc * 0.9, rc * 0.8, 4)))
            vs.append(v(f"{side}_up", side, cont, _taper(rc, rc * 0.9, 4)))
    return vs


def _y_valid(vessels, gap=3.0, junction=14.0, max_span=240.0):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from catheter_sim.vessel import smooth_centerline
    sm = [smooth_centerline(np.array(x["control_points"]), x["radius"]) for x in vessels]
    allp = np.vstack([p for p, _ in sm])
    if np.ptp(allp, 0).max() > max_span:
        return False
    starts = [p[0] for p, _ in sm]
    for i in range(len(sm)):
        for j in range(i + 1, len(sm)):
            (pa, ra), (pb, rb) = sm[i], sm[j]
            keep_a = np.ones(len(pa), bool)
            keep_b = np.ones(len(pb), bool)
            for s in (starts[i], starts[j]):     # ignore the shared junctions
                keep_a &= np.linalg.norm(pa - s, axis=1) > junction
                keep_b &= np.linalg.norm(pb - s, axis=1) > junction
            if not keep_a.any() or not keep_b.any():
                continue
            d = np.linalg.norm(pa[keep_a][:, None] - pb[keep_b][None], axis=2)
            need = ra[keep_a][:, None] + rb[keep_b][None] + gap
            if (d < need).any():
                return False
    return True


def y_variant(knobs):
    seed = knobs["seed"]
    for attempt in range(500):
        rng = np.random.default_rng(seed * 1000 + attempt)
        vs = _y_candidate(rng, knobs)
        if _y_valid(vs):
            return vs, seed * 1000 + attempt
    raise RuntimeError(f"no valid candidate for knobs {knobs}")


def main(names=None):
    OUT.mkdir(exist_ok=True)
    names = set(names or [n for n, *_ in Y_VARIANTS])
    for name, m in MODELS.items():
        if name not in names:
            continue
        data = {"name": name, "units": "mm", "split": SPLITS.get(name, "train"), **m}
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))
        print("wrote", OUT / f"{name}.json")
    for name, desc, knobs in Y_VARIANTS:
        if name not in names:
            continue
        vessels, rng_seed = y_variant(knobs)
        data = {"name": name, "units": "mm", "description": desc, "split": SPLITS.get(name, "train"),
                "generator_seed": rng_seed,
                "wall_thickness": 1.5, "blend_radius": 2.5, "vessels": vessels}
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))
        print("wrote", OUT / f"{name}.json", "(seed", rng_seed, ")")


if __name__ == "__main__":
    main(sys.argv[1:])
