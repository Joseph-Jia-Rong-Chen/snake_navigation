"""Generate the example vessel phantoms in ./models.

Each model is plain JSON so you can hand-edit or author your own:
  - vessels[0] is the trunk; the catheter enters at its first control point
  - every vessel: name, parent, control_points [[x,y],...] (mm, y up), radius
    (scalar or one per control point). Start a branch on its parent's
    centreline to create a bifurcation.
  - targets: optional; default = closed distal ends of the tree.

Run:  python tools/make_models.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "models"


def v(name, parent, pts, radius):
    return {"name": name, "parent": parent, "control_points": pts, "radius": radius}


MODELS = {
    # Similar in spirit to a Y-shaped bench phantom: a gently curved trunk that
    # splits into two branches, each of which splits again.
    "01_y_bifurcation": {
        "description": "Y-shaped phantom: trunk -> 2 branches -> 2 sub-branches each",
        "wall_thickness": 1.5,
        "blend_radius": 2.5,
        "vessels": [
            v("trunk", None, [[20, 0], [16, 30], [6, 60], [4, 90], [10, 112]], [5.0, 5.0, 4.8, 4.6, 4.6]),
            v("left", "trunk", [[10, 112], [-6, 128], [-24, 140], [-40, 152]], [4.0, 3.9, 3.8, 3.6]),
            v("left_up", "left", [[-40, 152], [-44, 175], [-36, 196], [-22, 206]], [3.2, 3.0, 2.9, 2.8]),
            v("left_out", "left", [[-24, 140], [-46, 138], [-66, 146], [-80, 160]], [3.0, 2.9, 2.8, 2.6]),
            v("right", "trunk", [[10, 112], [26, 126], [40, 142], [52, 160]], [4.0, 3.9, 3.7, 3.6]),
            v("right_up", "right", [[52, 160], [50, 180], [56, 198], [70, 208]], [3.2, 3.0, 2.9, 2.8]),
            v("right_out", "right", [[52, 160], [72, 166], [90, 162], [104, 172]], [3.0, 2.9, 2.8, 2.6]),
        ],
    },
    # Arch-type anatomy: enter from below on the right, go up and over an arch;
    # three branches leave the top of the arch (sharp take-off angles).
    "02_arch": {
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
    # Tortuous tree with smaller vessels, a hairpin-like recurrent branch and
    # three levels of bifurcation.
    "03_tortuous_tree": {
        "description": "Tortuous trunk with 3 bifurcation levels, a recurrent (backward) branch and small calibres",
        "wall_thickness": 1.2,
        "blend_radius": 2.0,
        "vessels": [
            v("trunk", None, [[0, 0], [10, 20], [-8, 42], [8, 64], [-6, 86], [2, 104]], [4.2, 4.0, 4.0, 3.8, 3.7, 3.6]),
            v("recurrent", "trunk", [[-6, 86], [-24, 100], [-36, 90], [-38, 70], [-30, 52]], [3.0, 2.8, 2.7, 2.6, 2.5]),
            v("L1", "trunk", [[2, 104], [-12, 122], [-20, 142], [-36, 156]], [3.4, 3.2, 3.1, 3.0]),
            v("L1a", "L1", [[-20, 142], [-14, 162], [-20, 182], [-10, 198]], [2.6, 2.5, 2.4, 2.3]),
            v("L1b", "L1", [[-36, 156], [-58, 158], [-72, 172], [-70, 192]], [2.6, 2.5, 2.4, 2.3]),
            v("R1", "trunk", [[2, 104], [20, 116], [34, 134], [36, 152]], [3.4, 3.2, 3.1, 3.0]),
            v("R1a", "R1", [[36, 152], [28, 172], [38, 190], [32, 206]], [2.6, 2.5, 2.4, 2.3]),
            v("R1b", "R1", [[34, 134], [54, 136], [66, 150], [86, 152]], [2.6, 2.5, 2.4, 2.3]),
            v("R1b_x", "R1b", [[66, 150], [74, 132], [92, 124]], [2.2, 2.1, 2.0]),
        ],
    },
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, m in MODELS.items():
        data = {"name": name, "units": "mm", **m}
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))
        print("wrote", OUT / f"{name}.json")


if __name__ == "__main__":
    main()
