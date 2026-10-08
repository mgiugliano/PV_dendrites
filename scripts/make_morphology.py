"""Generate morphologies/HL23PV_dend400.swc and morphologies/target_dendrite.json.

The target dendrite is the terminal path of the original HL23PV.swc whose tip
lies closest to 100 um (path distance from the soma centre): dend[27] ->
dend[29] -> dend[30], tip at 100.8 um. Its tip is grown with the random-walk
method of the earlier 1200 um extension until the tip is 400 um from the soma.

Run once; the outputs are committed so every user gets the same morphology.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvdend import morphology as morph  # noqa: E402
from pvdend._paths import MORPH_DIR, TARGET_META  # noqa: E402
from pvdend.cell import PVCell  # noqa: E402

SHORT_TARGET_UM = 100.0  # choose the terminal path whose tip is closest to this distance
LONG_TIP_UM = 400.0  # grow it until its tip lies this far from the soma (path distance)
ORIGINAL = MORPH_DIR / "HL23PV.swc"
GROWN = MORPH_DIR / "HL23PV_dend400.swc"


def main():
    # 1. Load the original cell (no target dendrite yet) and pick the tip closest to 100 um.
    cell = PVCell(swc_path=ORIGINAL)
    tip_sec = min(morph.leaves(cell),
                  key=lambda s: abs(cell.distance(s(1)) - SHORT_TARGET_UM))
    path = morph.path_to_soma(cell, tip_sec)
    short_tip_um = cell.distance(tip_sec(1))
    short_xyz = morph.end_xyz(tip_sec)
    print("target path:", " -> ".join(s.name().split(".")[-1] for s in path),
          f"| tip at {short_tip_um:.3f} um")

    # 2. Find that tip among the SWC nodes and grow it by the missing length (random walk, constant radius).
    _, nodes = morph.read_swc(ORIGINAL)
    tip_id = morph.swc_node_at(nodes, short_xyz)
    extra = LONG_TIP_UM - short_tip_um
    long_xyz = morph.grow_swc(ORIGINAL, GROWN, tip_id, extra)

    # 3. Reload the grown cell and check the new tip distance (it should be exactly 400 um).
    long_cell = PVCell(swc_path=GROWN, tip_xyz=long_xyz)
    print(f"grown by {extra:.3f} um -> tip at {long_cell.tip_distance:.3f} um "
          f"({len(long_cell.target_path)} sections on the path)")

    # 4. Record how the target dendrite is found in each file (tip coordinates) and how it was grown.
    meta = {
        "description": "Target dendrite of HL23PV: terminal path with tip closest to 100 um",
        "distance_origin": "soma[0](0.5), path distance",
        "short": {"file": ORIGINAL.name, "tip_swc_id": tip_id,
                  "tip_xyz": short_xyz.round(6).tolist(), "tip_distance_um": round(short_tip_um, 4),
                  "path_sections": [s.name().split(".")[-1] for s in path]},
        "long": {"file": GROWN.name, "tip_xyz": long_xyz.round(6).tolist(),
                 "tip_distance_um": round(long_cell.tip_distance, 4),
                 "growth": {"from_swc_id": tip_id, "added_um": round(extra, 4),
                            "step_um": morph.GROWTH_STEP_UM, "jitter_sd": morph.GROWTH_JITTER_SD,
                            "seed": morph.GROWTH_SEED, "radius_um": nodes[tip_id]["r"]}},
    }
    TARGET_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {GROWN.name} and {TARGET_META.name}")


if __name__ == "__main__":
    main()
