"""Morphology tools: the target dendrite, path distances, and growing a tip.

Distances are path distances from the centre of the soma, soma[0](0.5), the
same origin used by the original model code.
"""
from __future__ import annotations

import json

import numpy as np
from neuron import h

from ._paths import MORPH_DIR, TARGET_META

GROWTH_STEP_UM = 5.0  # random-walk step, as in the earlier 1200 um extension
GROWTH_JITTER_SD = 0.1  # s.d. of the direction perturbation per step
GROWTH_SEED = 42


def load_target_meta() -> dict:
    """Morphology files and target-tip coordinates written by scripts/make_morphology.py."""
    if not TARGET_META.exists():
        raise FileNotFoundError(
            f"{TARGET_META} not found. Run `python scripts/make_morphology.py` first."
        )
    return json.loads(TARGET_META.read_text())


def morphology_file(morph: str) -> str:
    return str(MORPH_DIR / load_target_meta()[morph]["file"])


def soma_distance(cell, seg) -> float:
    return h.distance(cell.soma[0](0.5), seg)


def end_xyz(sec) -> np.ndarray:
    n = sec.n3d() - 1
    return np.array([sec.x3d(n), sec.y3d(n), sec.z3d(n)])


def children(sec) -> list:
    return list(h.SectionRef(sec=sec).child)


def parent(sec):
    ref = h.SectionRef(sec=sec)
    return ref.parent if ref.has_parent() else None


def leaves(cell) -> list:
    return [sec for sec in cell.basal if not children(sec)]


def path_to_soma(cell, tip_sec) -> list:
    """Sections from the primary dendrite to `tip_sec` (soma excluded)."""
    path = []
    sec = tip_sec
    while sec is not None and sec != cell.soma[0]:
        path.append(sec)
        sec = parent(sec)
    return path[::-1]


def find_tip_section(cell, tip_xyz) -> object:
    """The terminal basal section whose distal end is closest to `tip_xyz`."""
    tip_xyz = np.asarray(tip_xyz)
    dists = [(np.linalg.norm(end_xyz(s) - tip_xyz), i, s) for i, s in enumerate(leaves(cell))]
    d, _, sec = min(dists, key=lambda x: (x[0], x[1]))
    if d > 1.0:
        raise RuntimeError(f"No dendritic tip within 1 um of {tip_xyz} (closest: {d:.2f} um)")
    return sec


def side_branches(cell, path) -> list:
    """Sections that leave `path` at one of its branch points (roots of side subtrees)."""
    on_path = set(path)
    return [c for sec in path for c in children(sec) if c not in on_path]


def subtree(sec) -> list:
    out = [sec]
    for c in children(sec):
        out.extend(subtree(c))
    return out


def path_location(cell, path, distance_um):
    """(section, x, actual_distance) of the point on `path` at `distance_um` from the soma.

    `actual_distance` is the distance of the segment centre that a point process
    placed there is attached to.
    """
    for sec in path:
        d0, d1 = soma_distance(cell, sec(0)), soma_distance(cell, sec(1))
        if d0 - 1e-3 <= distance_um <= d1 + 1e-3:
            x = min(max((distance_um - d0) / (d1 - d0), 0.0), 1.0)
            seg = sec(x)
            return sec, x, soma_distance(cell, seg)
    raise ValueError(
        f"{distance_um} um is beyond the target path (tip at {soma_distance(cell, path[-1](1)):.1f} um)"
    )


# --- SWC growth -----------------------------------------------------------------

def read_swc(path) -> tuple[list[str], dict]:
    lines, nodes = [], {}
    with open(path) as f:
        for line in f:
            lines.append(line)
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            nodes[int(p[0])] = dict(
                type=int(p[1]), xyz=np.array([float(v) for v in p[2:5]]), r=float(p[5]), parent=int(p[6])
            )
    return lines, nodes


def swc_node_at(nodes: dict, xyz, node_type=3) -> int:
    ids = [i for i, n in nodes.items() if n["type"] == node_type]
    d = [np.linalg.norm(nodes[i]["xyz"] - np.asarray(xyz)) for i in ids]
    k = int(np.argmin(d))
    if d[k] > 1e-3:
        raise RuntimeError(f"No SWC node at {xyz} (closest {d[k]:.4f} um)")
    return ids[k]


def grow_swc(in_path, out_path, tip_id: int, extra_um: float,
             step_um=GROWTH_STEP_UM, jitter_sd=GROWTH_JITTER_SD, seed=GROWTH_SEED) -> np.ndarray:
    """Append an unbranched random-walk extension of length `extra_um` to node `tip_id`.

    Same method as the earlier 1200 um extension (create_extended_morphology.py):
    the walk starts along the parent->tip direction, each step of `step_um`
    perturbs the direction with N(0, jitter_sd) noise, and the radius stays
    equal to the tip radius. Returns the xyz of the new tip.
    """
    lines, nodes = read_swc(in_path)
    tip = nodes[tip_id]
    direction = tip["xyz"] - nodes[tip["parent"]]["xyz"]
    direction /= np.linalg.norm(direction)
    rng = np.random.RandomState(seed)  # legacy generator, as in the earlier script

    n_full = int(extra_um // step_um)
    steps = [step_um] * n_full
    if extra_um - n_full * step_um > 1e-6:
        steps.append(extra_um - n_full * step_um)

    pos, parent_id, new_id = tip["xyz"].copy(), tip_id, max(nodes) + 1
    new_lines = []
    for s in steps:
        direction = direction + rng.normal(0.0, jitter_sd, 3)
        direction /= np.linalg.norm(direction)
        pos = pos + direction * s
        new_lines.append(
            f"{new_id} 3 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {tip['r']:.4f} {parent_id}\n"
        )
        parent_id, new_id = new_id, new_id + 1

    with open(out_path, "w") as f:
        f.writelines(lines)
        f.writelines(new_lines)
    return pos


def path_geometry(cell, reference=None) -> list[dict]:
    """One row per section of the target path: position, length, diameters, membrane area.

    `reference` (another cell, e.g. the short one) is used to label sections that
    exist only in `cell` as the grown extension.
    """
    ref_names = {s.name().split(".")[-1] for s in reference.target_path} if reference else None
    ref_tip = reference.target_path[-1].name().split(".")[-1] if reference else None
    path = cell.target_path
    rows = []
    for i, sec in enumerate(path):
        name = sec.name().split(".")[-1]
        side = [c.name().split(".")[-1] for c in children(sec) if c not in set(path)]
        if ref_names is not None and name not in ref_names:
            portion = "grown extension"
        elif i == len(path) - 1:
            portion = "terminal branch, to the tip"
        elif name == ref_tip:
            portion = "terminal branch, to the original tip"
        elif side:
            portion = f"to branch point ({', '.join(side)} leaves)"
        else:
            portion = "unbranched"
        d3 = [sec.diam3d(k) for k in range(sec.n3d())]
        rows.append(dict(section=name, portion=portion,
                         start_um=soma_distance(cell, sec(0)), end_um=soma_distance(cell, sec(1)),
                         length_um=sec.L, diam_mean_um=float(np.mean([s.diam for s in sec])),
                         diam_min_um=min(d3), diam_max_um=max(d3),
                         area_um2=sum(s.area() for s in sec), nseg=sec.nseg))
    return rows
