"""Interactive 3D view of the cell (plotly): rotate, zoom, and toggle the grown extension.

Works in Jupyter, VS Code, Colab and in the HTML report.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .morphology import path_location
from .plotting import INK, MORPH_STYLE

OTHER = "#a9a8a2"
GROWN = "#2a78d6"  # same blue as the long morphology elsewhere
ORIGINAL = "#0b0b0b"


def _xyz(sec, x0=0.0, x1=1.0):
    """3D points of a section between normalised positions x0 and x1 (interpolated at both ends)."""
    n = sec.n3d()
    arc = np.array([sec.arc3d(i) for i in range(n)]) / sec.L
    pts = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n)])
    keep = (arc > x0) & (arc < x1)
    ends = [np.array([np.interp(x, arc, pts[:, k]) for k in range(3)]) for x in (x0, x1)]
    return np.vstack([ends[0], pts[keep], ends[1]])


def _trace(polylines, name, color, width, legendgroup=None, showlegend=True):
    """One plotly 3D line trace for a list of polylines (separated by gaps)."""
    xs, ys, zs = [], [], []
    for p in polylines:
        xs += list(p[:, 0]) + [None]
        ys += list(p[:, 1]) + [None]
        zs += list(p[:, 2]) + [None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, line=dict(color=color, width=width),
                        hoverinfo="name", legendgroup=legendgroup, showlegend=showlegend)


def morphology_3d(long_cell, short_tip_um: float, sites_um=range(50, 401, 50), height=620):
    """The long cell; target dendrite split into its original part and the grown extension.

    Hide 'grown extension' in the legend to see the short morphology.
    """
    path = set(long_cell.target_path)
    others = [_xyz(s) for s in long_cell.all if s not in path and s.n3d() > 1]
    original, grown = [], []
    for sec in long_cell.target_path:
        d0, d1 = long_cell.distance(sec(0)), long_cell.distance(sec(1))
        if d1 <= short_tip_um + 1e-6:
            original.append(_xyz(sec))
        elif d0 >= short_tip_um - 1e-6:
            grown.append(_xyz(sec))
        else:
            xs = (short_tip_um - d0) / (d1 - d0)
            original.append(_xyz(sec, 0, xs))
            grown.append(_xyz(sec, xs, 1))

    soma = _xyz(long_cell.soma[0]).mean(0)
    fig = go.Figure([
        _trace(others, "other dendrites, axon", OTHER, 2),
        _trace(original, f"target dendrite, original (0–{short_tip_um:.1f} µm)", ORIGINAL, 6),
        _trace(grown, f"grown extension ({short_tip_um:.1f}–{long_cell.tip_distance:.0f} µm)", GROWN, 6),
        go.Scatter3d(x=[soma[0]], y=[soma[1]], z=[soma[2]], mode="markers", name="soma",
                     marker=dict(size=9, color=INK), hoverinfo="name"),
    ])
    pts, labels = [], []
    for d in sites_um:
        if d <= long_cell.tip_distance + 1e-3:
            sec, x, _ = path_location(long_cell, long_cell.target_path, d)
            pts.append(_xyz(sec, 0, x)[-1])
            labels.append(f"{d:g} µm")
    if pts:
        pts = np.array(pts)
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers+text", text=labels,
                                   textposition="top center", textfont=dict(size=10, color=INK),
                                   name="distance from soma", hoverinfo="text",
                                   marker=dict(size=3, color="white", line=dict(color=INK, width=1))))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
        title=dict(text=f"{MORPH_STYLE['long']['label']} morphology — drag to rotate, scroll to zoom, "
                        "click legend entries to hide/show", font=dict(size=13)),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        scene=dict(aspectmode="data", xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="z (µm)",
                   camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9))),
    )
    return fig
