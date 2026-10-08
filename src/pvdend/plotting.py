"""Publication-quality figures. The notebook and scripts/ use exactly these functions."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ._paths import FIGURES_DIR

MM = 1 / 25.4
ONE_COL, TWO_COL = 89 * MM, 183 * MM  # Nature column widths

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
MORPH_STYLE = {
    "short": dict(color=INK_2, ls="--", label="short (100 µm)"),
    "long": dict(color="#2a78d6", ls="-", label="long (400 µm)"),
}
# categorical slots in fixed order; 'none' is the neutral reference
PROFILE_COLORS = {"none": INK_2, "uniform": "#2a78d6", "hotspot": "#eb6834",
                  "increasing": "#1baf7a", "decreasing": "#eda100"}
SITE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("site", ["#b9d3f2", "#2a78d6", "#0d2c55"])
CA_CMAP = mpl.colors.LinearSegmentedColormap.from_list("ca", ["#f6d2c1", "#eb6834", "#7a2a0b"])

METRIC_LABELS = {
    "peak_soma_mV": "Somatic EPSP (mV)",
    "peak_syn_mV": "Local EPSP at synapse (mV)",
    "peak_tip_mV": "EPSP at tip (mV)",
    "attenuation": "Attenuation (soma / local)",
    "rise_10_90_soma_ms": "Somatic rise time 10–90% (ms)",
    "halfwidth_soma_ms": "Somatic half-width (ms)",
    "latency_soma_ms": "Somatic latency to peak (ms)",
    "summation_soma": "Train summation (max / first)",
    "area_soma_mVms": "Somatic EPSP area (mV·ms)",
    "dcai_syn_uM": "Local Δ[Ca$^{2+}$]$_i$ (µM)",
    "n_spikes_soma": "Somatic spikes",
}


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150,
    })


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


def save_figure(fig, stem, folder=None, formats=("pdf", "svg", "png")) -> list[Path]:
    folder = Path(folder or FIGURES_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = folder / f"{stem}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    return paths


# --- morphology -------------------------------------------------------------------------

def _xyz(sec):
    pts = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])
    arc = np.array([sec.arc3d(i) for i in range(sec.n3d())]) / sec.L
    return pts, arc


def view_basis(cell):
    """Orthonormal 3x2 projection onto the plane that best shows the target path (PCA)."""
    pts = np.vstack([_xyz(sec)[0] for sec in cell.target_path])
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    return vt[:2].T


def _segment_lines(sec, basis):
    """One projected polyline per segment of `sec`."""
    pts, arc = _xyz(sec)
    pts = pts @ basis
    edges = np.linspace(0, 1, sec.nseg + 1)
    lines = []
    for a, b in zip(edges[:-1], edges[1:]):
        inner = (arc > a) & (arc < b)
        xy = np.vstack([np.interp(a, arc, pts[:, 0]), np.interp(a, arc, pts[:, 1])]).T
        xy = np.vstack([xy, pts[inner, :2],
                        [np.interp(b, arc, pts[:, 0]), np.interp(b, arc, pts[:, 1])]])
        lines.append(xy)
    return lines


def plot_morphology(ax, cell, gbar=None, sites_um=None, gmax=None, scalebar_um=100, basis=None):
    """Whole cell in grey, target path in the morphology colour, or by Ca_LVA density.

    The cell is projected on `basis` (default: the PCA plane of its target path);
    pass the same basis to draw two cells in the same orientation.
    """
    basis = view_basis(cell) if basis is None else basis
    path = set(cell.target_path)
    others = [_xyz(sec)[0] @ basis for sec in cell.all if sec not in path and sec.n3d() > 1]
    ax.add_collection(LineCollection(others, colors="#b5b4ae", linewidths=0.5))
    sx, sy = _xyz(cell.soma[0])[0].mean(0) @ basis

    lines = [xy for sec in cell.target_path for xy in _segment_lines(sec, basis)]
    if gbar is not None and np.any(np.asarray(gbar) > 0):
        norm = mpl.colors.Normalize(0, gmax or np.max(gbar))
        lc = LineCollection(lines, cmap=CA_CMAP, norm=norm, linewidths=1.6)
        lc.set_array(np.asarray(gbar))
        ax.add_collection(lc)
        cb = plt.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Ca$_{LVA}$ g (S/cm²)", fontsize=6)
        cb.ax.tick_params(labelsize=5, width=0.5)
        cb.outline.set_linewidth(0.5)
    else:
        ax.add_collection(LineCollection(lines, colors=MORPH_STYLE[cell.label]["color"],
                                         linewidths=1.6))
    ax.plot(sx, sy, "o", color=INK, ms=4)

    if sites_um is not None:
        from .morphology import path_location
        for d in (d for d in sites_um if d <= cell.tip_distance + 1e-3):
            sec, x, _ = path_location(cell, cell.target_path, d)
            pts, arc = _xyz(sec)
            pts = pts @ basis
            ax.plot(np.interp(x, arc, pts[:, 0]), np.interp(x, arc, pts[:, 1]), "o",
                    ms=2.5, mfc="white", mec=INK, mew=0.5, zorder=5)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if scalebar_um:
        x0, x1 = ax.get_xlim()
        y0, _ = ax.get_ylim()
        ax.plot([x1 - scalebar_um, x1], [y0, y0], color=INK, lw=1)
        ax.annotate(f"{scalebar_um} µm", (x1 - scalebar_um / 2, y0), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top", fontsize=6)


# --- data panels -------------------------------------------------------------------------

def plot_ca_profile(ax, result, color=None, fill=True):
    prof = result.ca_profiles.get("long")
    if prof is None:
        return
    color = color or PROFILE_COLORS[result.config.ca_profile]
    if fill:
        ax.fill_between(prof["distance_um"], prof["gbar_Ca_LVA_S_cm2"], step="mid",
                        color=color, alpha=0.25, lw=0)
    ax.step(prof["distance_um"], prof["gbar_Ca_LVA_S_cm2"], where="mid", color=color, lw=1)
    ax.set_xlim(0, 400)
    ax.set_xlabel("Distance from soma (µm)")
    ax.set_ylabel("Ca$_{LVA}$ g (S/cm²)")


def plot_traces(ax, result, morph, where="soma", sites_um=None, xlim=None):
    """Voltage traces for several sites, coloured by site distance (shared 0-400 µm scale)."""
    norm = mpl.colors.Normalize(0, 400)
    keys = sorted(d for m, d in result.traces if m == morph)
    if sites_um is not None and keys:  # nearest simulated site to each requested one
        keys = sorted({min(keys, key=lambda k: abs(k - s)) for s in sites_um
                       if s <= max(keys) + result.config.site_step_um})
    for d in keys:
        tr = result.traces[(morph, d)]
        ax.plot(tr["t"], tr[f"v_{where}"], color=SITE_CMAP(norm(d)), lw=0.8)
    cfg = result.config
    ax.set_xlim(*(xlim or (cfg.onset_ms - 5, cfg.tstop_ms)))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"V$_{{{where}}}$ (mV)")
    return mpl.cm.ScalarMappable(norm=norm, cmap=SITE_CMAP)


def log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_vs_distance(ax, result, metric, morphs=None, color=None, label=None, **kw):
    df = result.summary
    for morph in morphs or result.config.morphologies:
        sub = df[df.morphology == morph].sort_values("site_um")
        style = dict(MORPH_STYLE[morph])
        if color is not None and morph == "long":
            style["color"] = color
        if label is not None and morph == "long":
            style["label"] = label
        ax.plot(sub["site_um"], sub[metric], marker="o", ms=2.5, mew=0, lw=1, **{**style, **kw})
    ax.set_xlim(0, 405)
    ax.set_xlabel("Synapse distance from soma (µm)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))


# --- composite figures ---------------------------------------------------------------------

def describe(cfg) -> str:
    ca = cfg.ca_profile if cfg.ca_profile == "none" else f"{cfg.ca_profile} ({cfg.ca_norm})"
    stim = ("single event" if cfg.n_events == 1
            else f"{cfg.n_events} events at {cfg.freq_hz:g} Hz")
    flags = [f"somatic Ca$_{{LVA}}$ {'on' if cfg.somatic_ca_lva else 'off'}"]
    if cfg.ttx:
        flags.append("TTX")
    if cfg.prune_side_branches:
        flags.append("side branches pruned")
    return (f"Dendritic Ca$_{{LVA}}$: {ca} | Exp2Syn {cfg.syn_weight_uS * 1e3:g} nS, {stim} | "
            + ", ".join(flags))


def overview_figure(result, cells: dict, sites_for_traces=None):
    """3 x 3 summary of one sweep: morphologies, Ca profile, traces, distance dependence."""
    set_style()
    cfg = result.config
    fig = plt.figure(figsize=(TWO_COL, 150 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 3)
    letters = iter("abcdefghi")

    long_g = result.ca_profiles["long"]["gbar_Ca_LVA_S_cm2"].to_numpy() if "long" in result.ca_profiles else None
    sites = sites_for_traces or [25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    xmax = 1.02 * max(c.distance(s(1)) for c in cells.values() for s in c.dend)
    for col, morph in enumerate(("short", "long")):
        ax = fig.add_subplot(gs[0, col])
        if morph in cells:
            g = long_g if morph == "long" else None
            plot_dendrogram(ax, cells[morph], gbar=g, sites_um=sites, xmax=xmax,
                            mark_um=cells["short"].tip_distance if morph == "long" and "short" in cells else None)
            ax.set_title(MORPH_STYLE[morph]["label"], color=INK)
        else:
            ax.axis("off")
        panel_label(ax, next(letters))

    ax = fig.add_subplot(gs[0, 2])
    if long_g is not None:
        plot_ca_profile(ax, result)
        ax.set_title("Ca$_{LVA}$ on the long dendrite")
    panel_label(ax, next(letters))

    sm = None
    trace_axes = []
    for col, (morph, where) in enumerate((("short", "soma"), ("long", "soma"), ("long", "syn"))):
        ax = fig.add_subplot(gs[1, col])
        if morph in cfg.morphologies:
            sm = plot_traces(ax, result, morph, where, sites_um=sites)
            ax.set_title(f"{MORPH_STYLE[morph]['label']}, {'soma' if where == 'soma' else 'at synapse'}")
        trace_axes.append(ax)
        panel_label(ax, next(letters))
    if len(cfg.morphologies) == 2:
        lo = min(a.get_ylim()[0] for a in trace_axes[:2])
        hi = max(a.get_ylim()[1] for a in trace_axes[:2])
        for a in trace_axes[:2]:
            a.set_ylim(lo, hi)
    if sm is not None:
        cb = fig.colorbar(sm, ax=trace_axes, fraction=0.02, pad=0.01)
        cb.set_label("Synapse distance (µm)", fontsize=6)
        cb.ax.tick_params(labelsize=5, width=0.5)
        cb.outline.set_linewidth(0.5)

    for metric in ("peak_soma_mV", "peak_syn_mV", "attenuation"):
        ax = fig.add_subplot(gs[2, ["peak_soma_mV", "peak_syn_mV", "attenuation"].index(metric)])
        plot_vs_distance(ax, result, metric, color=PROFILE_COLORS[cfg.ca_profile])
        if metric == "attenuation":
            log_axis(ax)
        if metric == "peak_soma_mV":
            ax.legend(loc="upper right")
        panel_label(ax, next(letters))

    fig.suptitle(describe(cfg), fontsize=7)
    return fig


def comparison_figure(results: dict, metrics=("peak_soma_mV", "peak_syn_mV", "attenuation",
                                              "dcai_syn_uM", "area_soma_mVms")):
    """Several Ca profiles (dict label -> Result) on the long dendrite vs the short reference."""
    set_style()
    n = len(metrics) + 1
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TWO_COL, 52 * MM * nrow), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    letters = iter("abcdefghijkl")

    ax = axes[0]
    for label, res in results.items():
        plot_ca_profile(ax, res, color=PROFILE_COLORS[res.config.ca_profile], fill=False)
    ax.set_title("Ca$_{LVA}$ profiles (long dendrite)")
    panel_label(ax, next(letters))

    first = next(iter(results.values()))
    for ax, metric in zip(axes[1:], metrics):
        if "short" in first.config.morphologies:  # short has no dendritic Ca: one reference curve
            plot_vs_distance(ax, first, metric, morphs=["short"])
        for label, res in results.items():
            plot_vs_distance(ax, res, metric, morphs=["long"],
                             color=PROFILE_COLORS[res.config.ca_profile], label=label)
        if metric == "attenuation":
            log_axis(ax)
        panel_label(ax, next(letters))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    for ax in axes[n:]:
        ax.axis("off")
    cfg = first.config
    fig.suptitle(describe(cfg).split(" | ", 1)[1], fontsize=7)
    return fig


def site_figure(result):
    """One synapse location: short vs long at the soma, at the synapse, and local Ca."""
    set_style()
    cfg = result.config
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 55 * MM), layout="constrained")
    d = sorted({k[1] for k in result.traces})[0]
    for ax, (key, title, ylabel) in zip(axes, (
            ("v_soma", "Soma", "V (mV)"),
            ("v_syn", f"At synapse ({d:g} µm)", "V (mV)"),
            ("cai_syn", "Local [Ca$^{2+}$]$_i$", "[Ca$^{2+}$]$_i$ (µM)"))):
        for morph in cfg.morphologies:
            tr = result.traces.get((morph, d))
            if tr is None or key not in tr:
                continue
            style = dict(MORPH_STYLE[morph])
            if morph == "long":
                style["color"] = PROFILE_COLORS[cfg.ca_profile]
            y = tr[key] * (1e3 if key == "cai_syn" else 1)
            ax.plot(tr["t"], y, lw=1, **style)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(cfg.onset_ms - 5, cfg.tstop_ms)
        panel_label(ax, "abc"[list(axes).index(ax)])
    if not axes[2].lines:
        axes[2].text(0.5, 0.5, "no dendritic Ca$_{LVA}$\nat this site", ha="center", va="center",
                     transform=axes[2].transAxes, color=INK_2)
    axes[0].legend(loc="upper right")
    fig.suptitle(describe(cfg), fontsize=7)
    return fig


def morphology_figure(cells: dict, sites_um=range(50, 401, 50)):
    """Short and long morphologies side by side, in the same orientation."""
    set_style()
    basis = view_basis(cells.get("long") or next(iter(cells.values())))
    fig, axes = plt.subplots(1, len(cells), figsize=(TWO_COL, 70 * MM), layout="constrained")
    for ax, (m, cell) in zip(np.atleast_1d(axes), cells.items()):
        plot_morphology(ax, cell, basis=basis, sites_um=sites_um)
        ax.set_title(f"{MORPH_STYLE[m]['label']}: tip at {cell.tip_distance:.1f} µm")
    return fig


TAU_CMAP = mpl.colors.LinearSegmentedColormap.from_list("tau", ["#9ec2ee", "#2a78d6", "#0d2c55"])


def mechanism_figure(D: dict):
    """Why the Ca_LVA event appears only beyond a distance (data from mechanism.collect)."""
    from .mechanism import EVENT_CRITERION_MV, SCAN_WEIGHTS
    set_style()
    fig, axes = plt.subplots(3, 3, figsize=(TWO_COL, 165 * MM), layout="constrained")
    a = axes.ravel()
    g, rest = D["gating"], D["rest_mV"]
    blue, orange, grey = PROFILE_COLORS["uniform"], PROFILE_COLORS["hotspot"], INK_2

    ax = a[0]
    ax.plot(g.v, g.m_inf ** 2, color=blue, label="activation m∞²")
    ax.plot(g.v, g.h_inf, color=orange, label="availability h∞")
    ax.axvline(rest, color=grey, lw=0.6, ls=":")
    ax.annotate(f"rest, h∞ = {D['h_rest']:.2f}", (rest, D["h_rest"]), xytext=(4, 8),
                textcoords="offset points", fontsize=6, color=grey)
    ax.set(xlabel="V (mV)", ylabel="Steady state", title="Ca$_{LVA}$ gating", xlim=(-100, 20))
    ax.legend(loc="center right")

    ax = a[1]
    ax.plot(g.v, g.m_tau, color=blue, label="τ$_m$")
    ax.plot(g.v, g.h_tau, color=orange, label="τ$_h$")
    ax.set(xlabel="V (mV)", ylabel="Time constant (ms)", title=f"Ca$_{{LVA}}$ kinetics, {D['cfg'].celsius:g} °C",
           xlim=(-100, 20), ylim=(0, None))
    ax.legend(loc="upper right")

    ax = a[2]
    for m in ("short", "long"):
        z = D["impedance"][m]
        st = MORPH_STYLE[m]
        ax.plot(z.distance_um, z["zin_0Hz_MOhm"], color=st["color"], ls=st["ls"], label=st["label"])
    ax.axhline(D["impedance"]["long"]["zin_soma_0Hz_MOhm"].iloc[0], color=grey, lw=0.6, ls=":")
    ax.text(395, D["impedance"]["long"]["zin_soma_0Hz_MOhm"].iloc[0] * 1.15, "soma", ha="right",
            fontsize=6, color=grey)
    log_axis(ax)
    ax.set(xlabel="Distance from soma (µm)", ylabel="Local input impedance (MΩ)",
           title="Electrotonic load (passive dendrite)", xlim=(0, 405))
    ax.legend(loc="upper left")

    for ax, col, ylabel, title in ((a[3], "peak_abs_mV", "Peak local V (mV)", "Passive local EPSP: peak"),
                                   (a[4], "t_above_ms", "Time above −50 mV (ms)", "Passive local EPSP: duration")):
        for m in ("short", "long"):
            e = D["epsp"][m]
            st = MORPH_STYLE[m]
            ax.plot(e.distance_um, e[col], color=st["color"], ls=st["ls"], marker="o", ms=2, mew=0)
        ax.set(xlabel="Synapse distance from soma (µm)", ylabel=ylabel, title=title, xlim=(0, 405))
    a[3].axhline(-40, color=blue, lw=0.6, ls=":")
    a[3].text(400, -41.5, "Ca$_{LVA}$ m∞ half-activation", fontsize=6, color=blue, ha="right", va="top")

    ax = a[5]
    th = D["thresholds"]
    ok = th.threshold_nS.notna()
    ax.plot(th.distance_um[ok], th.threshold_nS[ok], color=blue, marker="o", ms=3, mew=0)
    ax.plot(th.distance_um[~ok], [max(SCAN_WEIGHTS)] * int((~ok).sum()), "^", color=grey, ms=4, mew=0)
    ax.text(th.distance_um[~ok].mean() if (~ok).any() else 100, max(SCAN_WEIGHTS) * 0.82,
            f"no event up to {max(SCAN_WEIGHTS)} nS", ha="center", fontsize=6, color=grey)
    ax.axhline(D["cfg"].syn_weight_uS * 1e3, color=grey, lw=0.6, ls=":")
    ax.set(xlabel="Synapse distance from soma (µm)", ylabel="Threshold synaptic weight (nS)",
           title="Synaptic strength needed for an event", xlim=(0, 405), ylim=(0, max(SCAN_WEIGHTS) * 1.08))

    ax = a[6]
    taus = sorted(D["tau"])
    for i, t in enumerate(taus):
        df = D["tau"][t]
        ax.plot(df.distance_um, df.extra_local_mV, color=TAU_CMAP(i / max(len(taus) - 1, 1)),
                marker="o", ms=2, mew=0, label=f"τ$_{{decay}}$ = {t:g} ms")
    ax.axhline(EVENT_CRITERION_MV, color=grey, lw=0.6, ls=":")
    ax.set(xlabel="Synapse distance from soma (µm)", ylabel="Local boost by Ca$_{LVA}$ (mV)",
           title="Longer EPSPs trigger events closer in", xlim=(0, 405))
    ax.legend(loc="upper left")

    ex, site = D["example"], D["example_site"]
    on = D["cfg"].onset_ms
    ax = a[7]
    ax.plot(ex["none"]["t"], ex["none"]["v_syn"], color=grey, ls="--", label="no dendritic Ca$_{LVA}$")
    ax.plot(ex["uniform"]["t"], ex["uniform"]["v_syn"], color=blue, label="uniform Ca$_{LVA}$")
    ax.set(xlabel="Time (ms)", ylabel="Local V (mV)", title=f"Event at {site:g} µm (5 nS)", xlim=(on - 5, on + 80))
    ax.legend(loc="upper right")

    ax = a[8]
    u = ex["uniform"]
    ax.plot(u["t"], u["m"] ** 2, color=blue, label="m²")
    ax.plot(u["t"], u["h"], color=orange, label="h")
    ax.plot(u["t"], u["m"] ** 2 * u["h"] / max((u["m"] ** 2 * u["h"]).max(), 1e-12), color=INK, lw=0.8,
            label="m²h (normalised)")
    ax.set(xlabel="Time (ms)", ylabel="Gate", title="Activation, then inactivation", xlim=(on - 5, on + 80),
           ylim=(0, 1.05))
    ax.legend(loc="upper right")

    for ax, letter in zip(a, "abcdefghi"):
        panel_label(ax, letter)
    return fig


# --- dendrogram -----------------------------------------------------------------------------

def _dendro_layout(cell):
    """{section: (d_start, d_end, y)}; leaves get consecutive y, parents the mean of their children."""
    from .morphology import children
    layout, next_y = {}, [0.0]

    def place(sec):
        kids = children(sec)
        ys = [place(k) for k in kids]
        y = float(np.mean(ys)) if ys else next_y[0]
        if not ys:
            next_y[0] += 1.0
        layout[sec] = (cell.distance(sec(0)), cell.distance(sec(1)), y)
        return y

    for sec in (s for s in cell.dend if _parent_or_none(s) == cell.soma[0]):  # primary dendrites
        place(sec)
        next_y[0] += 0.6  # gap between primary trees
    return layout


def _parent_or_none(sec):
    from .morphology import parent
    return parent(sec)


def plot_dendrogram(ax, cell, gbar=None, gmax=None, mark_um=None, sites_um=None, xmax=None):
    """Dendrogram of the basal tree; the target path is highlighted (or coloured by Ca_LVA)."""
    from .morphology import children
    lay = _dendro_layout(cell)
    path = set(cell.target_path)
    base, conn = [], []
    for sec, (d0, d1, y) in lay.items():
        if sec not in path:
            base.append([(d0, y), (d1, y)])
        kids = children(sec)
        if kids:
            ys = [lay[k][2] for k in kids] + [y]
            conn.append([(d1, min(ys)), (d1, max(ys))])
        if _parent_or_none(sec) == cell.soma[0]:
            conn.append([(0, y), (d0, y)])
    ys_all = [v[2] for v in lay.values()]
    conn.append([(0, min(ys_all)), (0, max(ys_all))])
    ax.add_collection(LineCollection(base + conn, colors="#b5b4ae", linewidths=0.6))

    segs = [(seg, d) for seg, d in cell.path_segments()]
    lines, prev = [], cell.distance(cell.target_path[0](0))
    for (seg, d), nxt in zip(segs, [d for _, d in segs[1:]] + [cell.tip_distance]):
        y = lay[seg.sec][2]
        end = (d + nxt) / 2 if nxt != cell.tip_distance else cell.tip_distance
        lines.append([(prev, y), (end, y)])
        prev = end
    if gbar is not None and np.any(np.asarray(gbar) > 0):
        lc = LineCollection(lines, cmap=CA_CMAP, norm=mpl.colors.Normalize(0, gmax or np.max(gbar)),
                            linewidths=2.2)
        lc.set_array(np.asarray(gbar))
        ax.add_collection(lc)
        cax = ax.inset_axes([0.76, 0.9, 0.2, 0.035])  # inside the panel: keeps x axes aligned
        cb = plt.colorbar(lc, cax=cax, orientation="horizontal")
        cb.set_label("Ca$_{LVA}$ g (S/cm²)", fontsize=6, labelpad=1)
        cb.ax.xaxis.set_label_position("top")
        cb.set_ticks([0, lc.norm.vmax])
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        cb.ax.tick_params(labelsize=5, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)
    else:
        ax.add_collection(LineCollection(lines, colors=MORPH_STYLE[cell.label]["color"], linewidths=2.2))
    if sites_um is not None:
        from .morphology import path_location
        for d in (d for d in sites_um if d <= cell.tip_distance + 1e-3):
            sec, _, _ = path_location(cell, cell.target_path, d)
            ax.plot(d, lay[sec][2], "o", ms=2.5, mfc="white", mec=INK, mew=0.5, zorder=5)
    if mark_um is not None:
        y = lay[cell.target_path[-1]][2]
        ax.plot([mark_um], [y], "|", color=INK, ms=7, mew=1)
        ax.annotate("original tip", (mark_um, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6)
    ax.autoscale_view()
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Path distance from soma (µm)")


def dendrogram_figure(cells: dict, gbar=None, gmax=None):
    """Short and long dendrograms on a shared distance axis."""
    set_style()
    fig, axes = plt.subplots(len(cells), 1, figsize=(TWO_COL, 62 * MM * len(cells)), sharex=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    short_tip = cells["short"].tip_distance if "short" in cells else None
    xmax = 1.02 * max(c.distance(s(1)) for c in cells.values() for s in c.dend)
    for ax, (m, cell) in zip(axes, cells.items()):
        plot_dendrogram(ax, cell, gbar=gbar if m == "long" else None, gmax=gmax, xmax=xmax,
                        mark_um=short_tip if m == "long" else None)
        ax.set_title(f"{MORPH_STYLE[m]['label']}: target dendrite ends at {cell.tip_distance:.1f} µm",
                     loc="left")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    return fig
