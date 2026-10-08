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
PROFILE_COLORS = {"none": INK_2, "uniform": "#2a78d6", "increasing": "#1baf7a"}
ORANGE = "#eb6834"  # second categorical slot, used for inactivation (h) in gating plots
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

def ca_label(cfg) -> str:
    if cfg.ca_profile == "none":
        return "none"
    kin = "original kinetics" if cfg.ca_act_shift_mV == 0 else f"activation shifted {cfg.ca_act_shift_mV:g} mV"
    return f"{cfg.ca_profile}, {cfg.g_ca_mS_cm2:g} mS/cm², {kin}"


def describe(cfg) -> str:
    ca = ca_label(cfg)
    stim = ("single event" if cfg.n_events == 1
            else f"{cfg.n_events} events at {cfg.freq_hz:g} Hz")
    flags = [f"somatic Ca$_{{LVA}}$ {'on' if cfg.somatic_ca_lva else 'off'}"]
    if cfg.ttx:
        flags.append("TTX")
    if cfg.prune_side_branches:
        flags.append("side branches pruned")
    if cfg.mouth_scale != 1:
        flags.append(f"mouth {cfg.mouth_scale:g}×")
    if cfg.soma_bias_frac:
        flags.append(f"bias {100 * cfg.soma_bias_frac:.0f}% rheobase")
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
    blue, orange, grey = PROFILE_COLORS["uniform"], ORANGE, INK_2

    ax = a[0]
    if D["cfg"].ca_act_shift_mV:
        ax.plot(g.v, g.m0_inf ** 2, color=blue, ls="--", lw=0.8, label="m∞², original")
    ax.plot(g.v, g.m_inf ** 2, color=blue, label="activation m∞²" + (", shifted" if D["cfg"].ca_act_shift_mV else ""))
    ax.plot(g.v, g.h_inf, color=orange, label="availability h∞")
    ax.axvline(rest, color=grey, lw=0.6, ls=":")
    ax.annotate(f"rest, h∞ = {D['h_rest']:.2f}", (rest, D["h_rest"]), xytext=(4, 8),
                textcoords="offset points", fontsize=6, color=grey)
    ax.set(xlabel="V (mV)", ylabel="Steady state", title="Ca$_{LVA}$ gating", xlim=(-100, 20))
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08))

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
    if (~ok).any():
        ax.text(th.distance_um[~ok].mean(), max(SCAN_WEIGHTS) * 0.82,
                f"no event up to {max(SCAN_WEIGHTS)} nS", ha="center", fontsize=6, color=grey)
    ax.axhline(D["cfg"].syn_weight_uS * 1e3, color=grey, lw=0.6, ls=":")
    ax.set(xlabel="Synapse distance from soma (µm)", ylabel="Threshold synaptic weight (nS)",
           title="Synaptic strength needed for an event", xlim=(0, 405), ylim=(0, max(SCAN_WEIGHTS) * 1.08))

    ax = a[6]
    taus = sorted(D["tau"])
    for i, t in enumerate(taus):
        df = D["tau"][t]
        ax.plot(df.distance_um, np.maximum(df.extra_local_mV, df.extra_tip_mV), color=TAU_CMAP(i / max(len(taus) - 1, 1)),
                marker="o", ms=2, mew=0, label=f"τ$_{{decay}}$ = {t:g} ms")
    ax.axhline(EVENT_CRITERION_MV, color=grey, lw=0.6, ls=":")
    ax.set(xlabel="Synapse distance from soma (µm)", ylabel="Boost by Ca$_{LVA}$, synapse or tip (mV)",
           title="Longer EPSPs trigger events closer in", xlim=(0, 405))
    ax.legend(loc="upper left")

    ex, site = D["example"], D["example_site"]
    on = D["cfg"].onset_ms
    ax = a[7]
    ax.plot(ex["none"]["t"], ex["none"]["v_syn"], color=grey, ls="--", label="no dendritic Ca$_{LVA}$")
    prof = D["cfg"].ca_profile
    ax.plot(ex[prof]["t"], ex[prof]["v_syn"], color=PROFILE_COLORS[prof], label=f"{prof} Ca$_{{LVA}}$")
    ax.set(xlabel="Time (ms)", ylabel="Local V (mV)", title=f"Event at {site:g} µm (5 nS)", xlim=(on - 5, on + 80))
    ax.legend(loc="upper right")

    ax = a[8]
    u = ex[prof]
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


def mouth_cell_scale(cell) -> float:
    return getattr(cell, "mouth_scale", 1.0)


def diameter_figure(cells: dict, mouth_cell=None):
    """Diameter and cumulative membrane area along the target dendrite (short vs long)."""
    from .morphology import leaves, path_to_soma
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM), layout="constrained")
    ref = cells.get("long") or next(iter(cells.values()))

    def profile(cell, path):  # reconstruction points: (path distance, diameter)
        d, dia = [], []
        for sec in path:
            d0 = cell.distance(sec(0))
            d += [d0 + sec.arc3d(i) for i in range(sec.n3d())]
            dia += [sec.diam3d(i) for i in range(sec.n3d())]
        return d, dia

    for leaf in leaves(ref):  # every other soma-to-tip path, for context
        if leaf == ref.target_path[-1]:
            continue
        axes[0].plot(*profile(ref, path_to_soma(ref, leaf)), color="#cfcec9", lw=0.5, zorder=1)
    axes[0].plot([], [], color="#cfcec9", lw=0.5, label="other soma-to-tip paths")
    for m in ("long", "short"):
        if m not in cells:
            continue
        c, st = cells[m], MORPH_STYLE[m]
        segs = c.path_segments()
        axes[0].plot(*profile(c, c.target_path), color=st["color"], ls=st["ls"],
                     lw=1.4, label=f"target, {st['label']}", zorder=3)
        area = np.cumsum([s.area() for s, _ in segs])
        axes[1].plot([d for _, d in segs], area, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
    if mouth_cell is not None:
        axes[0].plot(*profile(mouth_cell, mouth_cell.target_path), color="#4a3aa7", ls="-.", lw=1.2,
                     label=f"target, long, {mouth_cell_scale(mouth_cell):g}× mouth", zorder=4)
    soma_area = sum(s.area() for s in ref.soma[0])
    axes[1].axhline(soma_area, color=INK_2, lw=0.6, ls=":")
    axes[1].text(5, soma_area * 1.03, f"soma membrane ({soma_area:.0f} µm²)", fontsize=6, color=INK_2, va="bottom")
    if "short" in cells:
        for ax in axes:
            ax.axvline(cells["short"].tip_distance, color=INK_2, lw=0.6, ls=":")
        axes[0].text(cells["short"].tip_distance + 4, 0.05, "original tip", fontsize=6, color=INK_2)
    axes[0].set(xlabel="Path distance from soma (µm)", ylabel="Diameter (µm)", ylim=(0, None),
                title="Diameter along the target dendrite", xlim=(0, None))
    axes[1].set(xlabel="Path distance from soma (µm)", ylabel="Cumulative membrane area (µm²)",
                title="Membrane area of the target dendrite", xlim=(0, 405), ylim=(0, None))
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="center right")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    return fig


def firing_figure(study: dict, example_site_um=300.0):
    """Somatic bias study: example traces, spikes and Ca_LVA boost vs distance, one row per bias level."""
    from .mechanism import EVENT_CRITERION_MV
    set_style()
    fracs = sorted(study)
    fig, axes = plt.subplots(len(fracs), 3, figsize=(TWO_COL, 52 * MM * len(fracs)), layout="constrained",
                             sharex="col")
    axes = np.atleast_2d(axes)
    letters = iter("abcdefghijklmnop")
    for row, frac in zip(axes, fracs):
        res = study[frac]
        none = res["none"]
        sl = none.summary
        b_long = sl[sl.morphology == "long"].bias_nA.iloc[0] if "bias_nA" in sl else 0.0
        v_rest = sl[sl.morphology == "long"].vrest_soma_mV.iloc[0]
        label = ("no bias current" if frac == 0 else
                 f"bias {100 * frac:.0f}% of rheobase ({b_long:.3f} nA)")

        ax = row[0]
        for prof, style in (("none", dict(color=INK_2, ls="--", label="long, no dendritic Ca$_{LVA}$")),
                            ("uniform", dict(color=PROFILE_COLORS["uniform"], label="long, uniform Ca$_{LVA}$"))):
            tr = res[prof].traces.get(("long", example_site_um))
            if tr is not None:
                ax.plot(tr["t"], tr["v_soma"], lw=0.9, **style)
        cfg = none.config
        ax.set(xlim=(cfg.onset_ms - 5, cfg.onset_ms + 80), ylabel="V$_{soma}$ (mV)",
               title=f"{label}; rest {v_rest:.1f} mV")
        ax.text(0.98, 0.95, f"synapse at {example_site_um:g} µm", transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color=INK_2)
        if row is axes[0]:
            ax.legend(loc="center right")

        ax = row[1]
        plot_vs_distance(ax, none, "n_spikes_soma", morphs=["short"])
        for prof, r in res.items():
            plot_vs_distance(ax, r, "n_spikes_soma", morphs=["long"], color=PROFILE_COLORS[prof], label=prof)
        ax.set(ylabel="Somatic spikes", title="Does the synapse fire the cell?", ylim=(-0.15, None))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        ax = row[2]
        base = none.summary.query("morphology == 'long'").set_index("site_um").peak_syn_mV
        for prof, r in res.items():
            if prof == "none":
                continue
            extra = r.summary.query("morphology == 'long'").set_index("site_um").peak_syn_mV - base
            ax.plot(extra.index, extra.values, color=PROFILE_COLORS[prof], marker="o", ms=2, mew=0, lw=1, label=prof)
        ax.axhline(EVENT_CRITERION_MV, color=INK_2, lw=0.6, ls=":")
        ax.set(xlim=(0, 405), ylabel="Local boost by Ca$_{LVA}$ (mV)", title="Dendritic Ca$_{LVA}$ event")
        for ax in row:
            panel_label(ax, next(letters))
    for ax in axes[-1]:
        ax.set_xlabel(ax.get_xlabel() or "Time (ms)")
    axes[-1][0].set_xlabel("Time (ms)")
    axes[-1][1].set_xlabel("Synapse distance from soma (µm)")
    axes[-1][2].set_xlabel("Synapse distance from soma (µm)")
    for ax in axes[:-1].ravel():
        ax.set_xlabel("")
    h_, l_ = axes[0][1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=len(l_))
    return fig


def spike_traces_figure(study: dict, frac: float, profiles=("none", "uniform", "increasing"), window_ms=60.0):
    """Somatic and local traces at one bias level, coloured by synapse distance; spikes when the cell fires."""
    set_style()
    res = study[frac]
    cols = [("short", "none")] + [("long", p) for p in profiles]
    fig, axes = plt.subplots(2, len(cols), figsize=(TWO_COL, 100 * MM), layout="constrained",
                             sharex=True, sharey="row")
    norm = mpl.colors.Normalize(0, 400)
    letters = iter("abcdefghij")
    for j, (morph, prof) in enumerate(cols):
        r = res[prof]
        cfg = r.config
        keys = sorted(d for m, d in r.traces if m == morph)
        keys = [d for d in keys if abs(d / 20 - round(d / 20)) < 1e-6 or d == keys[0]]  # every 20 µm
        summ = r.summary[r.summary.morphology == morph].set_index("site_um")
        for d in keys:
            tr = r.traces[(morph, d)]
            fired = summ.loc[d, "n_spikes_soma"] > 0
            for row, where in ((0, "v_soma"), (1, "v_syn")):
                axes[row, j].plot(tr["t"], tr[where], color=SITE_CMAP(norm(d)), lw=1.1 if fired else 0.6,
                                  zorder=3 if fired else 2)
        n_f = int((summ.n_spikes_soma > 0).sum())
        title = f"{MORPH_STYLE[morph]['label'].split(' ')[0]}, " + ("no dendritic Ca$_{LVA}$" if prof == "none"
                                                                   else f"{prof} Ca$_{{LVA}}$")
        axes[0, j].set_title(f"{title}\n{n_f} of {len(summ)} sites fire the cell", fontsize=7)
        axes[0, j].set_xlim(cfg.onset_ms - 5, cfg.onset_ms + window_ms)
        axes[1, j].set_xlabel("Time (ms)")
        for row in (0, 1):
            panel_label(axes[row, j], next(letters) if row == 0 else "")
    for j in range(len(cols)):
        panel_label(axes[1, j], "efgh"[j])
    axes[0, 0].set_ylabel("V$_{soma}$ (mV)")
    axes[1, 0].set_ylabel("V at synapse (mV)")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=SITE_CMAP), ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Synapse distance (µm)", fontsize=6)
    cb.ax.tick_params(labelsize=5, width=0.5)
    cb.outline.set_linewidth(0.5)
    sl = res["none"].summary.query("morphology == 'long'")
    b = sl.bias_nA.iloc[0] if "bias_nA" in sl else 0.0
    fig.suptitle(f"Somatic bias {100 * frac:.0f}% of rheobase ({b:.3f} nA, rest {sl.vrest_soma_mV.iloc[0]:.1f} mV); "
                 f"single synaptic events, 5 nS; thick traces: the cell fires", fontsize=7)
    return fig



# --- figures of the factorial design -----------------------------------------------------------------

def _cond_title(shift, g):
    kin = "original kinetics" if shift == 0 else f"activation {shift:g} mV"
    return f"{g:g} mS/cm², {kin}"


def factorial_figure(grid: dict, mouth=1.0):
    """Columns: (activation shift, density); rows: somatic EPSP and local Ca_LVA boost against distance."""
    from .mechanism import EVENT_CRITERION_MV
    set_style()
    keys = sorted({(s, g) for s, g, m in grid if m == mouth}, key=lambda k: (k[0] != 0, k[1]))
    fig, axes = plt.subplots(2, len(keys), figsize=(TWO_COL, 105 * MM), layout="constrained", sharey="row",
                             sharex=True)
    for j, (shift, g) in enumerate(keys):
        res = grid[(shift, g, mouth)]
        ax = axes[0, j]
        plot_vs_distance(ax, res["none"], "peak_soma_mV", morphs=["short"])
        for prof, r in res.items():
            plot_vs_distance(ax, r, "peak_soma_mV", morphs=["long"], color=PROFILE_COLORS[prof],
                             label=f"long, {prof}" if prof != "none" else "long, no Ca$_{LVA}$")
        ax.set_title(_cond_title(shift, g))
        ax.set_xlabel("")
        if j:
            ax.set_ylabel("")
        ax = axes[1, j]
        base = res["none"].summary.query("morphology == 'long'").set_index("site_um")
        for prof, r in res.items():
            if prof == "none":
                continue
            s = r.summary.query("morphology == 'long'").set_index("site_um")
            ax.plot(s.index, s.peak_syn_mV - base.peak_syn_mV, color=PROFILE_COLORS[prof], marker="o", ms=2, mew=0,
                    lw=1)
            ax.plot(s.index, s.peak_tip_mV - base.peak_tip_mV, color=PROFILE_COLORS[prof], ls="--", lw=0.8)
        ax.axhline(EVENT_CRITERION_MV, color=INK_2, lw=0.6, ls=":")
        ax.set(xlim=(0, 405), xlabel="Synapse distance from soma (µm)",
               ylabel="Boost by Ca$_{LVA}$ (mV)" if j == 0 else "")
        if j == 0:
            ax.plot([], [], color=INK_2, lw=1, label="at the synapse")
            ax.plot([], [], color=INK_2, lw=0.8, ls="--", label="at the tip")
            ax.legend(loc="upper left", fontsize=5.5)
        panel_label(axes[0, j], "abcd"[j])
        panel_label(axes[1, j], "efgh"[j])
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=len(l_))
    return fig


def mouth_figure(grid: dict, metric="peak_soma_mV"):
    """Effect of the 1.5x mouth: ratio of the metric (wide / original mouth) against distance."""
    set_style()
    keys = sorted({(s, g) for s, g, m in grid}, key=lambda k: (k[0] != 0, k[1]))
    mouths = sorted({m for _, _, m in grid})
    if len(mouths) < 2:
        return None
    m0, m1 = mouths[0], mouths[-1]
    fig, axes = plt.subplots(1, len(keys), figsize=(TWO_COL, 55 * MM), layout="constrained", sharey=True)
    for j, (shift, g) in enumerate(keys):
        ax = axes[j]
        for prof in grid[(shift, g, m0)]:
            for morph in (["short", "long"] if prof == "none" else ["long"]):
                a = grid[(shift, g, m0)][prof].summary.query("morphology == @morph").set_index("site_um")[metric]
                b = grid[(shift, g, m1)][prof].summary.query("morphology == @morph").set_index("site_um")[metric]
                st = dict(MORPH_STYLE[morph]) if prof == "none" else dict(color=PROFILE_COLORS[prof], ls="-")
                st.pop("label", None)
                lab = (MORPH_STYLE[morph]["label"] + (", no Ca$_{LVA}$" if morph == "long" else "")
                       if prof == "none" else f"long, {prof}")
                ax.plot(a.index, 100 * (b / a - 1), marker="o", ms=2, mew=0, lw=1, label=lab, **st)
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set(title=_cond_title(shift, g), xlabel="Synapse distance from soma (µm)", xlim=(0, 405),
               ylabel=f"Change with {m1:g}× mouth (%)" if j == 0 else "")
        panel_label(ax, "abcd"[j])
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=len(l_))
    return fig


def electrotonic_figure(imp: dict, grid_none: dict, grid_ca=None, ca_label_text=""):
    """Impedance map along the target dendrite and EPSP time course against distance.

    imp: {label: DataFrame from mechanism.impedance_profile}; grid_none: {'short': Result, 'long': Result,
    'long, mouth': Result} without dendritic Ca; grid_ca: optional Result with dendritic Ca (long).
    """
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL, 110 * MM), layout="constrained")
    a = axes.ravel()
    styles = {"short": dict(color=INK_2, ls="--"), "long": dict(color=INK, ls="-"),
              "long, 1.5× mouth": dict(color="#4a3aa7", ls="-.")}
    for label, df in imp.items():
        st = styles.get(label, dict(color=INK, ls="-"))
        a[0].plot(df.distance_um, df["zin_0Hz_MOhm"], lw=1.2, label=label, **st)
        a[0].plot(df.distance_um, df["zin_100Hz_MOhm"], lw=0.7, alpha=0.6, **st)
        a[1].plot(df.distance_um, df["ztr_0Hz_MOhm"], lw=1.2, **st)
        a[1].plot(df.distance_um, df["ztr_100Hz_MOhm"], lw=0.7, alpha=0.6, **st)
        a[2].plot(df.distance_um, df["ztr_0Hz_MOhm"] / df["zin_0Hz_MOhm"], lw=1.2, **st)
    a[0].axhline(next(iter(imp.values()))["zin_soma_0Hz_MOhm"].iloc[0], color=INK_2, lw=0.6, ls=":")
    log_axis(a[0])
    a[0].set(title="Local input impedance", ylabel="|Z$_{in}$| (MΩ)")
    a[1].set(title="Transfer impedance to the soma", ylabel="|Z$_{transfer}$| (MΩ)")
    a[2].set(title="Steady-state voltage attenuation", ylabel="V$_{soma}$ / V$_{local}$")
    a[0].plot([], [], color=INK_2, lw=1.2, label="0 Hz")
    a[0].plot([], [], color=INK_2, lw=0.7, alpha=0.6, label="100 Hz")
    a[0].legend(loc="lower right", fontsize=5.5)
    for ax in a[:3]:
        ax.set(xlabel="Distance from soma (µm)", xlim=(0, 405))

    metrics = (("area_mVms", "EPSP integral (mV·ms)"), ("tau_eff_ms", "Effective time constant (ms)"),
               ("tau_decay_ms", "Decay time constant (ms)"))
    for ax, (key, ylabel) in zip(a[3:], metrics):
        for label, res in grid_none.items():
            morph = "short" if label == "short" else "long"
            st = styles.get(label, dict(color=INK, ls="-"))
            sub = res.summary[res.summary.morphology == morph].sort_values("site_um")
            ax.plot(sub.site_um, sub[f"{key}_soma"], lw=1.2, marker="o", ms=1.8, mew=0, **st,
                    label=f"{label}, soma")
            ax.plot(sub.site_um, sub[f"{key}_syn"], lw=0.7, alpha=0.6, **st, label=f"{label}, synapse")
        if grid_ca is not None:
            sub = grid_ca.summary[grid_ca.summary.morphology == "long"].sort_values("site_um")
            col = PROFILE_COLORS[grid_ca.config.ca_profile]
            ax.plot(sub.site_um, sub[f"{key}_soma"], color=col, lw=1.2, marker="o", ms=1.8, mew=0,
                    label=f"long + Ca$_{{LVA}}$ ({ca_label_text}), soma")
            ax.plot(sub.site_um, sub[f"{key}_syn"], color=col, lw=0.7, alpha=0.6,
                    label="long + Ca$_{LVA}$, synapse")
        ax.set(xlabel="Synapse distance from soma (µm)", ylabel=ylabel, xlim=(0, 405), title=ylabel.split(" (")[0])
        if key == "area_mVms":
            log_axis(ax)
    h_, l_ = a[3].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=4, fontsize=5.5)
    for ax, letter in zip(a, "abcdef"):
        panel_label(ax, letter)
    return fig


NOISE_STYLE_BASE = {"short": dict(color=INK_2, ls="--"), "long, no Ca_LVA": dict(color=INK, ls="-")}


def noise_style(cond: str) -> dict:
    if cond in NOISE_STYLE_BASE:
        return dict(NOISE_STYLE_BASE[cond])
    prof = "increasing" if "increasing" in cond else "uniform"
    return dict(color=PROFILE_COLORS[prof], ls="-" if "shifted" in cond else ":")


def _smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode="same")


def noise_figure(psth_data: dict, evoked, examples: dict, N: dict, psth_sites=(100, 200, 300),
                 example_conditions=None):
    """Noisy somatic current: example traces, PSTHs of the spikes added by the synapse, evoked spikes vs distance."""
    set_style()
    fig = plt.figure(figsize=(TWO_COL, 165 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 6)
    conds = list(dict.fromkeys(evoked.condition))
    example_conditions = example_conditions or conds[1:2] + conds[-1:]
    letters = iter("abcdefghij")
    for j, cond in enumerate(example_conditions[:2]):
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        ex = examples.get(cond, {})
        for tag, col, lw, lab in (("nosyn", "#b5b4ae", 0.7, "noise only"),
                                  ("syn", PROFILE_COLORS["uniform"], 0.9, "noise + synapse (same noise)")):
            if tag in ex:
                ax.plot(*ex[tag], color=col, lw=lw, label=lab)
        for k in range(2):
            ax.axvline(k * N["period_ms"], color=INK, lw=0.6, ls=":")
        ax.set(xlim=(-50, 2 * N["period_ms"] - 50), xlabel="Time from first synaptic input (ms)", ylabel="V$_{soma}$ (mV)",
               title=f"{cond.replace('Ca_LVA', 'Ca$_{LVA}$')}: synapse at {N['example_site_um']:g} µm")
        if j == 0:
            ax.legend(loc="upper right", fontsize=5.5)
        panel_label(ax, next(letters))
    top = 0.0
    axes_p = []
    for j, site in enumerate(psth_sites):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        for cond in conds:
            if (cond, site) not in psth_data:
                continue
            t, h1, h0 = psth_data[(cond, site)]
            y = _smooth(h1 - h0)
            top = max(top, y.max())
            ax.plot(t, y, lw=1, **noise_style(cond))
        ax.axhline(0, color=INK_2, lw=0.5)
        ax.axvline(0, color=INK, lw=0.6, ls=":")
        ax.set(xlim=(-20, 100), xlabel="Time from synaptic input (ms)", title=f"Synapse at {site:g} µm",
               ylabel="Extra firing rate (Hz)" if j == 0 else "")
        axes_p.append(ax)
        panel_label(ax, next(letters))
    for ax in axes_p:
        ax.set_ylim(top=top * 1.1)
    ax = fig.add_subplot(gs[2, 0:3])
    for cond, grp in evoked.groupby("condition", sort=False):
        st = noise_style(cond)
        ax.plot(grp.site_um, grp.evoked, marker="o", ms=2.5, mew=0, lw=1, label=cond.replace("Ca_LVA", "Ca$_{LVA}$"),
                **st)
        ax.fill_between(grp.site_um, grp.ci_lo, grp.ci_hi, color=st["color"], alpha=0.07, lw=0)
    ax.axhline(0, color=INK_2, lw=0.5)
    c0, c1 = N["count_window_ms"]
    ax.set(xlim=(0, 405), xlabel="Synapse distance from soma (µm)", ylabel="Extra spikes per input",
           title=f"Spikes added by the synapse ({c0:g}–{c1:g} ms)")
    panel_label(ax, next(letters))
    ax2 = fig.add_subplot(gs[2, 3:6])
    site = N["example_site_um"]
    for cond in example_conditions[:2]:
        if (cond, site) in psth_data:
            t, h1, h0 = psth_data[(cond, site)]
            st = noise_style(cond)
            ax2.plot(t, _smooth(h1), lw=1, **st)
            ax2.plot(t, _smooth(h0), lw=0.6, color=st["color"], alpha=0.5, ls="--")
    ax2.axvline(0, color=INK, lw=0.6, ls=":")
    ax2.set(xlim=(-50, 150), xlabel="Time from synaptic input (ms)", ylabel="Firing rate (Hz)",
            title=f"PSTH, synapse at {site:g} µm (dashed: no synapse)")
    panel_label(ax2, next(letters))
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=3, fontsize=5.5)
    return fig
