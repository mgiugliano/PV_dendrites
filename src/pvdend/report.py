"""One self-contained HTML report of all studies in configs/studies.json.

Figures are drawn by the same functions as scripts/ and the notebook, from the cached
results (results/), embedded as vector SVG, and also saved as PDF/PNG in figures/.
Every number in the text is computed from the results when the report is built.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
from pathlib import Path

import matplotlib.pyplot as plt
import neuron
import numpy as np

from . import __version__, manuscript, mechanism, plotting, studies, viewer3d
from ._paths import FIGURES_DIR
from .config import Config
from .morphology import load_target_meta, path_geometry
from .protocols import get_cell

TABLE_SITES = (10, 100, 200, 300, 400)
SHIFT_TXT = {0.0: "original", -15.0: "shifted −15 mV"}

CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --rule:#d9d8d4; --bg:#fcfcfb; --accent:#2a78d6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system, "Helvetica Neue", Arial, sans-serif; }
main { max-width:980px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:20px; margin:48px 0 8px;
     padding-top:16px; border-top:1px solid var(--rule); } h3 { font-size:16px; margin:28px 0 6px; }
.meta { color:var(--ink2); font-size:13px; }
figure { margin:16px 0 8px; } figure img { width:100%; height:auto; background:#fff; }
figcaption { font-size:13px; color:var(--ink2); margin-top:6px; }
table { border-collapse:collapse; font-size:13px; margin:12px 0; width:100%; }
th, td { padding:4px 8px; text-align:right; border-bottom:1px solid var(--rule); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { font-weight:600; } .tablewrap { overflow-x:auto; }
code { font-size:13px; background:#efeeea; padding:1px 4px; border-radius:3px; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--accent); }
nav ol { padding-left:20px; } a { color:var(--accent); }
.eq { text-align:center; margin:10px 0; } .legend { margin:14px 0; }
ol.refs li { margin:4px 0; }
.exec { background:#eef4fc; border-left:4px solid var(--accent); padding:8px 18px 8px 8px; margin:18px 0; }
.exec h2 { border:none; margin:6px 0 4px 10px; padding:0; } .exec li { margin:8px 0; font-size:16px; }
.ai { margin-top:40px; padding-top:12px; border-top:1px solid var(--rule); font-size:13px; color:var(--ink2); }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
@media print { details { display:block; } summary { display:none; } h2 { break-before:page; }
               figure { break-inside:avoid; } }
"""

AI_STATEMENT = (
    "<b>Use of AI tools.</b> Claude (Anthropic) was used as an AI programming assistant: to convert the original "
    "HOC model code to Python, and to help write the simulation scripts, including the script that grows the "
    "dendrite, as well as the figures and this report. The scientific questions, the choice of models and analyses, "
    "and the interpretation of the results are the author's. The author takes full and sole responsibility "
    "for all results and text.")


# --- helpers ---------------------------------------------------------------------------------------

def _svg(fig, save_as=None) -> str:
    if save_as:
        plotting.save_figure(fig, save_as, formats=("pdf", "png"))
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue().encode()).decode()
    return f'<img alt="figure" src="data:image/svg+xml;base64,{data}">'


def _figure(fig, caption, save_as=None) -> str:
    return f"<figure>{_svg(fig, save_as)}<figcaption>{caption}</figcaption></figure>"


def _value(res, morph, site, col):
    df = res.summary
    row = df[(df.morphology == morph) & (np.isclose(df.site_um, site))]
    return float(row[col].iloc[0]) if len(row) else np.nan


def _fmt(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def ca_boost(results: dict, prof: str):
    """Extra depolarisation due to dendritic Ca_LVA (long cell), max of synapse and tip, per site."""
    a = results[prof].summary.query("morphology == 'long'").set_index("site_um")
    b = results["none"].summary.query("morphology == 'long'").set_index("site_um")
    return np.maximum(a.peak_syn_mV - b.peak_syn_mV, a.peak_tip_mV - b.peak_tip_mV)


def event_sites(results: dict, prof: str, criterion=mechanism.EVENT_CRITERION_MV) -> list:
    """Sites whose synapse triggers a Ca_LVA event (> criterion mV added at the synapse or at the tip)."""
    if "none" not in results or prof == "none":
        return []
    x = ca_boost(results, prof)
    return [float(d) for d in x[x > criterion].index]


def max_boost(results: dict, prof: str) -> float:
    return float(ca_boost(results, prof).max())


def nearest_boosted(results: dict, prof: str, rel=0.10) -> float:
    """Nearest synapse whose somatic EPSP Ca_LVA increases by more than `rel`."""
    a = results[prof].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
    b = results["none"].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
    hit = (a / b - 1)[(a / b - 1) > rel].index
    return float(min(hit)) if len(hit) else np.nan


def _range(sites) -> str:
    return f"{min(sites):g}–{max(sites):g}" if sites else "none"


def _keys(grid, mouth=1.0):
    return sorted({(s, g) for s, g, m in grid if m == mouth}, key=lambda k: (k[0] != 0, k[1]))


def condition_table(grid: dict, mouth=1.0, metric="peak_soma_mV", unit="Somatic EPSP (mV)") -> str:
    head = "".join(f"<th>{s} µm</th>" for s in TABLE_SITES)
    rows = []
    first = grid[next(iter(grid))]
    for morph in ("short", "long"):
        cells = "".join(f"<td>{_fmt(_value(first['none'], morph, s, metric))}</td>" for s in TABLE_SITES)
        lab = "short (no dendritic Ca)" if morph == "short" else "long, no dendritic Ca"
        rows.append(f"<tr><td><span class='swatch' style='background:{plotting.INK_2}'></span>{lab}</td>"
                    f"<td>–</td><td>–</td>{cells}<td>–</td><td>–</td></tr>")
    for shift, g in _keys(grid, mouth):
        res = grid[(shift, g, mouth)]
        for prof, r in res.items():
            if prof == "none":
                continue
            cells = "".join(f"<td>{_fmt(_value(r, 'long', s, metric))}</td>" for s in TABLE_SITES)
            rows.append(f"<tr><td><span class='swatch' style='background:{plotting.PROFILE_COLORS[prof]}'></span>"
                        f"long, {prof}</td><td>{SHIFT_TXT.get(shift, shift)}</td><td>{g:g}</td>{cells}"
                        f"<td>{_range(event_sites(res, prof))}</td><td>{max_boost(res, prof):.1f}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th rowspan='2'>Cell, Ca<sub>LVA</sub> profile</th>"
            "<th rowspan='2'>Kinetics</th><th rowspan='2'>g (mS/cm²)</th>"
            f"<th colspan='{len(TABLE_SITES)}'>{unit} for a synapse at</th>"
            "<th rowspan='2'>Ca event for synapses at (µm)</th><th rowspan='2'>max Ca boost (mV)</th></tr>"
            f"<tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# --- sections -------------------------------------------------------------------------------------

def design_html(S: dict, cfg: Config) -> str:
    F = S["factors"]
    meta = load_target_meta()
    return f"""
<h2 id='design'>Design and methods in brief</h2>
<p><b>Model.</b> Human L2/3 PV+ interneuron (HL23PV) of Yao <i>et al.</i> (2022), translated to Python with
unchanged parameters; dendrites passive with I<sub>h</sub>. <b>Morphologies.</b> <i>Short</i>: the original
reconstruction, target dendrite ending {meta['short']['tip_distance_um']:.1f} µm from the soma. <i>Long</i>: the
same dendrite grown to {meta['long']['tip_distance_um']:.0f} µm (random walk, constant tip diameter).</p>
<div class='tablewrap'><table><thead><tr><th>Factor</th><th style='text-align:left'>Levels</th></tr></thead><tbody>
<tr><td>Dendritic Ca<sub>LVA</sub> profile (long cell only)</td><td style='text-align:left'>none;
{', '.join(F['profiles'])} (uniform: g everywhere; increasing: 0 at the soma, g at {cfg.gradient_span_um / 2:g} µm,
2g at the tip)</td></tr>
<tr><td>Density g</td><td style='text-align:left'>{', '.join(f'{g:g}' for g in F['g_ca_mS_cm2'])} mS/cm²</td></tr>
<tr><td>Ca<sub>LVA</sub> activation</td><td style='text-align:left'>original kinetics (half-activation of the m gate
−40 mV); shifted by −15 mV (−55 mV), inactivation unchanged</td></tr>
<tr><td>Proximal diameter (“mouth”)</td><td style='text-align:left'>original; ×{max(F['mouth_scale']):g} at the soma,
fading to ×1 at {cfg.mouth_length_um:g} µm</td></tr>
<tr><td>Input</td><td style='text-align:left'>one Exp2Syn synapse ({cfg.syn_weight_uS * 1e3:g} nS, τ
{cfg.syn_tau1_ms:g}/{cfg.syn_tau2_ms:g} ms) at a time, every {cfg.site_step_um:g} µm; single event or 5 events
at 50 Hz</td></tr>
<tr><td>Somatic current</td><td style='text-align:left'>none; steady bias at 50 or 95% of rheobase; mean
(80% of rheobase) + Ornstein–Uhlenbeck noise</td></tr>
</tbody></table></div>
<p class='meta'>An event is counted when dendritic Ca<sub>LVA</sub> adds more than
{mechanism.EVENT_CRITERION_MV:g} mV to the peak depolarisation at the synapse or at the tip of the dendrite (events
triggered by proximal synapses start distally). {cfg.celsius:g} °C, dt {cfg.dt_ms:g} ms, segments ≤
{cfg.max_seg_len_um:g} µm along the target dendrite, every run from the steady resting state.</p>"""


def geometry_html(cells: dict, mouth_cell, mouth_length_um: float) -> str:
    short, long = cells["short"], cells["long"]
    rows_l = path_geometry(long, reference=short)
    in_short = {r["section"] for r in path_geometry(short)}
    tr = []
    for r in rows_l:
        rng = (f"{r['diam_mean_um']:.2f}" if r["diam_max_um"] - r["diam_min_um"] < 0.005 else
               f"{r['diam_mean_um']:.2f} ({r['diam_min_um']:.2f}–{r['diam_max_um']:.2f})")
        tr.append(f"<tr><td>{r['section']}</td><td style='text-align:left'>{r['portion']}</td>"
                  f"<td>{r['start_um']:.1f}–{r['end_um']:.1f}</td><td>{r['length_um']:.1f}</td><td>{rng}</td>"
                  f"<td>{r['area_um2']:.0f}</td><td>{'✓' if r['section'] in in_short else '–'}</td><td>✓</td></tr>")
    a_s = sum(r["area_um2"] for r in path_geometry(short))
    a_l = sum(r["area_um2"] for r in rows_l)
    m0 = path_geometry(mouth_cell, reference=short)[0]
    table = ("<div class='tablewrap'><table><thead><tr><th>Section</th><th style='text-align:left'>Portion</th>"
             "<th>Path distance (µm)</th><th>Length (µm)</th><th>Diameter (µm), mean (range)</th>"
             "<th>Membrane area (µm²)</th><th>short</th><th>long</th></tr></thead><tbody>" + "".join(tr) +
             f"<tr><td><b>total</b></td><td></td><td></td><td>{short.tip_distance:.1f} / {long.tip_distance:.1f}</td>"
             f"<td></td><td>{a_s:.0f} / {a_l:.0f}</td><td></td><td></td></tr></tbody></table></div>")
    fig = plotting.diameter_figure(cells, mouth_cell=mouth_cell)
    return (f"<h3 id='geometry'>Diameters and membrane area of the target dendrite</h3>{table}"
            f"<p>The target dendrite tapers from {rows_l[0]['diam_max_um']:.2f} µm at the soma to "
            f"{rows_l[2]['diam_min_um']:.2f}–{rows_l[2]['diam_max_um']:.2f} µm beyond {rows_l[1]['end_um']:.0f} µm; "
            f"the grown part keeps the tip diameter. With the wider mouth, the first section measures "
            f"{m0['diam_max_um']:.2f} µm at the soma (instead of {rows_l[0]['diam_max_um']:.2f}) and the extra width "
            f"fades to zero at {mouth_length_um:g} µm.</p>"
            + _figure(fig, "<b>Target dendrite geometry.</b> <b>a</b>, Diameter along the target dendrite "
                      "(reconstruction points): short (dashed), long (solid), long with the wider mouth "
                      "(dash-dot); grey, every other soma-to-tip path. <b>b</b>, Cumulative membrane area; "
                      "dotted line, soma.", "morphology/diameter"))


def morphology_html(cells, mouth_cell, mouth_length_um) -> str:
    fig3d = viewer3d.morphology_3d(cells["long"], cells["short"].tip_distance)
    return ("<h2 id='morphology'>Morphologies</h2>"
            + _figure(plotting.dendrogram_figure(cells),
                      "<b>Dendrograms.</b> Every basal branch at its path distance from the soma; the target "
                      f"dendrite is highlighted. Short cell: tip at {cells['short'].tip_distance:.1f} µm; long cell: "
                      f"the same dendrite continues from the original tip (tick) to {cells['long'].tip_distance:.1f} µm.",
                      "morphology/dendrogram")
            + geometry_html(cells, mouth_cell, mouth_length_um)
            + "<figure>" + fig3d.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
            + "<figcaption><b>Interactive 3D view</b> of the long cell: drag to rotate, scroll to zoom. Black: "
              "original target dendrite; blue: grown extension (click it in the legend to hide it).</figcaption>"
              "</figure>")


def electrotonic_html(cfg: Config, single: dict, mech_cfg: Config) -> str:
    imp = {"short": mechanism.impedance_profile("short", cfg), "long": mechanism.impedance_profile("long", cfg),
           "long, 1.5× mouth": mechanism.impedance_profile("long", cfg.replace(mouth_scale=1.5))}
    none = {"short": single[(0.0, 1.0, 1.0)]["none"], "long": single[(0.0, 1.0, 1.0)]["none"],
            "long, 1.5× mouth": single[(0.0, 1.0, 1.5)]["none"]}
    ca = single[(mech_cfg.ca_act_shift_mV, mech_cfg.g_ca_mS_cm2, 1.0)][mech_cfg.ca_profile]
    fig = plotting.electrotonic_figure(imp, none, ca, f"{mech_cfg.ca_profile}, {mech_cfg.g_ca_mS_cm2:g} mS/cm², "
                                                     f"{SHIFT_TXT[mech_cfg.ca_act_shift_mV]}")
    zl, zs = imp["long"], imp["short"]
    z = lambda df, d, c="zin_0Hz_MOhm": float(df.iloc[int(np.argmin(np.abs(df.distance_um - d)))][c])  # noqa: E731
    n = none["long"].summary.query("morphology == 'long'").set_index("site_um")
    c = ca.summary.query("morphology == 'long'").set_index("site_um")
    return f"""
<h2 id='electrotonic'>Electrotonic structure and EPSP time course along the dendrite</h2>
<ul>
<li>The local input impedance rises from {z(zl, 10):.0f} MΩ at 10 µm to {z(zl, 100):.0f} MΩ at 100 µm and
{z(zl, 400):.0f} MΩ at the long tip (soma: {zl.zin_soma_0Hz_MOhm.iloc[0]:.1f} MΩ); at 100 Hz it is lower and flatter.
The short tip has {z(zs, 100):.0f} MΩ, higher than the long dendrite at the same distance, because it ends there.</li>
<li>The transfer impedance to the soma falls only from {z(zl, 10, 'ztr_0Hz_MOhm'):.0f} to
{z(zl, 400, 'ztr_0Hz_MOhm'):.0f} MΩ at 0 Hz, but much more steeply at 100 Hz: slow signals reach the soma much
better than fast ones.</li>
<li>Without Ca<sub>LVA</sub>, the local EPSP lasts longer the farther it is from the soma (effective time constant
{n.tau_eff_ms_syn[10]:.1f} ms at 10 µm, {n.tau_eff_ms_syn[200]:.1f} ms at 200 µm, {n.tau_eff_ms_syn[400]:.1f} ms at
400 µm), while the somatic EPSP is broadened by dendritic filtering ({n.tau_eff_ms_soma[10]:.1f} →
{n.tau_eff_ms_soma[400]:.1f} ms). With Ca<sub>LVA</sub> ({mech_cfg.ca_profile}, {mech_cfg.g_ca_mS_cm2:g} mS/cm²,
{SHIFT_TXT[mech_cfg.ca_act_shift_mV]}), the somatic EPSP integral at 300 µm rises from
{n.area_mVms_soma[300]:.0f} to {c.area_mVms_soma[300]:.0f} mV·ms.</li>
<li>The wider mouth lowers the impedance near the soma only (at 10 µm: {z(imp['long, 1.5× mouth'], 10):.0f} against
{z(zl, 10):.0f} MΩ) and leaves the distal dendrite unchanged.</li>
</ul>""" + _figure(fig, "<b>Electrotonic structure.</b> <b>a</b>, Local input impedance along the target dendrite "
                        "(thick, 0 Hz; thin, 100 Hz; dotted, soma). <b>b</b>, Transfer impedance between each point and "
                        "the soma. <b>c</b>, Steady-state voltage attenuation from each point to the soma. <b>d–f</b>, "
                        "EPSP integral (log scale), effective time constant (integral / peak) and decay time constant "
                        "(exponential fit between 80% and 20% of the peak) against synapse distance, at the soma (thick) "
                        "and at the synapse (thin), without dendritic Ca<sub>LVA</sub> (grey and blue lines: short, long, "
                        "long with wider mouth) and with it (colour).", "electrotonic/electrotonic")


def single_html(single: dict) -> str:
    out = ["<h2 id='single'>Single synaptic events: density × kinetics × profile</h2>",
           condition_table(single),
           _figure(plotting.factorial_figure(single),
                   "<b>Single events.</b> Columns: Ca<sub>LVA</sub> density and kinetics. <b>a–d</b>, Somatic EPSP "
                   "against synapse distance: short cell (dashed grey), long cell without dendritic Ca<sub>LVA</sub> "
                   "(dark grey), and with the uniform (blue) or increasing (green) profile. <b>e–h</b>, Extra local "
                   "depolarisation due to Ca<sub>LVA</sub>; dotted line, event criterion.", "single/factorial")]
    for (shift, g), res in ((k, single[(k[0], k[1], 1.0)]) for k in _keys(single)):
        cells = {m: get_cell(m, res["none"].config) for m in ("short", "long")}
        parts = []
        for prof, r in res.items():
            if prof == "none":
                continue
            parts.append(_figure(plotting.overview_figure(r, cells), plotting.describe(r.config).replace("$", ""),
                                 f"single/overview_{prof}_g{g:g}_s{shift:g}"))
        out.append(f"<details><summary>Detail: {g:g} mS/cm², {SHIFT_TXT[shift]} kinetics</summary>"
                   + "".join(parts) + "</details>")
    return "".join(out)


def mouth_html(single: dict) -> str:
    fig = plotting.mouth_figure(single)
    if fig is None:
        return ""
    ch = []
    for (shift, g) in _keys(single):
        for prof in ("none", "uniform", "increasing"):
            a = single[(shift, g, 1.0)][prof].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
            b = single[(shift, g, 1.5)][prof].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
            ch.append((100 * (b / a - 1)).values)
    ch = np.concatenate(ch)
    near = single[(0.0, 1.0, 1.0)]["none"].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
    near_w = single[(0.0, 1.0, 1.5)]["none"].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
    ev = {k: (_range(event_sites(single[(k[0], k[1], 1.0)], 'increasing')),
              _range(event_sites(single[(k[0], k[1], 1.5)], 'increasing'))) for k in _keys(single)}
    return (f"<h2 id='mouth'>A wider dendritic mouth</h2><ul>"
            f"<li>Widening the first {single[(0.0, 1.0, 1.5)]['none'].config.mouth_length_um:g} µm of the target dendrite (×1.5 at the soma) changes somatic EPSPs by "
            f"{ch.min():+.1f}% to {ch.max():+.1f}% across all sites and conditions; the largest change is for the most "
            f"proximal synapses (10 µm: {near[10]:.2f} → {near_w[10]:.2f} mV without Ca<sub>LVA</sub>).</li>"
            f"<li>The sites that trigger a Ca<sub>LVA</sub> event are unchanged (increasing profile: "
            + "; ".join(f"{g:g} mS/cm² {SHIFT_TXT[s]}: {a} → {b}" for (s, g), (a, b) in ev.items()) + ").</li></ul>"
            + _figure(fig, "<b>Wider mouth.</b> Relative change of the somatic EPSP with the 1.5× mouth, against "
                           "synapse distance, for each density and kinetics.", "single/mouth"))


def train_html(train: dict) -> str:
    return ("<h2 id='train'>Trains of 5 inputs at 50 Hz</h2>"
            + condition_table(train, unit="Largest somatic depolarisation (mV)")
            + _figure(plotting.factorial_figure(train),
                      "<b>Trains.</b> As the single-event figure, for 5 inputs at 50 Hz; amplitudes are the largest "
                      "depolarisation during the train.", "train/factorial"))


def _fired(res, morph="long"):
    s = res.summary
    return s[(s.morphology == morph) & (s.n_spikes_soma > 0)].site_um.tolist()


def bias_html(bias: dict, S: dict) -> str:
    out = ["<h2 id='bias'>Firing with a steady somatic current</h2>"
           f"<p class='meta'>{S['bias']['note']} Density {', '.join(f'{g:g}' for g in S['bias']['g_ca_mS_cm2'])} "
           "mS/cm². Cells: synapse sites that make the cell fire at least one spike.</p>"]
    for (shift, g), study in bias.items():
        rows = []
        for frac, res in sorted(study.items()):
            sl = res["none"].summary.query("morphology == 'long'")
            b = sl.bias_nA.iloc[0] if "bias_nA" in sl else 0.0
            cells = "".join(f"<td>{_range(_fired(r)) if _fired(r) else 'none'}</td>" for r in res.values())
            short_f = _fired(res["none"], "short")
            rows.append(f"<tr><td>{100 * frac:.0f}%</td><td>{b:.3f}</td><td>{sl.vrest_soma_mV.iloc[0]:.1f}</td>"
                        f"<td>{_range(short_f) if short_f else 'none'}</td>{cells}</tr>")
        head = "".join(f"<th>long, {p}</th>" for p in next(iter(study.values())))
        out.append(f"<h3>{g:g} mS/cm², {SHIFT_TXT[shift]} kinetics</h3>"
                   "<div class='tablewrap'><table><thead><tr><th>Bias (of rheobase)</th><th>Bias (nA)</th>"
                   f"<th>Rest (mV)</th><th>short</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
        tag = f"g{g:g}_s{shift:g}"
        out.append(_figure(plotting.spike_traces_figure(study, max(study)),
                           f"<b>Action potentials evoked by single synapses</b> at {100 * max(study):.0f}% of rheobase "
                           f"({g:g} mS/cm², {SHIFT_TXT[shift]} kinetics). Top, soma; bottom, synapse; synapses every 20 µm "
                           "(colour: distance). Thick traces: the cell fires.", f"bias/spike_traces_{tag}"))
        out.append("<details><summary>Spikes and Ca<sub>LVA</sub> boost for every bias level</summary>"
                   + _figure(plotting.firing_figure(study, S["bias"]["example_site_um"]),
                             "One row per bias level. Left, somatic potential for a synapse at "
                             f"{S['bias']['example_site_um']:g} µm; middle, somatic spikes against synapse distance; "
                             "right, local Ca<sub>LVA</sub> boost.", f"bias/firing_{tag}") + "</details>")
    return "".join(out)


def noise_html(spikes, examples, S: dict) -> str:
    N = S["noise"]
    ev = studies.evoked_spikes(spikes, N)
    rows = []
    for cond, grp in ev.groupby("condition", sort=False):
        cells = "".join(
            (f"<td>{float(g.evoked.iloc[0]):.3f} [{float(g.ci_lo.iloc[0]):.3f}, {float(g.ci_hi.iloc[0]):.3f}]</td>"
             if len(g := grp[grp.site_um == d]) else "<td>–</td>") for d in N["sites_um"])
        rows.append(f"<tr><td>{cond.replace('Ca_LVA', 'Ca<sub>LVA</sub>')}</td>{cells}</tr>")
    head = "".join(f"<th>{d:g}</th>" for d in N["sites_um"])
    n_inputs = int(ev.n_inputs.iloc[0])
    c0, c1 = N["count_window_ms"]
    table = ("<div class='tablewrap'><table><thead><tr><th rowspan='2'>Condition</th>"
             f"<th colspan='{len(N['sites_um'])}'>Extra spikes per input ({c0:g}–{c1:g} ms), mean [95% CI], for a synapse "
             f"at (µm)</th></tr><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    contrast = studies.contrast_example(spikes, S)
    ps5 = studies.psth(spikes, {**N, "psth_bin_ms": 5.0})
    fig = plotting.noise_figure(ps5, ev, {**examples, "contrast": contrast}, {**N, "psth_bin_ms": 5.0},
                                example_conditions=list(studies.EXAMPLE_PAIR),
                                cumulative=studies.cumulative_extra(spikes, N))
    el = studies.early_late(spikes, N).set_index(["condition", "site_um"])
    site = N["example_site_um"]
    split = "".join(f"<tr><td>{c.replace('Ca_LVA', 'Ca<sub>LVA</sub>')}</td><td>{el.early[(c, site)]:.3f}</td>"
                    f"<td>{el.late[(c, site)]:.3f}</td></tr>" for c in dict.fromkeys(ev.condition) if (c, site) in el.index)
    table += ("<p>Dendritic Ca<sub>LVA</sub> starts to add current within about 1 ms of the input, while the EPSP is "
              "still rising, and its regenerative depolarisation reaches the soma later and lasts longer than the passive "
              "EPSP. It therefore increases the extra spikes both in the first 20 ms and, even more, in the following "
              "tens of milliseconds. For a synapse at "
              f"{site:g} µm, extra spikes per input:</p><div class='tablewrap'><table><thead><tr><th>Condition</th>"
              f"<th>0–20 ms</th><th>20–{N['count_window_ms'][1]:g} ms</th></tr></thead><tbody>{split}</tbody></table></div>")
    base_rate = np.mean([np.mean([len(x[(x >= -50) & (x < 0)]) for x in spikes[k]]) / 0.05
                         for k in spikes if k[1] is None])
    return (f"<h2 id='noise'>Firing with a noisy somatic current</h2><p class='meta'>{N['note']} Mean "
            f"{100 * N['mu_frac']:.0f}% and s.d. {100 * N['sigma_frac']:.0f}% of each cell's rheobase, correlation time "
            f"{N['tau_ms']:g} ms; background firing about {base_rate:.1f} Hz. {n_inputs} synaptic inputs per site "
            f"({N['n_blocks']} blocks × {N['inputs_per_block']} inputs, one every {N['period_ms']:g} ms); Ca<sub>LVA</sub> "
            f"{N['g_ca_mS_cm2']:g} mS/cm². Extra spikes = spikes with the synapse − spikes with the same noise and no "
            "synapse; confidence intervals by bootstrap over inputs.</p>" + table
            + _figure(fig, "<b>Noisy somatic current.</b> <b>a, b</b>, One synaptic input (dotted line) at "
                           f"{N['example_site_um']:g} µm, with the same noise seed in both cells (mean and amplitude scaled "
                           "to each cell's rheobase): grey, noise only; colour, noise + synapse. Without dendritic "
                           "Ca<sub>LVA</sub> (a) the synapse does not fire the cell; with increasing, shifted Ca<sub>LVA</sub> "
                           "(b) it does. This pattern occurred for "
                           f"{contrast.get('n_found', 0)} of {contrast.get('n_inputs', 0)} inputs (block "
                           f"{contrast.get('block', '–')}, input {contrast.get('input', '–')} shown). "
                           "<b>c–e</b>, Cumulative extra spikes per input (synapse − same noise without synapse) from the "
                           "input onwards, mean and bootstrap 95% CI, for synapses at 100, 200 and 300 µm; the value at "
                           f"{N['count_window_ms'][1]:g} ms (dotted line) is the one plotted in <b>f</b>, and the slope shows "
                           "when the extra spikes occur. <b>f</b>, Extra spikes per input against synapse distance, mean and "
                           "95% CI. <b>g</b>, Firing rate around the input (5 ms bins) for a synapse at "
                           f"{N['example_site_um']:g} µm, with (solid) and without (dashed) the synapse; the background "
                           "firing is included.", "noise/noise"))


def _threshold_sentence(th) -> str:
    fail = th[th.threshold_nS.isna()].distance_um
    ok = th[th.threshold_nS.notna()].sort_values("distance_um")
    parts = []
    if len(fail):
        parts.append(f"No synapse at {', '.join(f'{d:g}' for d in fail)} µm triggers an event up to "
                     f"{max(mechanism.SCAN_WEIGHTS)} nS")
    if len(ok):
        near, far = ok.iloc[0], ok.iloc[-1]
        parts.append(f"the smallest synaptic weight that triggers one rises from {far.threshold_nS:g} nS at "
                     f"{far.distance_um:g} µm to {near.threshold_nS:g} nS at {near.distance_um:g} µm")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] if s else ""


def mechanism_html(D: dict) -> str:
    cfg = D["cfg"]
    th = D["thresholds"]
    onset = {t: mechanism.onset_um(df) for t, df in D["tau"].items()}
    el = D["epsp"]["long"]
    t_above = lambda d: float(el.iloc[int(np.argmin(np.abs(el.distance_um - d)))].t_above_ms)  # noqa: E731
    u, n = D["example"][cfg.ca_profile], D["example"]["none"]
    gv = D["gating"]
    tm = gv[(gv.v >= -70) & (gv.v <= -50)].m_tau
    half = float(gv.v[np.argmin(np.abs(gv.m_inf - 0.5))])
    k = int(np.argmax(u["v_syn"]))
    cond = f"{cfg.ca_profile}, {cfg.g_ca_mS_cm2:g} mS/cm², {SHIFT_TXT[cfg.ca_act_shift_mV]} kinetics"
    return f"""
<h2 id="mechanism">Mechanism: what decides which synapses trigger a Ca<sub>LVA</sub> event</h2>
<p class='meta'>Analyses for the condition with robust events: {cond}.</p>
<ol>
<li><b>The channel is slow and mostly inactivated at rest.</b> The m gate is half-activated at {half:.0f} mV, with a
time constant of {tm.min():.1f}–{tm.max():.1f} ms between −70 and −50 mV; at rest ({D['rest_mV']:.1f} mV) only
{100 * D['h_rest']:.0f}% of the channels are available (M a, b).</li>
<li><b>The soma is a strong current sink; distally the dendrite is isolated</b> (M c), so the passive local EPSP lasts
longer with distance: {t_above(100):.1f} ms above −50 mV at 100 µm, {t_above(200):.1f} ms at 200 µm and
{t_above(400):.1f} ms at 400 µm, while its peak levels off (M d, e).</li>
<li><b>Proximal synapses need much stronger, or longer, input.</b> An event is counted when Ca<sub>LVA</sub> adds
more than {mechanism.EVENT_CRITERION_MV:g} mV at the synapse or at the tip. {_threshold_sentence(th)} (M f).
Slowing the synaptic decay moves the onset towards the soma:
{', '.join(f'τ<sub>decay</sub> {t:g} ms → {onset[t]:.0f} µm' for t in sorted(onset))} (M g).</li>
<li><b>The event is regenerative and self-terminating.</b> At {D['example_site']:g} µm the passive local EPSP peaks at
{n['v_syn'].max():.0f} mV; with Ca<sub>LVA</sub> the membrane reaches {u['v_syn'].max():.0f} mV
{u['t'][k] - cfg.onset_ms:.0f} ms after the input, and the inactivation gate falls from {u['h'][0]:.2f} to
{u['h'].min():.3f} (M h, i).</li>
</ol>""" + _figure(plotting.mechanism_figure(D),
                   "<b>Figure M.</b> <b>a</b>, Steady-state activation (m∞², dashed: original kinetics) and "
                   "availability (h∞); dotted line, rest. <b>b</b>, Time constants. <b>c</b>, Local input impedance. "
                   "<b>d, e</b>, Peak and time above −50 mV of the passive local EPSP. <b>f</b>, Smallest synaptic weight "
                   "that triggers an event (triangles: none up to the largest weight tested). <b>g</b>, Local boost for "
                   "different synaptic decay time constants. <b>h</b>, Local potential with and without "
                   "Ca<sub>LVA</sub>; <b>i</b>, gates.", "mechanism/mechanism")


# --- summary --------------------------------------------------------------------------------------

def executive_html(single, train, bias, noise_ev, D, S) -> str:
    n = single[(0.0, 1.0, 1.0)]["none"]
    items = [
        f"<b>A longer dendrite loses its distal inputs.</b> Without Ca<sub>LVA</sub>, a synapse at the 400 µm tip moves "
        f"the soma by {_value(n, 'long', 400, 'peak_soma_mV'):.2f} mV, against {_value(n, 'long', 10, 'peak_soma_mV'):.1f} "
        f"mV near the soma; local impedance rises steeply with distance and the local EPSP lasts longer."]
    for (shift, g) in _keys(single):
        res = single[(shift, g, 1.0)]
        parts = []
        for prof in ("uniform", "increasing"):
            sites = event_sites(res, prof)
            s300 = _value(res[prof], "long", 300, "peak_soma_mV")
            parts.append(f"{prof}: {'events at ' + _range(sites) + ' µm' if sites else 'no event'} "
                         f"(soma at 300 µm {s300:.2f} mV)")
        items.append(f"<b>{g:g} mS/cm², {SHIFT_TXT[shift]} kinetics</b>: " + "; ".join(parts) +
                     f"; no Ca: {_value(n, 'long', 300, 'peak_soma_mV'):.2f} mV.")
    near = {(s, g, p): nearest_boosted(single[(s, g, 1.0)], p) for s, g in _keys(single) for p in ("uniform", "increasing")}
    shifted_ev = [d for s, g in _keys(single) if s != 0 for p in ("uniform", "increasing")
                  for d in event_sites(single[(s, g, 1.0)], p)]
    shifted_min = min(shifted_ev) if shifted_ev else np.nan
    items.append("<b>The closest synapses are never boosted; how close the boost reaches depends on the channel.</b> "
                 "Nearest synapse whose somatic EPSP grows by more than 10%: "
                 + "; ".join(f"{g:g} mS/cm² {SHIFT_TXT[s]}, {p}: {_fmt(d, 0)} µm" for (s, g, p), d in near.items())
                 + f". With shifted activation, synapses as close as {_fmt(shifted_min, 0)} µm trigger an event, which "
                   "then starts in the distal dendrite rather than at the synapse.")
    items.append("<b>The wider mouth is almost irrelevant</b>: it strengthens proximal synapses by a few percent and "
                 "does not change where events occur.")
    for (shift, g), study in bias.items():
        hi = max(study)
        fired = {p: _fired(r) for p, r in study[hi].items()}
        items.append(f"<b>Near threshold ({100 * hi:.0f}% of rheobase), {g:g} mS/cm², {SHIFT_TXT[shift]} kinetics</b>: "
                     "synapses that fire the cell are at "
                     + ", ".join(f"{_range(v) if v else 'none'} µm ({p})" for p, v in fired.items()) + ".")
    if noise_ev is not None:
        mid = noise_ev[(noise_ev.site_um >= 150) & (noise_ev.site_um <= 300)].groupby("condition", sort=False).evoked.mean()
        items.append(f"<b>With a noisy somatic current</b> (mean {100 * S['noise']['mu_frac']:.0f}% of rheobase, "
                     f"{int(noise_ev.n_inputs.iloc[0])} inputs per site), extra spikes per input for synapses at "
                     "150–300 µm: " + ", ".join(f"{v:.3f} ({c.replace('Ca_LVA', 'Ca<sub>LVA</sub>')})"
                                                for c, v in mid.items()) + ".")
    return ("<section class='exec'><h2 id='summary'>Key results</h2><ol>" + "".join(f"<li>{i}</li>" for i in items)
            + "</ol></section>")


# --- build ----------------------------------------------------------------------------------------

def build(set_file=None, out=None, progress=None, write_html=True) -> Path:
    S = studies.spec(set_file)
    out = Path(out or FIGURES_DIR / "report.html")
    cfg = studies.base_config(S)
    M = S["mechanism"]
    mech_cfg = cfg.replace(name="mechanism", ca_profile=M["profile"], g_ca_mS_cm2=M["g_ca_mS_cm2"],
                           ca_act_shift_mV=M["ca_act_shift_mV"])

    single = studies.grid("single", S, progress)
    train = studies.grid("train", S, progress)
    bias = studies.bias_grid(S, progress)
    noise_sp, noise_ex = studies.noise_study(S, progress)
    noise_ev = studies.evoked_spikes(noise_sp, S["noise"])
    el = studies.early_late(noise_sp, S["noise"]).set_index(["condition", "site_um"])
    noise_split = None
    if ("long, increasing, shifted", 300) in el.index:
        noise_split = dict(inc="increasing", e0=el.early[("long, no Ca_LVA", 300)], l0=el.late[("long, no Ca_LVA", 300)],
                           e1=el.early[("long, increasing, shifted", 300)], l1=el.late[("long, increasing, shifted", 300)])
    if progress:
        progress("mechanism analyses")
    D = mechanism.collect(mech_cfg)

    cells = {m: get_cell(m, cfg) for m in ("short", "long")}
    mouth_cell = get_cell("long", cfg.replace(mouth_scale=max(S["factors"]["mouth_scale"])))
    sections = [
        ("design", "Design and methods in brief", design_html(S, cfg)),
        ("morphology", "Morphologies", morphology_html(cells, mouth_cell, cfg.mouth_length_um)),
        ("electrotonic", "Electrotonic structure and EPSP time course", electrotonic_html(cfg, single, mech_cfg)),
        ("single", "Single synaptic events", single_html(single)),
        ("mouth", "A wider dendritic mouth", mouth_html(single)),
        ("mechanism", "Mechanism", mechanism_html(D)),
        ("train", "Trains", train_html(train)),
        ("bias", "Firing with a steady somatic current", bias_html(bias, S)),
        ("noise", "Firing with a noisy somatic current", noise_html(noise_sp, noise_ex, S)),
        ("manuscript", "Draft manuscript material",
         manuscript.html(single, train, bias, noise_ev, D, cells, cfg, S, noise_split)),
    ]
    toc = "".join(f"<li><a href='#{a}'>{t}</a></li>" for a, t, _ in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PV dendrite report</title><style>{CSS}</style></head>
<body><main>
<h1>Short (100 µm) vs grown (400 µm) dendrite of a human PV+ interneuron</h1>
<p class="meta">Generated {dt.datetime.now():%Y-%m-%d %H:%M} · pvdend {__version__} · NEURON {neuron.__version__}
· studies <code>configs/studies.json</code></p>
{executive_html(single, train, bias, noise_ev, D, S)}
<nav><ol>{toc}</ol></nav>
{''.join(body for _, _, body in sections)}
<p class="ai">{AI_STATEMENT}</p>
</main></body></html>"""
    if write_html:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page)
    return out
