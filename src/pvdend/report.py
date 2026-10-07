"""One self-contained HTML report with every figure set in configs/figure_set.json.

Figures are drawn by the same functions as scripts/ and the notebook, from the
cached results (results/), and embedded as vector SVG.
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import neuron
import numpy as np

from . import __version__, mechanism, plotting
from ._paths import CONFIG_DIR, FIGURES_DIR
from .config import Config
from .morphology import load_target_meta
from .protocols import get_cell, load_or_run

TABLE_SITES = (10, 100, 200, 300, 400)

CAPTION_COMPARISON = (
    "<b>a</b>, Ca<sub>LVA</sub> density along the target dendrite of the long morphology for each profile. "
    "<b>b</b>, Peak somatic EPSP as a function of synapse distance from the soma. Dashed grey: short "
    "morphology (no dendritic Ca<sub>LVA</sub>); solid: long morphology, one colour per profile. "
    "<b>c</b>, Peak local EPSP at the synapse. <b>d</b>, Attenuation, somatic / local peak (log scale). "
    "<b>e</b>, Peak local change in [Ca<sup>2+</sup>]<sub>i</sub> at the synapse. "
    "<b>f</b>, Area of the somatic depolarisation above rest. "
    "Where a coloured curve leaves the dark-grey <i>none</i> curve, the synapse triggers a regenerative "
    "Ca<sub>LVA</sub> event in the dendrite (large jump in <b>c</b> and <b>e</b>); the event spreads to the soma "
    "and lifts the somatic EPSP (<b>b</b>, <b>f</b>). Why this happens only beyond a certain distance is "
    "explained in the <a href='#mechanism'>Mechanism</a> section.")
CAPTION_OVERVIEW = (
    "<b>a, b</b>, Short and long morphologies; the target dendrite is highlighted (in <b>b</b> coloured by "
    "Ca<sub>LVA</sub> density) and circles mark the synapse sites whose traces are shown. "
    "<b>c</b>, Ca<sub>LVA</sub> density along the long dendrite. <b>d, e</b>, Somatic membrane potential for "
    "synapses at increasing distance (colour scale). <b>f</b>, Membrane potential at the synapse, long "
    "morphology. <b>g–i</b>, Somatic EPSP, local EPSP and attenuation versus synapse distance. "
    "In <b>f</b>, traces with a delayed second hump are synapses that trigger a Ca<sub>LVA</sub> event: the "
    "synaptic EPSP comes first, the Ca<sub>LVA</sub> depolarisation builds up a few milliseconds later.")

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
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
@media print { details { display:block; } summary { display:none; } h2 { break-before:page; }
               figure { break-inside:avoid; } }
"""


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue().encode()).decode()
    return f'<img alt="figure" src="data:image/svg+xml;base64,{data}">'


def describe_html(cfg: Config) -> str:
    stim = "single event" if cfg.n_events == 1 else f"{cfg.n_events} events at {cfg.freq_hz:g} Hz"
    norm = ("peak densities (uniform {:g}, others {:g} S/cm²)".format(cfg.g_uniform, cfg.g_peak)
            if cfg.ca_norm == "peak" else
            f"equal total conductance (uniform-equivalent {cfg.g_total_equiv:g} S/cm²)")
    parts = [f"Exp2Syn {cfg.syn_weight_uS * 1e3:g} nS (τ {cfg.syn_tau1_ms:g}/{cfg.syn_tau2_ms:g} ms), {stim}",
             f"dendritic Ca<sub>LVA</sub> normalised by {norm}",
             f"somatic Ca<sub>LVA</sub> {'on' if cfg.somatic_ca_lva else 'off'}",
             "TTX" if cfg.ttx else "no TTX",
             f"sites every {cfg.site_step_um:g} µm from {cfg.site_start_um:g} µm"]
    if cfg.prune_side_branches:
        parts.append("side branches pruned")
    return "; ".join(parts) + "."


def _value(res, morph, site, col):
    df = res.summary
    row = df[(df.morphology == morph) & (np.isclose(df.site_um, site))]
    return row[col].iloc[0] if len(row) else np.nan


def _fmt(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def event_sites(results: dict, prof: str, criterion=mechanism.EVENT_CRITERION_MV) -> list:
    """Sites (long cell) where dendritic Ca_LVA adds > criterion mV to the local EPSP."""
    if "none" not in results or prof == "none":
        return []
    a = results[prof].summary.query("morphology == 'long'").set_index("site_um").peak_syn_mV
    b = results["none"].summary.query("morphology == 'long'").set_index("site_um").peak_syn_mV
    return [float(d) for d in (a - b)[(a - b) > criterion].index]


def _range(sites) -> str:
    return f"{min(sites):g}–{max(sites):g}" if sites else "none"


def summary_table(results: dict) -> str:
    head = "".join(f"<th>{s} µm</th>" for s in TABLE_SITES)
    rows = []
    first = next(iter(results.values()))
    if "short" in first.config.morphologies:
        cells = "".join(f"<td>{_fmt(_value(first, 'short', s, 'peak_soma_mV'))}</td>" for s in TABLE_SITES)
        n_spk = int(first.summary[first.summary.morphology == "short"].n_spikes_soma.sum())
        rows.append(f"<tr><td><span class='swatch' style='background:{plotting.INK_2}'></span>"
                    f"short (no dendritic Ca)</td>{cells}<td>–</td><td>–</td>"
                    f"<td>{n_spk}</td></tr>")
    for prof, res in results.items():
        long = res.summary[res.summary.morphology == "long"]
        cells = "".join(f"<td>{_fmt(_value(res, 'long', s, 'peak_soma_mV'))}</td>" for s in TABLE_SITES)
        rows.append(f"<tr><td><span class='swatch' style='background:{plotting.PROFILE_COLORS[prof]}'></span>"
                    f"long, {prof}</td>{cells}<td>{_range(event_sites(results, prof)) if prof != 'none' else '–'}</td>"
                    f"<td>{_fmt(long.dcai_syn_uM.max(), 3)}</td>"
                    f"<td>{int(long.n_spikes_soma.sum())}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th rowspan='2'>Morphology, Ca profile</th>"
            f"<th colspan='{len(TABLE_SITES)}'>Somatic EPSP (mV) for a synapse at</th>"
            "<th rowspan='2'>Ca event for synapses at (µm)</th>"
            "<th rowspan='2'>max local Δ[Ca<sup>2+</sup>]<sub>i</sub> (µM)</th>"
            "<th rowspan='2'>somatic spikes (all sites)</th></tr>"
            f"<tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def set_findings(results: dict, note: str = "") -> str:
    """Short, data-driven list of what each figure set shows."""
    first = next(iter(results.values()))
    items = []
    if "none" in results:
        n = results["none"]
        items.append(
            f"Without dendritic Ca<sub>LVA</sub>, the somatic EPSP of the long cell falls from "
            f"{_value(n, 'long', 10, 'peak_soma_mV'):.2f} mV (synapse at 10 µm) to "
            f"{_value(n, 'long', 400, 'peak_soma_mV'):.2f} mV (400 µm); in the short cell it is "
            f"{_value(n, 'short', 100, 'peak_soma_mV'):.2f} mV at its 100 µm tip, against "
            f"{_value(n, 'long', 100, 'peak_soma_mV'):.2f} mV at 100 µm in the long cell, which has more "
            "membrane beyond that point to charge.")
    for prof, res in results.items():
        if prof == "none":
            continue
        sites = event_sites(results, prof)
        long = res.summary[res.summary.morphology == "long"]
        if sites:
            ev = long[long.site_um.isin(sites)]
            items.append(f"<b>{prof}</b>: Ca<sub>LVA</sub> event for synapses at {_range(sites)} µm; somatic EPSP "
                         f"{ev.peak_soma_mV.min():.1f}–{ev.peak_soma_mV.max():.1f} mV for those synapses.")
        else:
            items.append(f"<b>{prof}</b>: no Ca<sub>LVA</sub> event at any site.")
    spikes = int(sum(r.summary.n_spikes_soma.sum() for r in results.values()))
    items.append("No somatic action potentials at any site." if spikes == 0
                 else f"{spikes} somatic action potentials in total across sites and profiles.")
    out = "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    if note:
        out += f"<p>{note}</p>"
    return out


def _at(df, col, d, xcol="distance_um"):
    return float(df.iloc[int(np.argmin(np.abs(df[xcol].to_numpy() - d)))][col])


def mechanism_html(D: dict, ref: dict) -> str:
    """Explanation of the distance dependence of the Ca_LVA event, with numbers from mechanism.collect."""
    cfg = D["cfg"]
    zl, zs = D["impedance"]["long"], D["impedance"]["short"]
    el = D["epsp"]["long"]
    th = D["thresholds"]
    fail = th[th.threshold_nS.isna()].distance_um
    ok = th[th.threshold_nS.notna()]
    tmin = ok.threshold_nS.min()
    onset = {t: mechanism.onset_um(df) for t, df in D["tau"].items()}
    pon = {p: mechanism.onset_um(df) for p, df in D["profiles"].items()}
    hs = D["profiles"]["hotspot"]
    hs_sites = hs[hs.extra_local_mV > mechanism.EVENT_CRITERION_MV].distance_um
    lo, hi = cfg.hotspot_center_um - cfg.hotspot_width_um / 2, cfg.hotspot_center_um + cfg.hotspot_width_um / 2
    ex, site, on = D["example"], D["example_site"], cfg.onset_ms
    u, n = ex["uniform"], ex["none"]
    k_ev = int(np.argmax(u["v_syn"]))
    h_tau_rest = float(mechanism.ca_lva_rates(D["rest_mV"], cfg.celsius)[3])
    gv = D["gating"]
    tau_m_rng = gv[(gv.v >= -60) & (gv.v <= -40)].m_tau
    cross40 = el[el.peak_abs_mV > -40].distance_um.min()
    plateau = el[(el.distance_um >= 130) & (el.distance_um <= 250)].peak_abs_mV
    none_l = ref["none"].summary.query("morphology == 'long'").sort_values("site_um")
    uni_300 = _value(ref["uniform"], "long", 300, "peak_soma_mV") if "uniform" in ref else np.nan
    equiv = float(np.interp(-uni_300, -none_l.peak_soma_mV.to_numpy(), none_l.site_um.to_numpy()))
    ttx = ""
    z = lambda d: _at(zl, "zin_0Hz_MOhm", d)  # noqa: E731

    fig = _svg(plotting.mechanism_figure(D))
    return f"""
<h2 id="mechanism">Mechanism: why only distal synapses trigger a Ca<sub>LVA</sub> event</h2>
<p><b>In short.</b> The dendritic Ca<sub>LVA</sub> response is a regenerative, all-or-none event. Whether a
synapse can start it depends less on how many channels sit next to it than on how <i>long</i> the synapse can
keep the surrounding membrane depolarised. Ca<sub>LVA</sub> opens slowly. Close to the soma, the soma and the
rest of the cell drain the synaptic charge within a few milliseconds, so the local EPSP is too brief to open
enough channels, however strong the synapse. Farther out, the thin dendrite is electrically more isolated,
the local EPSP lasts longer, the channels have time to open, and their inward current depolarises the
membrane further, which opens more channels. With the default synapse (Exp2Syn,
{cfg.syn_weight_uS * 1e3:g} nS, τ<sub>decay</sub> {cfg.syn_tau2_ms:g} ms) and uniform Ca<sub>LVA</sub>
({cfg.g_uniform:g} S/cm²), this transition lies near {onset[cfg.syn_tau2_ms]:.0f} µm from the soma.
The analyses behind each step are in Figure M, all with the uniform profile unless stated.</p>

<h3>1. The channel is slow, and most of it is inactivated at rest (M a, b)</h3>
<p>Ca<sub>LVA</sub> conducts in proportion to m²h. Its activation is half-maximal at −40 mV (m∞² = 0.25 there),
and its activation time constant τ<sub>m</sub> is {tau_m_rng.min():.1f}–{tau_m_rng.max():.1f} ms between −60
and −40 mV ({cfg.celsius:g} °C); it falls to about 2 ms only above −20 mV. At rest ({D['rest_mV']:.1f} mV) only
a fraction h∞ = {D['h_rest']:.2f} of the channels is available, and inactivation is itself slow
(τ<sub>h</sub> ≈ {h_tau_rest:.0f} ms near rest). To recruit Ca<sub>LVA</sub>, a depolarisation must therefore
pass about −50 to −40 mV <i>and stay there for several milliseconds</i>.</p>

<h3>2. The soma is a strong current sink, and its pull weakens with distance (M c)</h3>
<p>The input impedance at the soma is {zl.zin_soma_0Hz_MOhm.iloc[0]:.1f} MΩ. Along the target dendrite the
local input impedance rises from {z(10):.0f} MΩ at 10 µm to {z(100):.0f} MΩ at 100 µm, {z(200):.0f} MΩ at
200 µm and {z(400):.0f} MΩ at the 400 µm tip. The leak length constant of the thin grown dendrite is about
{D['lam_um']:.0f} µm, so the 400 µm dendrite is about {400 / D['lam_um']:.1f} length constants long and ends in a
sealed tip. Near the soma, synaptic current escapes quickly into the large, low-impedance soma; distally, it
charges a small, isolated piece of membrane that discharges slowly. The short dendrite has a higher impedance
at its 100 µm tip ({_at(zs, 'zin_0Hz_MOhm', 100):.0f} MΩ) than the long dendrite at the same distance, because
it ends there.</p>

<h3>3. Distally, the local EPSP is not much larger, but it lasts longer (M d, e)</h3>
<p>In the passive dendrite, the peak of the local EPSP crosses −40 mV already at about {cross40:.0f} µm, then
levels off near {plateau.mean():.0f} mV between 130 and 250 µm, because the membrane approaches the synaptic
reversal potential (0 mV) and the driving force shrinks. The peak alone therefore cannot explain why events
start only around 180–220 µm. What keeps growing is the duration: the time spent above −50 mV is
{_at(el, 't_above_ms', 100):.1f} ms at 100 µm, {_at(el, 't_above_ms', 200):.1f} ms at 200 µm and
{_at(el, 't_above_ms', 400):.1f} ms at 400 µm. Events appear where this time becomes comparable to
τ<sub>m</sub> (about 8 ms).</p>

<h3>4. A stronger synapse does not help near the soma; a longer one does (M f, g)</h3>
<p>Scanning the synaptic weight from {min(mechanism.SCAN_WEIGHTS)} to {max(mechanism.SCAN_WEIGHTS)} nS, no
synapse at {', '.join(f'{d:g}' for d in fail)} µm triggers an event, even at 8× the default strength. From
{ok.distance_um.min():g} µm onwards an event appears, with a threshold that falls from
{ok.threshold_nS.iloc[0]:g} nS to {tmin:g} nS ({', '.join(f'{d:g}' for d in ok[ok.threshold_nS == tmin].distance_um)} µm).
Larger conductances push the proximal EPSP towards the synaptic reversal potential, but that is not enough.
In contrast,
keeping the conductance at {cfg.syn_weight_uS * 1e3:g} nS but slowing its decay moves the onset towards the
soma: {', '.join(f'τ<sub>decay</sub> {t:g} ms → {onset[t]:.0f} µm' for t in sorted(onset))}. This is the causal test
that duration, not amplitude, gates the event. At the most proximal sites, even a 20 ms decay is not enough:
there, the electrotonic load of the soma dominates.</p>

<h3>5. The event is regenerative and ends by itself (M h, i)</h3>
<p>At {site:g} µm, the passive local EPSP peaks at {n['v_syn'].max():.0f} mV and decays. With Ca<sub>LVA</sub>
the membrane, after a short shoulder, depolarises again to {u['v_syn'].max():.0f} mV,
{u['t'][k_ev] - on:.0f} ms after the synaptic input: this delay reflects the slow activation. m² rises close
to 1, but h falls from {u['h'][0]:.2f} to {u['h'].min():.3f}, which ends the event. Recovery from inactivation
takes tens of milliseconds (τ<sub>h</sub> ≈ {h_tau_rest:.0f} ms at rest), so in a 50 Hz train the channels
are still inactivated when the next inputs arrive. This is why in trains only the first input triggers the
event (figure set 3).</p>

<h3>6. Where the channels are matters less than where the synapse is (Figure 1, M)</h3>
<p>With equal peak densities, the onset (on a 20 µm grid) is
{', '.join(f'{p} {pon[p]:.0f} µm' for p in pon)}. The <i>decreasing</i> profile has its highest density
({cfg.g_peak:g} S/cm²) next to the soma, yet proximal synapses still fail: more channels do not overcome the
load. With the hotspot ({lo:g}–{hi:g} µm), synapses at {', '.join(f'{d:g}' for d in hs_sites if d < lo or d > hi) or 'none'} µm,
outside the hotspot, still trigger the event, because their depolarisation spreads into the channel-rich
membrane; the event happens in the hotspot, not at the synapse.</p>

<h3>7. Consequence at the soma</h3>
<p>Without Ca<sub>LVA</sub>, a synapse at 300 µm gives a {_value(ref['none'], 'long', 300, 'peak_soma_mV'):.2f} mV
somatic EPSP. With uniform Ca<sub>LVA</sub> it gives {uni_300:.2f} mV, as much as a passive synapse at about
{equiv:.0f} µm. The dendritic event therefore compensates much of the distance-dependent attenuation for
synapses beyond the onset, and does nothing for synapses closer in. Blocking Na<sup>+</sup> channels and
somatic Ca<sub>LVA</sub> together (figure set 4) leaves these numbers almost unchanged: the event is carried by
the dendritic Ca<sub>LVA</sub>, and single inputs stay below the somatic spike threshold.</p>

<p class="meta"><b>Caveats.</b> These conclusions are for this model: the dendritic Ca<sub>LVA</sub> uses the
kinetics of the model's somatic channel, and its dendritic density is an assumption, not a measurement. An
event is counted when Ca<sub>LVA</sub> adds more than {mechanism.EVENT_CRITERION_MV:g} mV to the local EPSP.
The threshold scan uses the uniform profile and the sites listed above; onsets are given on the scan grids
(10–20 µm).{ttx}</p>
<figure>{fig}<figcaption><b>Figure M.</b> <b>a</b>, Steady-state activation (m∞²) and availability (h∞) of
Ca<sub>LVA</sub>; dotted line: resting potential. <b>b</b>, Activation and inactivation time constants.
<b>c</b>, Local input impedance (0 Hz) along the target dendrite in the passive cell; dotted line: soma.
<b>d, e</b>, Peak and time above −50 mV of the local EPSP without dendritic Ca<sub>LVA</sub>
({cfg.syn_weight_uS * 1e3:g} nS). <b>f</b>, Smallest synaptic weight that triggers an event (uniform
Ca<sub>LVA</sub>); triangles: no event up to {max(mechanism.SCAN_WEIGHTS)} nS; dotted line: default weight.
<b>g</b>, Extra local depolarisation caused by Ca<sub>LVA</sub> for synaptic decay time constants of
{', '.join(f'{t:g}' for t in sorted(D['tau']))} ms; dotted line: event criterion. <b>h</b>, Local membrane
potential at {site:g} µm with and without Ca<sub>LVA</sub>. <b>i</b>, Ca<sub>LVA</sub> gates at the same site:
fast activation (m²), slow inactivation (h), and their product, the open probability (normalised).
</figcaption></figure>"""


def methods_html(cfg: Config) -> str:
    meta = load_target_meta()
    s, l = meta["short"], meta["long"]
    g = l["growth"]
    return f"""
<p><b>Model.</b> Human L2/3 PV+ interneuron (HL23PV) of Yao <i>et al.</i> (2022), <i>Cell Reports</i> 38, 110232
(doi:<a href="https://doi.org/10.1016/j.celrep.2021.110232">10.1016/j.celrep.2021.110232</a>), with the morphology,
ion channels and parameters of the Zenodo release
(doi:<a href="https://doi.org/10.5281/zenodo.5771000">10.5281/zenodo.5771000</a>), translated to Python.
Dendrites are passive with I<sub>h</sub>; soma and axon carry the original active conductances.</p>
<p><b>Morphologies.</b> <i>Short</i>: the original reconstruction; the target dendrite
({' → '.join(s['path_sections'])}) ends {s['tip_distance_um']:.1f} µm from the soma (path distance from the
soma centre). <i>Long</i>: the same cell with that tip grown by {g['added_um']:.1f} µm (random walk, {g['step_um']:g} µm
steps, direction noise s.d. {g['jitter_sd']:g}, seed {g['seed']}, constant diameter {2 * g['radius_um']:.2f} µm), so
that the tip lies {l['tip_distance_um']:.1f} µm from the soma. Everything else is identical.</p>
<p><b>Dendritic Ca<sub>LVA</sub>.</b> Only on the target path of the long cell (0–400 µm), with CaDynamics
(γ {cfg.cadyn_gamma:g}, τ {cfg.cadyn_decay_ms:.0f} ms). Profiles: none; uniform; hotspot
({cfg.hotspot_width_um:g} µm wide, centred at {cfg.hotspot_center_um:g} µm); increasing and decreasing linear
gradients over 0–{cfg.gradient_span_um:g} µm. The short dendrite never carries Ca channels.</p>
<p><b>Input and simulation.</b> One Exp2Syn conductance synapse at a time, placed at the same absolute
distances in both cells. {cfg.celsius:g} °C, dt {cfg.dt_ms:g} ms, segments ≤ {cfg.max_seg_len_um:g} µm along the
target dendrite, each run started from the steady resting state.</p>"""


def build(set_file=None, out=None, progress=None) -> Path:
    set_file = Path(set_file or CONFIG_DIR / "figure_set.json")
    out = Path(out or FIGURES_DIR / "report.html")
    entries = json.loads(set_file.read_text())["figures"]
    base = Config.from_json(CONFIG_DIR / entries[0]["base"])

    toc, body = [], []
    cells = {m: get_cell(m, base) for m in ("short", "long")}
    body.append("<h2 id='morphology'>Morphologies</h2>")
    body.append(f"<figure>{_svg(plotting.morphology_figure(cells))}<figcaption>Short (left) and long (right) "
                "morphologies in the same orientation. The target dendrite is highlighted; circles every "
                "50 µm.</figcaption></figure>")
    toc.append("<li><a href='#morphology'>Morphologies</a></li>")

    for k, entry in enumerate(entries, 1):
        cfg0 = Config.from_json(CONFIG_DIR / entry["base"]).replace(**entry.get("overrides", {}))
        results = {}
        for prof in entry["profiles"]:
            cfg = cfg0.replace(name=f"{entry['name']}__{prof}", ca_profile=prof)
            if progress:
                progress(cfg.name)
            results[prof] = load_or_run(cfg)
        anchor = entry["name"]
        toc.append(f"<li><a href='#{anchor}'>{html.escape(anchor)}</a></li>")
        body.append(f"<h2 id='{anchor}'>{k}. {html.escape(anchor)}</h2>")
        body.append(f"<p class='meta'>{describe_html(cfg0)}</p>")
        body.append(set_findings(results, entry.get("note", "")))
        body.append(summary_table(results))
        body.append(f"<figure>{_svg(plotting.comparison_figure(results))}"
                    f"<figcaption><b>Figure {k}.</b> {CAPTION_COMPARISON}</figcaption></figure>")
        for prof, res in results.items():
            fig = plotting.overview_figure(res, {m: get_cell(m, res.config) for m in res.config.morphologies})
            body.append(f"<details><summary>Detail: {prof}</summary><figure>{_svg(fig)}"
                        f"<figcaption><b>Figure {k}–{prof}.</b> {CAPTION_OVERVIEW}</figcaption></figure></details>")
        body.append(f"<p class='meta'>Reproduce: <code>python scripts/make_all_figures.py --only {anchor}</code></p>")

    if progress:
        progress("mechanism analyses")
    D = mechanism.collect(base)
    first = json.loads(set_file.read_text())["figures"][0]
    ref = {p: load_or_run(Config.from_json(CONFIG_DIR / first["base"]).replace(
        **first.get("overrides", {}), name=f"{first['name']}__{p}", ca_profile=p)) for p in first["profiles"]}
    toc.insert(1, "<li><a href='#mechanism'>Mechanism: why only distal synapses trigger a Ca event</a></li>")
    body.insert(2, mechanism_html(D, ref))
    plotting.save_figure(plotting.mechanism_figure(D), "mechanism/mechanism", formats=("pdf", "png"))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PV dendrite report</title><style>{CSS}</style></head>
<body><main>
<h1>Short (100 µm) vs grown (400 µm) dendrite of a human PV+ interneuron</h1>
<p class="meta">Generated {dt.datetime.now():%Y-%m-%d %H:%M} · pvdend {__version__} · NEURON {neuron.__version__}
· figure set <code>{set_file.name}</code></p>
<nav><ol>{''.join(toc)}</ol></nav>
<h2 id="methods">Methods in brief</h2>{methods_html(base)}
{''.join(body)}
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
