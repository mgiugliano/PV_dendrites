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

from . import __version__, plotting
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
    "<b>f</b>, Area of the somatic depolarisation above rest.")
CAPTION_OVERVIEW = (
    "<b>a, b</b>, Short and long morphologies; the target dendrite is highlighted (in <b>b</b> coloured by "
    "Ca<sub>LVA</sub> density) and circles mark the synapse sites whose traces are shown. "
    "<b>c</b>, Ca<sub>LVA</sub> density along the long dendrite. <b>d, e</b>, Somatic membrane potential for "
    "synapses at increasing distance (colour scale). <b>f</b>, Membrane potential at the synapse, long "
    "morphology. <b>g–i</b>, Somatic EPSP, local EPSP and attenuation versus synapse distance.")

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


def summary_table(results: dict) -> str:
    head = "".join(f"<th>{s} µm</th>" for s in TABLE_SITES)
    rows = []
    first = next(iter(results.values()))
    if "short" in first.config.morphologies:
        cells = "".join(f"<td>{_fmt(_value(first, 'short', s, 'peak_soma_mV'))}</td>" for s in TABLE_SITES)
        n_spk = int(first.summary[first.summary.morphology == "short"].n_spikes_soma.sum())
        rows.append(f"<tr><td><span class='swatch' style='background:{plotting.INK_2}'></span>"
                    f"short (no dendritic Ca)</td>{cells}<td>–</td>"
                    f"<td>{n_spk}</td></tr>")
    for prof, res in results.items():
        long = res.summary[res.summary.morphology == "long"]
        cells = "".join(f"<td>{_fmt(_value(res, 'long', s, 'peak_soma_mV'))}</td>" for s in TABLE_SITES)
        rows.append(f"<tr><td><span class='swatch' style='background:{plotting.PROFILE_COLORS[prof]}'></span>"
                    f"long, {prof}</td>{cells}<td>{_fmt(long.dcai_syn_uM.max(), 3)}</td>"
                    f"<td>{int(long.n_spikes_soma.sum())}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th rowspan='2'>Morphology, Ca profile</th>"
            f"<th colspan='{len(TABLE_SITES)}'>Somatic EPSP (mV) for a synapse at</th>"
            "<th rowspan='2'>max local Δ[Ca<sup>2+</sup>]<sub>i</sub> (µM)</th>"
            "<th rowspan='2'>somatic spikes (all sites)</th></tr>"
            f"<tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


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
        body.append(summary_table(results))
        body.append(f"<figure>{_svg(plotting.comparison_figure(results))}"
                    f"<figcaption><b>Figure {k}.</b> {CAPTION_COMPARISON}</figcaption></figure>")
        for prof, res in results.items():
            fig = plotting.overview_figure(res, {m: get_cell(m, res.config) for m in res.config.morphologies})
            body.append(f"<details><summary>Detail: {prof}</summary><figure>{_svg(fig)}"
                        f"<figcaption><b>Figure {k}–{prof}.</b> {CAPTION_OVERVIEW}</figcaption></figure></details>")
        body.append(f"<p class='meta'>Reproduce: <code>python scripts/make_all_figures.py --only {anchor}</code></p>")

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
