"""Draft manuscript text (Methods, Results, figure legends, references) for the report.

Every number is computed from the simulation results, so the text stays consistent
with the figures whenever parameters change and the report is rebuilt. The prose is a
draft for the authors to edit, not final text.
"""
from __future__ import annotations

import platform
import re

import matplotlib
import neuron
import numpy as np
import pandas
import plotly

from . import mechanism
from .cell import ALL, AXONAL, SOMATIC
from .config import Config
from .morphology import load_target_meta, path_geometry, side_branches

REPO_URL = "https://github.com/mgiugliano/PV_dendrites"
COLAB_URL = ("https://colab.research.google.com/github/mgiugliano/PV_dendrites/blob/main/"
             "notebooks/explore.ipynb")

# Numbered references, cited as [n] in the text below. DOIs checked against Crossref
# (and Zenodo for [2]).
REFERENCES = [
    ("yao2022", "Yao HK, Guet-McCreight A, Mazza F, Moradi Chameh H, Prevot TD, Griffiths JD, Tripathy SJ, "
                "Valiante TA, Sibille E, Hay E (2022). Reduced inhibition in depression impairs stimulus "
                "processing in human cortical microcircuits. <i>Cell Reports</i> 38(2):110232.",
     "10.1016/j.celrep.2021.110232"),
    ("zenodo", "Yao HK, Hay E (2021). Human cortical layer 2/3 microcircuits in health and depression "
               "[code and data]. Zenodo.", "10.5281/zenodo.5771000"),
    ("neuron97", "Hines ML, Carnevale NT (1997). The NEURON simulation environment. <i>Neural Computation</i> "
                 "9(6):1179–1209.", "10.1162/neco.1997.9.6.1179"),
    ("nrnbook", "Carnevale NT, Hines ML (2006). <i>The NEURON Book</i>. Cambridge University Press.",
     "10.1017/CBO9780511541612"),
    ("nrnpy", "Hines ML, Davison AP, Muller E (2009). NEURON and Python. <i>Frontiers in Neuroinformatics</i> "
              "3:1.", "10.3389/neuro.11.001.2009"),
    ("bluepyopt", "Van Geit W, Gevaert M, Chindemi G, Rössert C, Courcol JD, Muller EB, Schürmann F, Segev I, "
                  "Markram H (2016). BluePyOpt: leveraging open source software and cloud infrastructure to "
                  "optimise model parameters in neuroscience. <i>Frontiers in Neuroinformatics</i> 10:17.",
     "10.3389/fninf.2016.00017"),
    ("hay2011", "Hay E, Hill S, Schürmann F, Markram H, Segev I (2011). Models of neocortical layer 5b "
                "pyramidal cells capturing a wide range of dendritic and perisomatic active properties. "
                "<i>PLoS Computational Biology</i> 7(7):e1002107.", "10.1371/journal.pcbi.1002107"),
    ("avery1996", "Avery RB, Johnston D (1996). Multiple channel types contribute to the low-voltage-activated "
                  "calcium current in hippocampal CA3 pyramidal neurons. <i>Journal of Neuroscience</i> "
                  "16(18):5567–5582.", "10.1523/JNEUROSCI.16-18-05567.1996"),
    ("randall1997", "Randall AD, Tsien RW (1997). Contrasting biophysical and pharmacological properties of "
                    "T-type and R-type calcium channels. <i>Neuropharmacology</i> 36(7):879–893.",
     "10.1016/S0028-3908(97)00086-5"),
    ("destexhe1994", "Destexhe A, Mainen ZF, Sejnowski TJ (1994). Synthesis of models for excitable membranes, "
                     "synaptic transmission and neuromodulation using a common kinetic formalism. <i>Journal "
                     "of Computational Neuroscience</i> 1(3):195–230.", "10.1007/BF00961734"),
    ("kole2006", "Kole MHP, Hallermann S, Stuart GJ (2006). Single I<sub>h</sub> channels in pyramidal neuron "
                 "dendrites: properties, distribution, and impact on action potential output. <i>Journal of "
                 "Neuroscience</i> 26(6):1677–1687.", "10.1523/JNEUROSCI.3664-05.2006"),
    ("rall1959", "Rall W (1959). Branching dendritic trees and motoneuron membrane resistivity. "
                 "<i>Experimental Neurology</i> 1(5):491–527.", "10.1016/0014-4886(59)90046-9"),
    ("hu2010", "Hu H, Martina M, Jonas P (2010). Dendritic mechanisms underlying rapid synaptic activation of "
               "fast-spiking hippocampal interneurons. <i>Science</i> 327(5961):52–58.",
     "10.1126/science.1177876"),
    ("norenberg2010", "Nörenberg A, Hu H, Vida I, Bartos M, Jonas P (2010). Distinct nonuniform cable "
                      "properties optimize rapid and efficient activation of fast-spiking GABAergic "
                      "interneurons. <i>Proceedings of the National Academy of Sciences USA</i> 107(2):894–899.",
     "10.1073/pnas.0910716107"),
    ("numpy", "Harris CR, Millman KJ, van der Walt SJ, et al. (2020). Array programming with NumPy. "
              "<i>Nature</i> 585(7825):357–362.", "10.1038/s41586-020-2649-2"),
    ("matplotlib", "Hunter JD (2007). Matplotlib: a 2D graphics environment. <i>Computing in Science &amp; "
                   "Engineering</i> 9(3):90–95.", "10.1109/MCSE.2007.55"),
    ("pandas", "McKinney W (2010). Data structures for statistical computing in Python. <i>Proceedings of the "
               "9th Python in Science Conference</i>, 56–61.", "10.25080/Majora-92bf1922-00a"),
]
_IDX = {key: i + 1 for i, (key, _, _) in enumerate(REFERENCES)}


def cite(*keys) -> str:
    return "[" + ",".join(str(i) for i in sorted(_IDX[k] for k in keys)) + "]"


def references_html() -> str:
    items = "".join(f"<li id='ref-{k}'>{text} doi:<a href='https://doi.org/{doi}'>{doi}</a></li>"
                    for k, text, doi in REFERENCES)
    return f"<h2 id='references'>References</h2><ol class='refs'>{items}</ol>"


# --- helpers ------------------------------------------------------------------------------------

def _v(res, morph, site, col):
    df = res.summary
    row = df[(df.morphology == morph) & np.isclose(df.site_um, site)]
    return float(row[col].iloc[0]) if len(row) else np.nan


def _events(results, prof, crit=mechanism.EVENT_CRITERION_MV):
    a = results[prof].summary.query("morphology == 'long'").set_index("site_um").peak_syn_mV
    b = results["none"].summary.query("morphology == 'long'").set_index("site_um").peak_syn_mV
    return [float(d) for d in (a - b)[(a - b) > crit].index]


def _rng(sites):
    return f"{min(sites):g}–{max(sites):g} µm" if sites else "no site"


def _soma_range(res, sites):
    long = res.summary[(res.summary.morphology == "long") & res.summary.site_um.isin(sites)]
    return long.peak_soma_mV.min(), long.peak_soma_mV.max()


def _event_peaks(res, site, where="v_syn"):
    """Peak local depolarisation within each inter-event interval of a train."""
    c = res.config
    tr = res.traces[("long", site)]
    t, v = tr["t"], tr[where]
    base = v[t < c.onset_ms].mean()
    return [float(v[(t >= c.onset_ms + i * c.interval_ms) & (t < c.onset_ms + (i + 1) * c.interval_ms)].max() - base)
            for i in range(c.n_events)]


def _n_sites(res, morph):
    return int((res.summary.morphology == morph).sum())


# --- Methods ------------------------------------------------------------------------------------

def methods_html(cells: dict, cfg: Config, rest_mV: float) -> str:
    meta = load_target_meta()
    s_meta, l_meta = meta["short"], meta["long"]
    g = l_meta["growth"]
    short, long = cells["short"], cells["long"]
    geo = path_geometry(long, reference=short)
    n_prim = sum(1 for s in short.dend if s.parentseg() is not None and s.parentseg().sec == short.soma[0])
    soma = short.soma[0]
    q10 = 2.3 ** ((cfg.celsius - 21) / 10)
    return f"""
<h3>Neuron model</h3>
<p>We simulated the human layer 2/3 parvalbumin-positive (PV+) interneuron model (HL23PV) of Yao <i>et al.</i>
{cite('yao2022')}, whose passive and active parameters had been optimised with BluePyOpt {cite('bluepyopt')}
against electrophysiological features of human cortical PV+ interneurons, and which we obtained from the authors'
public release {cite('zenodo')}. We translated the original HOC template and biophysics files into Python, using
the NEURON simulator {cite('neuron97', 'nrnbook')} through its Python interface {cite('nrnpy')}, without changing
any parameter; an automated test compares every value of the translated model with the original
<code>biophys_HL23PV.hoc</code>. All compartments have a specific membrane capacitance of {ALL['cm']:g} µF/cm²,
an axial resistivity of {ALL['Ra']:g} Ω·cm, a leak conductance of {ALL['g_pas']:.3e} S/cm² reversing at
{ALL['e_pas']:.2f} mV, and a hyperpolarisation-activated cation conductance (I<sub>h</sub>
{cite('kole2006')}; {ALL['gbar_Ih']:.3e} S/cm²). In the original model the dendrites carry no other conductance,
whereas the soma and the axon carry transient (NaTg) and persistent (Nap) Na<sup>+</sup> conductances, four
K<sup>+</sup> conductances (K_P, K_T, Kv3.1, I<sub>M</sub>), a Ca<sup>2+</sup>-activated K<sup>+</sup>
conductance (SK), high- and low-voltage-activated Ca<sup>2+</sup> conductances (Ca_HVA, Ca<sub>LVA</sub>) and
intracellular Ca<sup>2+</sup> dynamics, with E<sub>Na</sub> = {SOMATIC['ena']:g} mV and E<sub>K</sub> =
{SOMATIC['ek']:g} mV; for instance, the somatic densities of NaTg and Ca<sub>LVA</sub> are
{SOMATIC['gbar_NaTg']:.3f} and {SOMATIC['gbar_Ca_LVA']:.4f} S/cm², respectively. Following the BluePyOpt
convention used by the original model, we replaced the reconstructed axon with two 30 µm cylindrical sections
whose diameters ({long.axon[0].diam:.2f} and {long.axon[1].diam:.2f} µm) we took from the reconstruction. We
set the number of segments of every section to 1 + 2·int(L/40 µm), as in the original code, except along the
target dendrite (see below), and we ran all simulations at {cfg.celsius:g} °C.</p>

<h3>Morphologies and growth of the target dendrite</h3>
<p>The reconstruction ({s_meta['file']}, distributed with {cite('zenodo')}) has a soma of {soma.diam:.1f} µm
diameter, {n_prim} primary dendrites and {len(list(short.basal))} basal sections. We measured all distances as
path distances from the centre of the soma. As the <i>target dendrite</i> we chose the terminal path whose tip
lies closest to 100 µm from the soma, which runs through {', '.join(s_meta['path_sections'])} and ends at
{s_meta['tip_distance_um']:.1f} µm (<i>short</i> morphology). The path itself has no branch point other than the
two at which side branches leave it ({', '.join(b.name().split('.')[-1] for b in side_branches(short, short.target_path))});
its diameter tapers from {geo[0]['diam_max_um']:.2f} µm at the soma to
{geo[2]['diam_min_um']:.2f}–{geo[2]['diam_max_um']:.2f} µm beyond {geo[1]['end_um']:.0f} µm. To build the
<i>long</i> morphology, we extended the tip of this dendrite (SWC node {g['from_swc_id']}) by
{g['added_um']:.2f} µm, so that the new tip lies {l_meta['tip_distance_um']:.1f} µm from the soma, with the
random-walk procedure used in our earlier simulations. Starting from the direction of the last original segment
<b>u</b><sub>0</sub>, we appended points every Δ = {g['step_um']:g} µm along <b>u</b><sub>k+1</sub> =
(<b>u</b><sub>k</sub> + <b>ξ</b><sub>k</sub>)/‖<b>u</b><sub>k</sub> + <b>ξ</b><sub>k</sub>‖, with
<b>ξ</b><sub>k</sub> drawn from a three-dimensional Gaussian distribution of zero mean and standard deviation
{g['jitter_sd']:g} per component (seed {g['seed']}), keeping the radius equal to that of the original tip
({g['radius_um']:.4f} µm). The two morphologies are therefore identical except for the
{g['added_um']:.1f} µm unbranched, constant-diameter ({2 * g['radius_um']:.2f} µm) extension, which adds
{sum(r['area_um2'] for r in geo) - sum(r['area_um2'] for r in path_geometry(short)):.0f} µm² of membrane
(Supplementary Fig. S1). As a control, the code can also remove the side branches of the target dendrite in
both morphologies; all results shown here keep them.</p>

<h3>Spatial discretisation</h3>
<p>To resolve synaptic sites spaced by 10 µm and the spatial profiles of channel density, we divided every
section of the target dendrite into an odd number of segments no longer than {cfg.max_seg_len_um:g} µm
({sum(s.nseg for s in short.target_path)} segments in the short and {sum(s.nseg for s in long.target_path)} in
the long morphology), in both morphologies alike.</p>

<h3>Dendritic Ca<sub>LVA</sub> conductance</h3>
<p>Since the original model has passive dendrites, we added a low-voltage-activated Ca<sup>2+</sup> conductance
to the target dendrite of the long morphology only, while the short morphology kept its original, Ca-free
dendrite. We used the Ca<sub>LVA</sub> model already present in the soma and axon of HL23PV, which originates
from Hay <i>et al.</i> {cite('hay2011')} and is based on the data of Avery and Johnston {cite('avery1996')} and
Randall and Tsien {cite('randall1997')}. The current is I<sub>Ca</sub> = ḡ m² h (V − E<sub>Ca</sub>), and
both gates obey first-order kinetics, dx/dt = (x<sub>∞</sub> − x)/τ<sub>x</sub>, with (V in mV, shifted by
+10 mV to account for the liquid junction potential, V′ = V + 10)</p>
<p class="eq">m<sub>∞</sub> = 1 / (1 + exp(−(V′ + 30)/6)),&nbsp;&nbsp;
τ<sub>m</sub> = [5 + 20 / (1 + exp((V′ + 25)/5))] / q ms,<br>
h<sub>∞</sub> = 1 / (1 + exp((V′ + 80)/6.4)),&nbsp;&nbsp;
τ<sub>h</sub> = [20 + 50 / (1 + exp((V′ + 40)/7))] / q ms,</p>
<p>where q = 2.3<sup>(T − 21 °C)/10</sup> = {q10:.2f} at {cfg.celsius:g} °C. Wherever we inserted
Ca<sub>LVA</sub>, we also inserted the intracellular Ca<sup>2+</sup> dynamics of the original model
{cite('destexhe1994', 'hay2011')}, a submembrane shell of depth d = 0.1 µm in which
d[Ca<sup>2+</sup>]<sub>i</sub>/dt = −10<sup>4</sup> γ I<sub>Ca</sub> / (2 F d) − ([Ca<sup>2+</sup>]<sub>i</sub> −
[Ca<sup>2+</sup>]<sub>min</sub>)/τ<sub>Ca</sub>, with γ = {cfg.cadyn_gamma:g}, τ<sub>Ca</sub> =
{cfg.cadyn_decay_ms:.0f} ms (the somatic values) and [Ca<sup>2+</sup>]<sub>min</sub> = 100 nM. E<sub>Ca</sub>
followed the Nernst equation with [Ca<sup>2+</sup>]<sub>o</sub> = 2 mM (about 131 mV at rest).</p>
<p>We defined the channel density ḡ(d) as a function of the path distance d of each segment from the soma,
over the whole target dendrite (0–{l_meta['tip_distance_um']:.0f} µm), for five profiles: <i>none</i>
(ḡ = 0); <i>uniform</i> (ḡ = g); <i>hotspot</i> (ḡ = g for |d − {cfg.hotspot_center_um:g} µm| ≤
{cfg.hotspot_width_um / 2:g} µm and 0 elsewhere); <i>increasing</i> (ḡ = g·min(d/L, 1)); and
<i>decreasing</i> (ḡ = g·max(1 − d/L, 0)), with L = {cfg.gradient_span_um:g} µm. We set g in two ways. With
<i>peak normalisation</i>, g = {cfg.g_uniform:g} S/cm² for the uniform profile and g = {cfg.g_peak:g} S/cm² for the
other profiles, as in our earlier simulations. With <i>total normalisation</i>, we scaled each profile so that
the total conductance on the dendrite, Σ<sub>i</sub> ḡ(d<sub>i</sub>) A<sub>i</sub> over all segments i of
membrane area A<sub>i</sub>, equals that of a uniform density of {cfg.g_total_equiv:g} S/cm², so that profiles
differ only in where the channels are. We note that these dendritic densities are model assumptions, not
measurements.</p>

<h3>Pharmacological manipulations</h3>
<p>We mimicked TTX by setting the densities of NaTg and Nap to zero in all compartments, and we removed somatic
Ca<sub>LVA</sub> by setting its somatic density to zero; both switches act identically in the two
morphologies.</p>

<h3>Synaptic input</h3>
<p>We modelled an excitatory synapse as a conductance with a double-exponential time course (NEURON
<code>Exp2Syn</code>), g<sub>syn</sub>(t) = w·f·[exp(−t/τ<sub>2</sub>) − exp(−t/τ<sub>1</sub>)], normalised
(through f) to a peak of w, with τ<sub>1</sub> = {cfg.syn_tau1_ms:g} ms, τ<sub>2</sub> = {cfg.syn_tau2_ms:g} ms,
reversal potential {cfg.syn_e_mV:g} mV and w = {cfg.syn_weight_uS * 1e3:g} nS, unless stated otherwise. We
activated it either once, {cfg.onset_ms:g} ms after the start of the simulation, or with a regular train of
5 events at 50 Hz. In each simulation we placed a single synapse on the target dendrite, at path distances from
{cfg.site_start_um:g} µm to the tip in steps of {cfg.site_step_um:g} µm, the same in both morphologies (that is,
10–100 µm in the short and 10–400 µm in the long morphology); each synapse sits at the centre of the
segment containing that distance, within {cfg.max_seg_len_um / 2:g} µm of it.</p>

<h3>Simulations</h3>
<p>We integrated the equations with NEURON's fixed-step backward Euler method (Δt = {cfg.dt_ms * 1e3:g} µs).
Before each simulation, we initialised the membrane potential at {cfg.v_init_mV:g} mV and brought the cell to
its steady state by a few implicit integration steps of very large size at negative times, so that every
simulation started from rest ({rest_mV:.1f} mV) with flat
baselines (fluctuations below 10<sup>−10</sup> mV). We then simulated {cfg.t_post_ms:g} ms beyond the last
synaptic event.</p>

<h3>Measurements</h3>
<p>We recorded the membrane potential at the soma, at the synapse and at the tip of the target dendrite, and
[Ca<sup>2+</sup>]<sub>i</sub> and I<sub>Ca</sub> at the synapse. We defined the somatic (local) EPSP amplitude as
the peak depolarisation after the input relative to the mean potential in the 5 ms before it, the attenuation
as the ratio between somatic and local amplitudes, the EPSP area as the time integral of the somatic
depolarisation, the 10–90% rise time and the half-width of single-event somatic EPSPs, and the local
Δ[Ca<sup>2+</sup>]<sub>i</sub> as the peak increase of [Ca<sup>2+</sup>]<sub>i</sub>. We counted a dendritic
<i>Ca<sub>LVA</sub> event</i> whenever Ca<sub>LVA</sub> added more than {mechanism.EVENT_CRITERION_MV:g} mV to the
local EPSP peak, compared with the same synapse in the same morphology without dendritic Ca<sub>LVA</sub>, and we
defined the onset distance as the most proximal site with an event.</p>

<h3>Mechanistic analyses</h3>
<p>We computed the local input impedance along the target dendrite, and the transfer impedance to the soma, at
0 and 100 Hz with NEURON's <code>Impedance</code> class, linearising the active conductances around rest
(extended mode), in both morphologies without dendritic Ca<sub>LVA</sub>. We estimated the leak length constant
of the grown dendrite as λ = (R<sub>m</sub> d / 4 R<sub>a</sub>)<sup>1/2</sup> {cite('rall1959')}, with
R<sub>m</sub> = 1/g<sub>pas</sub>. We characterised the passive local EPSP by its peak and by the time it spends
above −50 mV. To find the smallest synaptic weight that triggers an event, we scanned w over
{', '.join(f'{x:g}' for x in mechanism.SCAN_WEIGHTS)} nS at {', '.join(f'{x:g}' for x in mechanism.SCAN_SITES)}
µm, with uniform Ca<sub>LVA</sub>; to test the role of the EPSP duration, we varied τ<sub>2</sub> over
{', '.join(f'{x:g}' for x in mechanism.TAU2_VALUES)} ms at fixed w, at sites every 20 µm. For the gating
analysis, we recorded m and h of Ca<sub>LVA</sub> at a synapse {mechanism.EXAMPLE_SITE:g} µm from the soma.</p>

<h3>Software, code and data availability</h3>
<p>All code, morphologies, configuration files and figures are available at
<a href="{REPO_URL}">{REPO_URL}</a> under a BSD 3-Clause licence (the files of the original model keep the terms
of their release {cite('zenodo')}). The repository contains the Python package that implements the model and the
protocols, the scripts that regenerate every figure and this report from the configuration files
(<code>scripts/make_all_figures.py</code>, <code>scripts/make_report.py</code>), automated tests, and an
interactive notebook that runs in the browser on Google Colab (<a href="{COLAB_URL}">link</a>). We used
Python {platform.python_version()}, NEURON {neuron.__version__}, NumPy {np.__version__} {cite('numpy')},
pandas {pandas.__version__} {cite('pandas')}, Matplotlib {matplotlib.__version__} {cite('matplotlib')} and
Plotly {plotly.__version__}.</p>"""


# --- Results ------------------------------------------------------------------------------------

def results_html(R: dict, D: dict, cells: dict, cfg: Config) -> str:
    sp, st, tr, tx = R["single_peak"], R["single_total"], R.get("train5x50Hz_peak"), R.get("single_peak_TTX_noSomaCa")
    n = sp["none"]
    short, long = cells["short"], cells["long"]
    zl, zs = D["impedance"]["long"], D["impedance"]["short"]
    z = lambda df, d: float(df.iloc[int(np.argmin(np.abs(df.distance_um - d)))]["zin_0Hz_MOhm"])  # noqa: E731
    el = D["epsp"]["long"]
    t_above = lambda d: float(el.iloc[int(np.argmin(np.abs(el.distance_um - d)))].t_above_ms)  # noqa: E731
    ev = {p: _events(sp, p) for p in ("uniform", "hotspot", "increasing", "decreasing")}
    ev_t = {p: _events(st, p) for p in ("uniform", "hotspot", "increasing", "decreasing")}
    u_lo, u_hi = _soma_range(sp["uniform"], ev["uniform"])
    none_l = n.summary.query("morphology == 'long'").sort_values("site_um")
    u300 = _v(sp["uniform"], "long", 300, "peak_soma_mV")
    equiv = float(np.interp(-u300, -none_l.peak_soma_mV.to_numpy(), none_l.site_um.to_numpy()))
    th = D["thresholds"]
    fail = th[th.threshold_nS.isna()].distance_um
    ok = th[th.threshold_nS.notna()]
    onset_tau = {t: mechanism.onset_um(df) for t, df in D["tau"].items()}
    u = D["example"]["uniform"]
    k = int(np.argmax(u["v_syn"]))
    dec_peak_near = cfg.g_peak
    geo = path_geometry(long, reference=short)
    lo_h, hi_h = cfg.hotspot_center_um - cfg.hotspot_width_um / 2, cfg.hotspot_center_um + cfg.hotspot_width_um / 2
    outside = max([lo_h - d for d in ev["hotspot"] if d < lo_h] + [d - hi_h for d in ev["hotspot"] if d > hi_h] + [0])
    ul = sp["uniform"].summary
    ul = ul[(ul.morphology == "long") & ul.site_um.isin(ev["uniform"])]
    loc_abs = (ul.peak_syn_mV + ul.vrest_syn_mV)
    spikes = int(sum(r.summary.n_spikes_soma.sum() for r in sp.values()))
    gv = D["gating"]
    tm = gv[(gv.v >= -60) & (gv.v <= -40)].m_tau
    out = f"""
<h3>Growing the dendrite increases the attenuation of its synaptic inputs</h3>
<p>The dendrites of fast-spiking PV+ interneurons are thin, and their cable properties shape how quickly and how
efficiently excitatory inputs reach the soma {cite('hu2010', 'norenberg2010')}. To ask how the length of a single
dendrite affects this, we compared the original reconstruction of a human L2/3 PV+ interneuron model
{cite('yao2022')}, in which the chosen dendrite ends {short.tip_distance:.1f} µm from the soma, with a copy in
which we grew the same dendrite to {long.tip_distance:.0f} µm, leaving every other branch untouched (Fig. 1;
Supplementary Fig. S1). Beyond its first {geo[1]['end_um']:.0f} µm, the target dendrite is a thin cable of nearly
constant diameter ({geo[2]['diam_min_um']:.2f}–{geo[2]['diam_max_um']:.2f} µm), and its local input impedance rises
steeply with distance, from {z(zl, 10):.0f} MΩ at 10 µm to {z(zl, 100):.0f} MΩ at 100 µm and
{z(zl, 400):.0f} MΩ at the grown tip, against {zl.zin_soma_0Hz_MOhm.iloc[0]:.1f} MΩ at the soma (Fig. 4c).
With passive dendrites, a {cfg.syn_weight_uS * 1e3:g} nS synapse 10 µm from the soma produced somatic EPSPs of
{_v(n, 'short', 10, 'peak_soma_mV'):.2f} and {_v(n, 'long', 10, 'peak_soma_mV'):.2f} mV in the short and long
morphologies, and the amplitude fell steeply with distance (Fig. 2g). Already at 100 µm, the somatic EPSP was
smaller in the long ({_v(n, 'long', 100, 'peak_soma_mV'):.2f} mV) than in the short morphology
({_v(n, 'short', 100, 'peak_soma_mV'):.2f} mV), where the same site is the sealed tip of the dendrite and
has a higher local input impedance ({z(zs, 100):.0f} against {z(zl, 100):.0f} MΩ). At the grown tip, the somatic EPSP was only
{_v(n, 'long', 400, 'peak_soma_mV'):.2f} mV, {100 * _v(n, 'long', 400, 'attenuation'):.1f}% of the local
depolarisation ({_v(n, 'long', 400, 'peak_syn_mV'):.0f} mV). Taken together, these results indicate that
lengthening the dendrite both adds a distal stretch of strongly attenuated inputs and slightly weakens inputs
on the part that the two morphologies share.</p>

<h3>Dendritic Ca<sub>LVA</sub> produces regenerative events at distal synapses only</h3>
<p>We then inserted Ca<sub>LVA</sub>, with the kinetics of the model's somatic channel, along the dendrite of the
long morphology. With a uniform density of {cfg.g_uniform:g} S/cm², synapses up to about
{min(ev['uniform']) - cfg.site_step_um:.0f} µm behaved as in the passive dendrite, whereas every synapse from
{_rng(ev['uniform'])} triggered a regenerative depolarisation: a second, slower component that followed the
synaptic EPSP by several milliseconds and brought the local membrane potential to {loc_abs.min():.0f} to
{loc_abs.max():.0f} mV (Fig. 2f, Fig. 4h). These
events increased the somatic EPSP to {u_lo:.1f}–{u_hi:.1f} mV; a synapse at 300 µm, for instance, produced
{u300:.2f} mV instead of {_v(n, 'long', 300, 'peak_soma_mV'):.2f} mV, as much as a passive synapse
{equiv:.0f} µm from the soma (Fig. 2g). All other profiles gave the same all-or-none behaviour, with events
for synapses at {_rng(ev['hotspot'])} (hotspot), {_rng(ev['increasing'])} (increasing) and
{_rng(ev['decreasing'])} (decreasing gradient) (Fig. 3). Interestingly, synapses up to {outside:.0f} µm outside the
hotspot ({lo_h:g}–{hi_h:g} µm) also triggered events, and the decreasing gradient, which has its highest density
({dec_peak_near:g} S/cm²) next to the soma, did not produce any event proximally. {('Single synapses never evoked somatic action potentials.' if spikes == 0 else f'Single synapses evoked {spikes} somatic action potentials in total.')} These observations suggest that where the synapse sits, rather than where the
channels sit, decides whether an event occurs.</p>

<h3>The duration of the local depolarisation, not its amplitude, gates the event</h3>
<p>To understand why events appear only beyond about 200 µm, we examined the passive local EPSP and the
Ca<sub>LVA</sub> kinetics (Fig. 4). At rest ({D['rest_mV']:.1f} mV) only {100 * D['h_rest']:.0f}% of the channels
are available, the activation is half-maximal at −40 mV, and its time constant is {tm.min():.1f}–{tm.max():.1f} ms between
−60 and −40 mV
(Fig. 4a,b). The peak of the passive local EPSP crossed −40 mV already about 60 µm from the soma, but it then
levelled off near −30 mV between 130 and 250 µm, as the membrane approached the synaptic reversal potential
(Fig. 4d), so that the peak alone cannot explain the onset. In contrast, the time the local EPSP spent above
−50 mV kept increasing with distance, from {t_above(100):.1f} ms at 100 µm to {t_above(200):.1f} ms at 200 µm and
{t_above(400):.1f} ms at the tip, reaching the activation time constant of Ca<sub>LVA</sub> where events began
(Fig. 4e). Two manipulations supported this interpretation. First, no synapse at
{', '.join(f'{d:g}' for d in fail)} µm triggered an event, even when we increased its weight to
{max(mechanism.SCAN_WEIGHTS)} nS, whereas from {ok.distance_um.min():g} µm onwards the threshold fell from
{ok.threshold_nS.iloc[0]:g} to {ok.threshold_nS.min():g} nS (Fig. 4f). Second, keeping the weight at
{cfg.syn_weight_uS * 1e3:g} nS but slowing the decay of the synaptic conductance moved the onset towards the soma,
from {onset_tau[min(onset_tau)]:.0f} µm (τ<sub>2</sub> = {min(onset_tau):g} ms) to
{onset_tau[max(onset_tau)]:.0f} µm (τ<sub>2</sub> = {max(onset_tau):g} ms) (Fig. 4g). Once started, the event
terminated itself: the activation gate rose close to 1 within a few milliseconds, while the inactivation gate fell
from {u['h'][0]:.2f} to {u['h'].min():.3f} and recovered only slowly (Fig. 4i); the local peak occurred
{u['t'][k] - cfg.onset_ms:.0f} ms after the synaptic input. Taken together, these results indicate that
proximal synapses fail to recruit Ca<sub>LVA</sub> because the soma drains their charge before the slowly
activating channels open, and that the electrically more isolated distal dendrite prolongs the local EPSP enough
for the channels to activate and regenerate the depolarisation. We note that, at the most proximal sites, even a
20 ms conductance decay did not suffice, which indicates that there the load of the soma dominates.</p>

<h3>Channel placement matters when the total amount of channels is fixed</h3>
<p>Because the profiles of Fig. 3 differ in their total amount of channels, we repeated the comparison with
equal total conductance (Fig. 5). The uniform profile was unchanged by construction ({_rng(ev_t['uniform'])}),
the denser hotspot triggered events at {_rng(ev_t['hotspot'])} with the largest local
Δ[Ca<sup>2+</sup>]<sub>i</sub> ({st['hotspot'].summary.query("morphology == 'long'").dcai_syn_uM.max():.2f} µM),
and the increasing gradient at {_rng(ev_t['increasing'])}. Unexpectedly at first sight, the decreasing gradient
produced {('no event at any site' if not ev_t['decreasing'] else 'events at ' + _rng(ev_t['decreasing']))}, since
it places most of its channels on the proximal dendrite, where synapses cannot trigger an event regardless of the
local density (Fig. 4f). Among the distributions we tested, the one that weights the distal membrane most
(increasing gradient) boosted the largest number of synapses ({len(ev_t['increasing'])} sites, against
{len(ev_t['uniform'])} for the uniform and {len(ev_t['hotspot'])} for the hotspot profile), which suggests that,
for a fixed number of channels, distal placement extends the boost to more synapses.</p>"""

    if tr is not None:
        ev_tr = _events(tr, "uniform")
        s1 = _v(tr["uniform"], "long", 300, "summation_soma")
        s0 = _v(tr["none"], "long", 300, "summation_soma")
        pk_u, pk_n = _event_peaks(tr["uniform"], 300.0), _event_peaks(tr["none"], 300.0)
        late = max(abs(a - b) for a, b in zip(pk_u[2:], pk_n[2:]))
        n_ev = tr["uniform"].config.n_events
        f_hz = tr["uniform"].config.freq_hz
        out += f"""
<h3>Trains and pharmacology</h3>
<p>During a train of {n_ev} inputs at {f_hz:g} Hz, synapses at {_rng(ev_tr)} again triggered an event with uniform
Ca<sub>LVA</sub>, but only once, at the start of the train. For a synapse at 300 µm, the local response to the
first input reached {pk_u[0]:.0f} mV (against {pk_n[0]:.0f} mV without dendritic Ca<sub>LVA</sub>), the event
extended into the response to the second input ({pk_u[1]:.0f} against {pk_n[1]:.0f} mV), and from the third input
onwards the responses differed by less than {late:.1f} mV from those of the passive dendrite, consistent with the
slow recovery of Ca<sub>LVA</sub> from inactivation (Fig. 4i). Because the event dominates the response to the
first input, the train added little to it: the largest somatic depolarisation during the train was {s1:.2f} times
the first response, against {s0:.2f} times without dendritic Ca<sub>LVA</sub> (Supplementary Fig. S2)."""
        if tx is not None:
            out += f""" Blocking Na<sup>+</sup> channels and removing somatic Ca<sub>LVA</sub> left the results almost
unchanged (somatic EPSP at 300 µm with uniform Ca<sub>LVA</sub>: {_v(tx['uniform'], 'long', 300, 'peak_soma_mV'):.2f}
against {u300:.2f} mV; Supplementary Fig. S3), indicating that the events are carried by the dendritic
Ca<sub>LVA</sub> conductance and do not need Na<sup>+</sup> channels."""
        out += "</p>"
    out += f"""
<p class="meta">A limit of these simulations is that the dendritic Ca<sub>LVA</sub> borrows the kinetics of the
somatic channel and that its dendritic densities are assumptions; the conclusions about where events can be
triggered, however, rest on the passive cable properties of the dendrite and on the slow gating of the channel,
which we varied systematically.</p>"""
    return out


# --- Figure legends ------------------------------------------------------------------------------

def legends_html(R: dict, D: dict, cells: dict, cfg: Config) -> str:
    sp = R["single_peak"]
    ns, nl = _n_sites(sp["none"], "short"), _n_sites(sp["none"], "long")
    short, long = cells["short"], cells["long"]
    syn = (f"Exp2Syn, τ<sub>1</sub> = {cfg.syn_tau1_ms:g} ms, τ<sub>2</sub> = {cfg.syn_tau2_ms:g} ms, "
           f"w = {cfg.syn_weight_uS * 1e3:g} nS")
    legends = [
        ("Figure 1", "figures/morphology/dendrogram.pdf",
         "Short and long versions of the same dendrite of a human L2/3 PV+ interneuron model.",
         f"Dendrograms of the basal tree of the original reconstruction (<b>top</b>, short morphology) and of the same "
         f"cell with the target dendrite grown (<b>bottom</b>, long morphology). Each branch is drawn at its path "
         f"distance from the soma; vertical lines are branch points. The target dendrite (thick) ends at "
         f"{short.tip_distance:.1f} µm in the short and at {long.tip_distance:.1f} µm in the long morphology; the tick "
         f"marks the original tip. All other branches are identical. An interactive 3D view is available in the "
         f"online report."),
        ("Figure 2", "figures/single_peak/overview_uniform.pdf",
         "Uniform dendritic Ca<sub>LVA</sub> produces regenerative events for distal synapses.",
         f"<b>a, b</b>, Dendrograms of the short and long morphologies; the target dendrite of the long morphology is "
         f"coloured by Ca<sub>LVA</sub> density (uniform, {cfg.g_uniform:g} S/cm²) and circles mark the synapse sites "
         f"of the traces in <b>d–f</b>. <b>c</b>, Ca<sub>LVA</sub> density along the long dendrite. <b>d, e</b>, Somatic "
         f"membrane potential for single synaptic events ({syn}) at increasing distance from the soma (colour scale) in "
         f"the short (<b>d</b>) and long (<b>e</b>) morphology. <b>f</b>, Membrane potential at the synapse in the long "
         f"morphology; the delayed second hump is the Ca<sub>LVA</sub> event. <b>g–i</b>, Somatic EPSP amplitude "
         f"(<b>g</b>), local EPSP amplitude (<b>h</b>) and attenuation (<b>i</b>, somatic/local, log scale) as a "
         f"function of synapse distance; dashed, short morphology (n = {ns} sites); solid, long morphology "
         f"(n = {nl} sites). Each point is one simulation with one synapse."),
        ("Figure 3", "figures/single_peak/comparison.pdf",
         "The distance dependence of synaptic efficacy for different Ca<sub>LVA</sub> distributions.",
         f"<b>a</b>, Ca<sub>LVA</sub> density along the long dendrite for the uniform ({cfg.g_uniform:g} S/cm²), hotspot "
         f"({cfg.hotspot_center_um - cfg.hotspot_width_um / 2:g}–{cfg.hotspot_center_um + cfg.hotspot_width_um / 2:g} µm), "
         f"increasing and decreasing profiles (peak {cfg.g_peak:g} S/cm²). <b>b–f</b>, Somatic EPSP (<b>b</b>), local "
         f"EPSP (<b>c</b>), attenuation (<b>d</b>), local Δ[Ca<sup>2+</sup>]<sub>i</sub> (<b>e</b>) and somatic EPSP "
         f"area (<b>f</b>) against synapse distance, for the short morphology (dashed grey) and for the long morphology "
         f"with each profile (colours; dark grey, no dendritic Ca<sub>LVA</sub>). Single events, {syn}; one synapse per "
         f"simulation, sites every {cfg.site_step_um:g} µm (n = {ns} short, {nl} long per profile)."),
        ("Figure 4", "figures/mechanism/mechanism.pdf",
         "Ca<sub>LVA</sub> events require a long-lasting local depolarisation.",
         f"<b>a</b>, Steady-state activation (m<sub>∞</sub>²) and availability (h<sub>∞</sub>) of Ca<sub>LVA</sub>; "
         f"dotted line, resting potential ({D['rest_mV']:.1f} mV). <b>b</b>, Activation and inactivation time constants at "
         f"{cfg.celsius:g} °C. <b>c</b>, Local input impedance (0 Hz) along the target dendrite of the passive short "
         f"(dashed) and long (solid) morphologies; dotted line, soma. <b>d, e</b>, Peak (<b>d</b>) and time above −50 mV "
         f"(<b>e</b>) of the local EPSP without dendritic Ca<sub>LVA</sub> ({syn}). <b>f</b>, Smallest synaptic weight "
         f"that triggers a Ca<sub>LVA</sub> event (uniform profile); triangles, no event up to "
         f"{max(mechanism.SCAN_WEIGHTS)} nS; dotted line, default weight. <b>g</b>, Extra local depolarisation due to "
         f"Ca<sub>LVA</sub> for synaptic decay time constants τ<sub>2</sub> of "
         f"{', '.join(f'{t:g}' for t in mechanism.TAU2_VALUES)} ms at fixed weight; dotted line, event criterion "
         f"({mechanism.EVENT_CRITERION_MV:g} mV). <b>h</b>, Local membrane potential for a synapse at "
         f"{D['example_site']:g} µm with (blue) and without (dashed) Ca<sub>LVA</sub>. <b>i</b>, Ca<sub>LVA</sub> gates at "
         f"the same site: activation (m²), inactivation (h) and open probability (m²h, normalised)."),
        ("Figure 5", "figures/single_total/comparison.pdf",
         "With equal total conductance, distal channels boost more synapses.",
         f"As in Fig. 3, but with every profile scaled to the same total Ca<sub>LVA</sub> conductance on the dendrite, "
         f"equal to a uniform density of {cfg.g_total_equiv:g} S/cm²."),
        ("Supplementary Figure S1", "figures/morphology/diameter.pdf",
         "Geometry of the target dendrite.",
         "<b>a</b>, Diameter along the target dendrite (reconstruction points) in the short (dashed) and long (solid) "
         "morphologies; grey, every other soma-to-tip path of the cell. <b>b</b>, Cumulative membrane area of the "
         "target dendrite from the soma; dotted line, membrane area of the soma."),
        ("Supplementary Figure S2", "figures/train5x50Hz_peak/comparison.pdf",
         "Responses to trains of synaptic inputs.",
         "As in Fig. 3, for trains of 5 events at 50 Hz; amplitudes are the largest depolarisation during the train."),
        ("Supplementary Figure S3", "figures/single_peak_TTX_noSomaCa/comparison.pdf",
         "Dendritic events do not require Na<sup>+</sup> channels or somatic Ca<sub>LVA</sub>.",
         "As in Fig. 3, with NaTg and Nap removed from all compartments (TTX) and somatic Ca<sub>LVA</sub> removed."),
    ]
    items = "".join(f"<div class='legend'><p><b>{name} | {title}</b> {body}</p>"
                    f"<p class='meta'>File: <code>{path}</code></p></div>"
                    for name, path, title, body in legends)
    return items


def _minus(text: str) -> str:
    """Typographic minus for negative numbers in prose (not inside words, DOIs or exponents)."""
    return re.sub(r"(?<![\w.\-/])-(\d)", "\u2212\\1", text)


def html(R: dict, D: dict, cells: dict, cfg: Config) -> str:
    return _minus(f"""
<h2 id="manuscript">Draft manuscript material</h2>
<p class="meta">Draft text for a manuscript, written from the simulations in this report; every number is filled in
from the data when the report is built. Figure numbers refer to the proposed figure set listed under
<i>Figure legends</i>; reference numbers refer to the list at the end. To be edited by the authors.</p>
<h2 id="ms-methods">Methods</h2>{methods_html(cells, cfg, D['rest_mV'])}
<h2 id="ms-results">Results</h2>{results_html(R, D, cells, cfg)}
<h2 id="ms-legends">Figure legends</h2>{legends_html(R, D, cells, cfg)}
{references_html()}""")
