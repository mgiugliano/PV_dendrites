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
from .cell import ALL, SOMATIC
from .config import Config
from .morphology import load_target_meta, path_geometry, side_branches
from .protocols import RHEOBASE_STEP_MS

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
    a = results[prof].summary.query("morphology == 'long'").set_index("site_um")
    b = results["none"].summary.query("morphology == 'long'").set_index("site_um")
    x = np.maximum(a.peak_syn_mV - b.peak_syn_mV, a.peak_tip_mV - b.peak_tip_mV)
    return [float(d) for d in x[x > crit].index]


def _rng(sites):
    if not sites:
        return "no site"
    return f"{min(sites):g} µm" if min(sites) == max(sites) else f"{min(sites):g}–{max(sites):g} µm"








# --- Methods ------------------------------------------------------------------------------------

def methods_html(cells: dict, cfg: Config, rest_mV: float, S: dict, mech_cfg: Config) -> str:
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
<p>To test the role of the proximal diameter, we also simulated both morphologies with a wider "mouth" of the
target dendrite: we multiplied the diameter of each reconstruction point at distance d by
1 + (s − 1)·max(1 − d/ℓ, 0), with s = {max(S['factors']['mouth_scale']):g} and ℓ = {cfg.mouth_length_um:g} µm, the
distance of the second branch point, so that the diameter at the soma grows from {geo[0]['diam_max_um']:.2f} to
{max(S['factors']['mouth_scale']) * geo[0]['diam_max_um']:.2f} µm while the taper and the rest of the dendrite are
unchanged.</p>

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
<p>To test whether channels that activate at more negative potentials change the results, we also used a
copy of this channel for the dendrite (<code>Ca_LVA_dend.mod</code>) in which only the activation gate is shifted:
m<sub>∞</sub> and τ<sub>m</sub> are evaluated at V − ΔV, with ΔV = −15 mV, which moves the half-activation of
m from −40 to −55 mV and leaves inactivation unchanged; with ΔV = 0 the copy reproduces the original channel
exactly. The soma and the axon always kept the original channel.</p>
<p>We defined the dendritic channel density ḡ(d) as a function of the path distance d of each segment from the
soma, over the whole target dendrite (0–{l_meta['tip_distance_um']:.0f} µm), for two profiles: <i>uniform</i>
(ḡ = g) and <i>increasing</i> (ḡ = 2g·min(d/L, 1), with L = {cfg.gradient_span_um:g} µm), which is zero at the
soma, equals g at the midpoint ({cfg.gradient_span_um / 2:g} µm) and 2g at the tip, so that over the nearly
cylindrical dendrite both profiles carry about the same total conductance. We used g =
{' and '.join(f'{g:g}' for g in S['factors']['g_ca_mS_cm2'])} mS/cm². We note that these dendritic densities are
model assumptions, not measurements.</p>

<h3>Somatic current injection</h3>
<p>To let the cell fire, we injected current at the soma in two ways, always relative to the rheobase of each cell
and condition, which we defined as the smallest {RHEOBASE_STEP_MS:g} ms current step that evokes a spike and determined by
bisection (to 2 pA). First, a steady current of {', '.join(f'{100 * f:.0f}' for f in S['bias']['fractions'] if f)}%
of rheobase, applied throughout the simulation including the initialisation, so that each run started from the new
steady state. Second, a fluctuating current with mean {100 * S['noise']['mu_frac']:.0f}% of rheobase and an
Ornstein–Uhlenbeck component of standard deviation {100 * S['noise']['sigma_frac']:.0f}% of rheobase and
correlation time {S['noise']['tau_ms']:g} ms, which made the cell fire irregularly at a few Hz. For each synapse
position we ran {S['noise']['n_trials']} trials with the synapse activated {S['noise']['warmup_ms']:g} ms after the
start, and the same {S['noise']['n_trials']} noise realisations without the synapse; the evoked spike probability is
the difference between the fractions of trials with at least one spike in the {S['noise']['window_ms']:g} ms after
the input.</p>

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
peak depolarisation at the synapse or at the tip of the dendrite, compared with the same synapse in the same
morphology without dendritic Ca<sub>LVA</sub> (the tip is included because events triggered by proximal synapses
start in the distal dendrite), and we
defined the onset distance as the most proximal site with an event. To describe the time course of the EPSP,
both at the soma and at the synapse, we computed its integral above rest, an effective time constant equal to the
integral divided by the peak, and a decay time constant from a log-linear fit of the falling phase between 80% and
20% of the peak.</p>

<h3>Mechanistic analyses</h3>
<p>We computed the local input impedance along the target dendrite, and the transfer impedance to the soma, at
0 and 100 Hz with NEURON's <code>Impedance</code> class, linearising the active conductances around rest
(extended mode), in both morphologies without dendritic Ca<sub>LVA</sub>. We estimated the leak length constant
of the grown dendrite as λ = (R<sub>m</sub> d / 4 R<sub>a</sub>)<sup>1/2</sup> {cite('rall1959')}, with
R<sub>m</sub> = 1/g<sub>pas</sub>. We characterised the passive local EPSP by its peak and by the time it spends
above −50 mV. For these analyses we used {mech_cfg.ca_profile} Ca<sub>LVA</sub> at {mech_cfg.g_ca_mS_cm2:g}
mS/cm² with {'the original activation' if mech_cfg.ca_act_shift_mV == 0 else f'activation shifted by {mech_cfg.ca_act_shift_mV:g} mV'},
the condition with the most robust events. To find the smallest synaptic weight that triggers an event, we scanned w over
{', '.join(f'{x:g}' for x in mechanism.SCAN_WEIGHTS)} nS at {', '.join(f'{x:g}' for x in mechanism.SCAN_SITES)}
µm; to test the role of the EPSP duration, we varied τ<sub>2</sub> over
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

KIN = {0.0: "original", -15.0: "shifted"}


def _cond(shift, g):
    return f"{g:g} mS/cm² with {'the original' if shift == 0 else 'shifted'} activation"


def _events_txt(res, prof):
    ev = _events(res, prof)
    return f"events for synapses at {_rng(ev)}" if ev else "no event at any site"


def _boost(res, prof, site):
    return _v(res[prof], "long", site, "peak_soma_mV") / _v(res["none"], "long", site, "peak_soma_mV")


def _fmt0(d):
    return "–" if d != d else f"{d:.0f}"


def _distal_initiation(res) -> str:
    """Sentence on events triggered by intermediate synapses but generated distally (from the tip boost)."""
    if res is None:
        return ""
    a = res["uniform"].summary.query("morphology == 'long'").set_index("site_um")
    b = res["none"].summary.query("morphology == 'long'").set_index("site_um")
    loc, tip = a.peak_syn_mV - b.peak_syn_mV, a.peak_tip_mV - b.peak_tip_mV
    sites = [d for d in loc.index if tip[d] > mechanism.EVENT_CRITERION_MV and loc[d] < 2]
    if not sites:
        return ""
    d = sites[len(sites) // 2]
    return (f"Interestingly, with shifted activation, synapses at {min(sites):g}–{max(sites):g} µm triggered events "
            f"that were barely visible at the synapse but large at the tip: for a synapse at {d:g} µm, Ca<sub>LVA</sub> "
            f"added {loc[d]:.1f} mV at the synapse and {tip[d]:.0f} mV at the tip. Their depolarisation spreads to the "
            "sealed distal end, where the input impedance is highest, and the event starts there. ")


def _threshold_txt(fail, ok) -> str:
    parts = []
    if len(fail):
        parts.append(f"no synapse at {', '.join(f'{d:g}' for d in fail)} µm triggered an event even at "
                     f"{max(mechanism.SCAN_WEIGHTS)} nS")
    if len(ok):
        o = ok.sort_values("distance_um")
        parts.append(f"the synaptic weight needed to trigger an event rose from {o.threshold_nS.iloc[-1]:g} nS at "
                     f"{o.distance_um.iloc[-1]:g} µm to {o.threshold_nS.iloc[0]:g} nS at {o.distance_um.iloc[0]:g} µm")
    return ", and ".join(parts)


def _rest_firing(single, train) -> str:
    n = sum(int(r.summary.n_spikes_soma.sum()) for grid in (single, train) for res in grid.values() for r in res.values())
    return ("No single synapse or train made the resting cell fire." if n == 0
            else f"At rest, single synapses or trains evoked {n} somatic spikes in total.")


def results_html(single, train, bias, noise_ev, D, cells, cfg, S) -> str:
    n = single[(0.0, 1.0, 1.0)]["none"]
    short, long = cells["short"], cells["long"]
    zl, zs = D["impedance"]["long"], D["impedance"]["short"]
    z = lambda df, d: float(df.iloc[int(np.argmin(np.abs(df.distance_um - d)))]["zin_0Hz_MOhm"])  # noqa: E731
    ns = n.summary.query("morphology == 'long'").set_index("site_um")
    keys = sorted({(s, g) for s, g, m in single if m == 1.0}, key=lambda k: (k[0] != 0, k[1]))
    out = f"""
<h3>Growing the dendrite isolates its distal inputs electrically</h3>
<p>The dendrites of fast-spiking PV+ interneurons are thin, and their cable properties shape how quickly and how
efficiently excitatory inputs reach the soma {cite('hu2010', 'norenberg2010')}. To ask how the length of a single
dendrite affects this, we compared the original reconstruction of a human L2/3 PV+ interneuron model
{cite('yao2022')}, in which the chosen dendrite ends {short.tip_distance:.1f} µm from the soma, with a copy in which
we grew the same dendrite to {long.tip_distance:.0f} µm (Fig. 1). Along the grown dendrite, the local input
impedance rose from {z(zl, 10):.0f} MΩ at 10 µm to {z(zl, 400):.0f} MΩ at the tip, against
{zl.zin_soma_0Hz_MOhm.iloc[0]:.1f} MΩ at the soma (Fig. 2a), so that a {cfg.syn_weight_uS * 1e3:g} nS synapse
depolarised its own membrane more, and for longer, the farther it was from the soma: the effective time constant
of the local EPSP grew from {ns.tau_eff_ms_syn[50]:.1f} ms at 50 µm to {ns.tau_eff_ms_syn[400]:.1f} ms at the tip
(Fig. 2e). At the soma, however, the same synapse produced {_v(n, 'long', 10, 'peak_soma_mV'):.2f} mV at 10 µm and
only {_v(n, 'long', 400, 'peak_soma_mV'):.2f} mV at 400 µm (Fig. 3a). Already at 100 µm the somatic EPSP was smaller
in the long ({_v(n, 'long', 100, 'peak_soma_mV'):.2f} mV) than in the short morphology
({_v(n, 'short', 100, 'peak_soma_mV'):.2f} mV), where the same site is the sealed tip, with a higher local input
impedance ({z(zs, 100):.0f} against {z(zl, 100):.0f} MΩ).</p>
"""
    # Ca_LVA, original kinetics, then shifted
    par = []
    for shift, g in keys:
        res = single[(shift, g, 1.0)]
        par.append(f"at {_cond(shift, g)}, the uniform profile gave {_events_txt(res, 'uniform')} and the increasing "
                   f"profile {_events_txt(res, 'increasing')}, raising the somatic EPSP of a synapse at 300 µm "
                   f"{_boost(res, 'uniform', 300):.2f}-fold and {_boost(res, 'increasing', 300):.2f}-fold, respectively")
    near = []
    for shift, g in keys:
        res = single[(shift, g, 1.0)]
        for prof in ("uniform", "increasing"):
            r = res[prof].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
            b = res["none"].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
            hit = (r / b - 1)[(r / b - 1) > 0.10].index
            near.append((shift, g, prof, float(min(hit)) if len(hit) else np.nan))
    ex = single[(-15.0, max(g for _, g in keys), 1.0)] if (-15.0, max(g for _, g in keys), 1.0) in single else None
    out += f"""
<h3>Dendritic Ca<sub>LVA</sub> boosts distal inputs, above a density threshold</h3>
<p>We then inserted a low-voltage-activated Ca<sup>2+</sup> conductance (Ca<sub>LVA</sub>) along the grown dendrite,
either with a uniform density or with a density increasing linearly from the soma, at two densities and with the
original or a shifted activation curve (Fig. 3). Specifically, {'; '.join(par)}. The synapses closest to the soma
were never boosted, but how close the boost reached depended on the condition: the nearest synapse whose somatic
EPSP grew by more than 10% sat at {'; '.join(f'{_fmt0(d)} µm ({g:g} mS/cm², {KIN[s]}, {p})' for s, g, p, d in near)}.
{_distal_initiation(ex)}Taken together, these results indicate that dendritic Ca<sub>LVA</sub> selectively amplifies
inputs beyond a distance that shrinks as the channel density increases and as activation moves to more negative
potentials, and that the increasing profile, which places more channels distally, is the more effective.</p>
"""
    # mouth
    ch = []
    for shift, g in keys:
        for prof in ("none", "uniform", "increasing"):
            a = single[(shift, g, 1.0)][prof].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
            b = single[(shift, g, 1.5)][prof].summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
            ch.append((100 * (b / a - 1)).values)
    ch = np.concatenate(ch)
    out += f"""
<h3>A wider proximal dendrite changes little</h3>
<p>Because the soma acts as a strong current sink, we asked whether widening the proximal end of the dendrite
would change these results. Increasing the diameter at the soma 1.5-fold, fading back to the original taper within
{cfg.mouth_length_um:g} µm, changed the somatic EPSP by {ch.min():+.1f}% to {ch.max():+.1f}% across all sites and
conditions, with the largest effect on the most proximal synapses, and did not change which synapses triggered
Ca<sub>LVA</sub> events (Supplementary Fig. S2).</p>
"""
    # mechanism
    mc = D["cfg"]
    th = D["thresholds"]
    fail = th[th.threshold_nS.isna()].distance_um
    ok = th[th.threshold_nS.notna()]
    onset = {t: mechanism.onset_um(df) for t, df in D["tau"].items()}
    el = D["epsp"]["long"]
    t_above = lambda d: float(el.iloc[int(np.argmin(np.abs(el.distance_um - d)))].t_above_ms)  # noqa: E731
    u = D["example"][mc.ca_profile]
    out += f"""
<h3>The duration of the depolarisation, not its amplitude, gates the event</h3>
<p>To understand why synapses close to the soma rarely trigger events, we analysed the condition with the most robust events
({mc.ca_profile}, {mc.g_ca_mS_cm2:g} mS/cm², {KIN[mc.ca_act_shift_mV]} activation; Fig. 4). At rest only
{100 * D['h_rest']:.0f}% of the channels are available, and activation is slow (Fig. 4a,b). The time the passive
local EPSP spent above −50 mV increased with distance, from {t_above(100):.1f} ms at 100 µm to {t_above(400):.1f} ms
at the tip (Fig. 4e), while its peak levelled off as the membrane approached the synaptic reversal potential
(Fig. 4d). Two manipulations supported this interpretation. First, {_threshold_txt(fail, ok)} (Fig. 4f). Second, slowing the synaptic conductance decay at fixed weight moved the onset towards the soma, from
{onset[min(onset)]:.0f} µm (τ<sub>2</sub> = {min(onset):g} ms) to {onset[max(onset)]:.0f} µm (τ<sub>2</sub> =
{max(onset):g} ms) (Fig. 4g). Once started, the event terminated itself as the inactivation gate fell from
{u['h'][0]:.2f} to {u['h'].min():.3f} (Fig. 4i).</p>
"""
    # trains
    tk = sorted({(s, g) for s, g, m in train}, key=lambda k: (k[0] != 0, k[1]))
    tr_par = "; ".join(f"{_cond(s, g)}: uniform {_events_txt(train[(s, g, 1.0)], 'uniform')}, increasing "
                       f"{_events_txt(train[(s, g, 1.0)], 'increasing')}" for s, g in tk)
    out += f"""
<h3>Trains, steady depolarisation and noisy input</h3>
<p>With trains of 5 inputs at 50 Hz, the same pattern held ({tr_par}; Supplementary Fig. S3). {_rest_firing(single, train)} To let the cell fire, we first held the soma with a steady current below rheobase
(Supplementary Fig. S4)."""
    for (shift, g), study in bias.items():
        hi = max(study)
        res = study[hi]
        rest = res["none"].summary.query("morphology == 'long'").vrest_soma_mV.iloc[0]
        fired = {p: res[p].summary.query("morphology == 'long' and n_spikes_soma > 0").site_um.tolist() for p in res}
        out += (f" With {_cond(shift, g)} at {100 * hi:.0f}% of rheobase (rest {rest:.1f} mV), the synapses that "
                f"fired the cell were at {', '.join(f'{_rng(v)} ({p})' if v else f'no site ({p})' for p, v in fired.items())}.")
        mid = 0.5 if 0.5 in study else None
        if mid is not None:
            b0, bm = _boost(study[0.0], "uniform", 300), _boost(study[mid], "uniform", 300)
            out += (f" Below threshold, the uniform-profile boost at 300 µm went from {b0:.2f}-fold at rest to "
                    f"{bm:.2f}-fold at {100 * mid:.0f}% of rheobase"
                    + (", consistent with depolarisation inactivating Ca<sub>LVA</sub>." if bm < b0 - 0.05 else "."))
    if noise_ev is not None and len(noise_ev):
        far = noise_ev[(noise_ev.site_um >= 150) & (noise_ev.site_um <= 250)].groupby("condition", sort=False).p_evoked.mean()
        near = noise_ev[noise_ev.site_um <= 100].groupby("condition", sort=False).p_evoked.mean()
        out += (" Finally, with a fluctuating somatic current that made the cell fire irregularly (Fig. 5), the "
                "extra spike probability added by a synapse at 50–100 µm was "
                + ", ".join(f"{near[c]:.2f} ({c.replace('Ca_LVA', 'Ca<sub>LVA</sub>')})" for c in near.index)
                + ", and at 150–250 µm "
                + ", ".join(f"{far[c]:.2f} ({c.replace('Ca_LVA', 'Ca<sub>LVA</sub>')})" for c in far.index)
                + f"; with {S['noise']['n_trials']} trials per site (one trial = {1 / S['noise']['n_trials']:.2f}) "
                  "these estimates are preliminary.")
    out += "</p>"
    out += """
<p class="meta">A limit of these simulations is that the dendritic Ca<sub>LVA</sub> borrows the kinetics of the
somatic channel and that its dendritic densities are assumptions; we therefore varied density, distribution and
activation range systematically.</p>"""
    return out


# --- Figure legends ------------------------------------------------------------------------------

def legends_html(single, D, cells, cfg, S) -> str:
    short, long = cells["short"], cells["long"]
    F = S["factors"]
    syn = (f"Exp2Syn, τ<sub>1</sub> = {cfg.syn_tau1_ms:g} ms, τ<sub>2</sub> = {cfg.syn_tau2_ms:g} ms, "
           f"w = {cfg.syn_weight_uS * 1e3:g} nS")
    gs = " and ".join(f"{g:g}" for g in F["g_ca_mS_cm2"])
    N = S["noise"]
    legends = [
        ("Figure 1", "figures/morphology/dendrogram.pdf",
         "Short and long versions of the same dendrite of a human L2/3 PV+ interneuron model.",
         f"Dendrograms of the basal tree of the original reconstruction (top, short morphology) and of the same cell with "
         f"the target dendrite grown (bottom, long morphology). Each branch is drawn at its path distance from the soma; "
         f"vertical lines are branch points. The target dendrite (thick) ends at {short.tip_distance:.1f} µm in the short "
         f"and at {long.tip_distance:.1f} µm in the long morphology; the tick marks the original tip."),
        ("Figure 2", "figures/electrotonic/electrotonic.pdf",
         "Electrotonic structure of the target dendrite and EPSP time course.",
         "<b>a</b>, Local input impedance along the target dendrite at 0 Hz (thick) and 100 Hz (thin) in the short "
         "(dashed), long (solid) and long morphology with a wider mouth (dash-dot); dotted line, soma. <b>b</b>, Transfer "
         "impedance to the soma. <b>c</b>, Steady-state voltage attenuation to the soma. <b>d–f</b>, Integral (log "
         "scale), effective time constant (integral/peak) and decay time constant (exponential fit, 80–20% of the peak) "
         f"of the EPSP at the soma (thick) and at the synapse (thin) against synapse distance ({syn}), without and with "
         f"dendritic Ca<sub>LVA</sub> ({D['cfg'].ca_profile}, {D['cfg'].g_ca_mS_cm2:g} mS/cm², "
         f"{KIN[D['cfg'].ca_act_shift_mV]} activation)."),
        ("Figure 3", "figures/single/factorial.pdf",
         "Dendritic Ca<sub>LVA</sub> boosts distal inputs depending on density, distribution and activation range.",
         f"Columns: Ca<sub>LVA</sub> density ({gs} mS/cm²) with the original (half-activation of m at −40 mV) or shifted "
         f"(−55 mV) activation. <b>a–d</b>, Somatic EPSP against synapse distance for the short morphology (dashed grey), "
         f"the long morphology without dendritic Ca<sub>LVA</sub> (dark grey), and with uniform (blue) or increasing "
         f"(green) density. <b>e–h</b>, Extra local depolarisation due to Ca<sub>LVA</sub>; dotted line, event "
         f"criterion ({mechanism.EVENT_CRITERION_MV:g} mV). Single events, {syn}; one synapse per simulation, every "
         f"{cfg.site_step_um:g} µm."),
        ("Figure 4", "figures/mechanism/mechanism.pdf",
         "Ca<sub>LVA</sub> events require a long-lasting local depolarisation.",
         f"Condition: {D['cfg'].ca_profile}, {D['cfg'].g_ca_mS_cm2:g} mS/cm², {KIN[D['cfg'].ca_act_shift_mV]} "
         "activation. <b>a</b>, Steady-state activation (m∞², dashed: original) and availability (h∞); dotted line, rest. "
         "<b>b</b>, Time constants. <b>c</b>, Local input impedance. <b>d, e</b>, Peak and time above −50 mV of the "
         "passive local EPSP. <b>f</b>, Smallest synaptic weight that triggers an event; triangles, no event up to "
         f"{max(mechanism.SCAN_WEIGHTS)} nS. <b>g</b>, Local boost for synaptic decay time constants of "
         f"{', '.join(f'{t:g}' for t in mechanism.TAU2_VALUES)} ms. <b>h</b>, Local potential with and without "
         f"Ca<sub>LVA</sub> for a synapse at {D['example_site']:g} µm. <b>i</b>, Ca<sub>LVA</sub> gates at the same site."),
        ("Figure 5", "figures/noise/noise.pdf",
         "Distal synapses and firing with a noisy somatic current.",
         f"The soma receives a mean current of {100 * N['mu_frac']:.0f}% of rheobase plus Ornstein–Uhlenbeck noise "
         f"(s.d. {100 * N['sigma_frac']:.0f}% of rheobase, {N['tau_ms']:g} ms). <b>a, b</b>, Somatic potential in three "
         f"trials with the same noise without (grey) and with (blue) a synapse at {N['example_site_um']:g} µm. <b>c</b>, "
         f"Spike probability added by the synapse within {N['window_ms']:g} ms, against its distance ({N['n_trials']} "
         "paired trials per site). <b>d</b>, Median latency of the first spike."),
        ("Supplementary Figure S1", "figures/morphology/diameter.pdf", "Geometry of the target dendrite.",
         "<b>a</b>, Diameter along the target dendrite in the short, long and wide-mouth morphologies; grey, every other "
         "soma-to-tip path. <b>b</b>, Cumulative membrane area; dotted line, soma."),
        ("Supplementary Figure S2", "figures/single/mouth.pdf", "Effect of a wider proximal dendrite.",
         "Relative change of the somatic EPSP with the 1.5× mouth against synapse distance, for each density and "
         "activation range."),
        ("Supplementary Figure S3", "figures/train/factorial.pdf", "Trains of synaptic inputs.",
         "As Fig. 3, for 5 inputs at 50 Hz; amplitudes are the largest depolarisation during the train."),
        ("Supplementary Figure S4", "figures/bias/", "Firing with a steady somatic current.",
         "Somatic and local potential for synapses every 20 µm with a steady somatic current at 95% of rheobase, for "
         "each activation range; thick traces, the cell fires."),
    ]
    return "".join(f"<div class='legend'><p><b>{name} | {title}</b> {body}</p>"
                   f"<p class='meta'>File: <code>{path}</code></p></div>" for name, path, title, body in legends)


def _minus(text: str) -> str:
    """Typographic minus for negative numbers in prose (not inside words, DOIs or exponents)."""
    return re.sub(r"(?<![\w.\-/])-(\d)", "\u2212\\1", text)


def html(single, train, bias, noise_ev, D, cells, cfg, S) -> str:
    return _minus(f"""
<h2 id="manuscript">Draft manuscript material</h2>
<p class="meta">Draft text for a manuscript, written from the simulations in this report; every number is filled in
from the data when the report is built. Figure numbers refer to the proposed figure set listed under
<i>Figure legends</i>; reference numbers refer to the list at the end. To be edited by the authors.</p>
<h2 id="ms-methods">Methods</h2>{methods_html(cells, cfg, D['rest_mV'], S, D['cfg'])}
<h2 id="ms-results">Results</h2>{results_html(single, train, bias, noise_ev, D, cells, cfg, S)}
<h2 id="ms-legends">Figure legends</h2>{legends_html(single, D, cells, cfg, S)}
{references_html()}""")
