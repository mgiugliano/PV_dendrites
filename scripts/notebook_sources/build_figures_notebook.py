"""Generate notebooks/figures.ipynb (the step-by-step figure notebook).

The notebook is generated from this script so that its text, equations and code stay in one
reviewable place. Run:  python scripts/notebook_sources/build_figures_notebook.py
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
ROOT = Path(__file__).resolve().parents[2]
BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/mgiugliano/PV_dendrites/blob/main/notebooks/figures.ipynb)")

cells = [
md(f"""# Every figure of the report, step by step

{BADGE}

This notebook regenerates **every figure** of the report on a short (100 µm) versus grown (400 µm) dendrite of a
human layer 2/3 parvalbumin-positive (PV+) interneuron model, one figure at a time. Each section first explains
what is simulated and how it is measured (with the equations), then runs the simulations and draws the figure.
Run the cells from top to bottom; nothing else is needed, locally or on Google Colab.

**Run time.** With `QUICK = True` (default) the notebook uses a coarser spacing of the synapse positions (every 20 µm
from 20 µm, instead of every 10 µm from 10 µm), two instead of three levels of somatic bias, and fewer noise trials: all figures are produced in about
40 minutes on a recent laptop and about 1–1.5 hours on Colab's free tier, with noisier curves in the last section. With `QUICK = False` it
reproduces the report figures exactly, but needs several hours (most of them for the noisy-current study).
Simulations are cached on disk: re-running a cell only redraws.

The model is that of Yao *et al.* (2022, *Cell Reports* 38:110232; code doi:10.5281/zenodo.5771000), translated
to Python for NEURON. All the code called here lives in `src/pvdend/` (one function per step) and is the same code
that builds the HTML report."""),

md("""## 0. Setup

On Colab this cell downloads the repository, installs NEURON and compiles the ion-channel files (`mod/*.mod`)
into a library that NEURON loads. Locally it only compiles them if needed. It also chooses where results and
figures go: in quick mode, `results_quick/` and `figures_quick/`, so that they never mix with the full results."""),
code('''QUICK = True   # True: faster, coarser (all figures in ~40 min locally, ~1-1.5 h on Colab); False: exact report figures (several hours)

import os, sys, shutil, subprocess
from pathlib import Path

REPO_URL = "https://github.com/mgiugliano/PV_dendrites"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    ROOT = Path("/content/PV_dendrites")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuron", "plotly"], check=True)
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# Compile the NEURON mechanisms once, with the nrnivmodl that belongs to this Python's NEURON.
def compiled():
    return any((ROOT / a / lib).exists() for a in ("x86_64", "arm64", "aarch64")
               for lib in (".libs/libnrnmech.so", "libnrnmech.so", "libnrnmech.dylib"))
if not compiled():
    local = Path(sys.executable).parent / "nrnivmodl"
    nrnivmodl = str(local) if local.exists() else shutil.which("nrnivmodl")
    r = subprocess.run([nrnivmodl, "mod"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not compiled():
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("Compiling the NEURON mechanisms failed (see above).")

# Where results and figures go (must be set before importing pvdend).
os.environ["PVDEND_ROOT"] = str(ROOT)
if QUICK:
    os.environ["PVDEND_RESULTS"] = str(ROOT / "results_quick")
    os.environ["PVDEND_FIGURES"] = str(ROOT / "figures_quick")
sys.path.insert(0, str(ROOT / "src"))
print("Repository:", ROOT, "| Colab:", IN_COLAB, "| quick mode:", QUICK)'''),
code('''%matplotlib inline
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from pvdend import studies, mechanism, plotting, viewer3d, get_cell
from pvdend._paths import FIGURES_DIR

# The study definition (configs/studies.json): factors, protocols and their parameters.
S = copy.deepcopy(studies.spec())
if QUICK:
    S["base_overrides"] = {"site_step_um": 20.0, "site_start_um": 20.0}  # synapse every 20 µm (20-400) instead of 10
    S["bias"]["fractions"] = [0.0, 0.95]                # no bias and near threshold only
    S["noise"].update(n_blocks=4, sites_um=[100, 200, 300, 400], workers=max(1, os.cpu_count() or 1))
cfg = studies.base_config(S)                            # common settings of every simulation
plotting.set_style()

def show(fig, stem):
    """Display a figure in the notebook and save it as PDF and PNG in FIGURES_DIR."""
    plotting.save_figure(fig, stem, formats=("pdf", "png"))
    plt.show()

progress = lambda name: print(f"\\r  {name:<60}", end="", flush=True)
print("Figures are saved in", FIGURES_DIR)
print("Factors:", S["factors"])'''),

md(r"""## 1. Model and morphologies (Figure 1, Supplementary Figure S1)

**The cell.** Each compartment of the model obeys the cable equation

$$
c_m \frac{\partial V}{\partial t} \;=\; \frac{d}{4 R_a}\,\frac{\partial^2 V}{\partial x^2}
\;-\; g_{\mathrm{pas}}\,(V - E_{\mathrm{pas}}) \;-\; I_h \;-\; I_{\mathrm{active}} \;-\; I_{\mathrm{syn}},
$$

with membrane capacitance $c_m = 2\ \mu\mathrm{F/cm^2}$, axial resistivity $R_a = 100\ \Omega\,\mathrm{cm}$, leak
$g_{\mathrm{pas}} = 1.18\times10^{-4}\ \mathrm{S/cm^2}$ reversing at $E_{\mathrm{pas}} = -83.9$ mV, and the
hyperpolarisation-activated current $I_h$ everywhere. Active Na⁺, K⁺ and Ca²⁺ currents are present only in the soma and
the axon, as in the original model; the dendrites are passive.

**The two morphologies.** *Short*: the original reconstruction; the *target dendrite* is the terminal path whose tip is
closest to 100 µm from the soma (100.8 µm). *Long*: the same cell with that tip grown to 400 µm by a random walk,
$\mathbf{u}_{k+1} = (\mathbf{u}_k + \boldsymbol{\xi}_k)/\lVert \mathbf{u}_k + \boldsymbol{\xi}_k \rVert$ with steps of 5 µm,
Gaussian $\boldsymbol{\xi}_k$ (s.d. 0.1, seed 42) and constant tip diameter. Distances are path distances from the centre
of the soma.

**The wider mouth.** In one condition the diameter of the target dendrite is multiplied by
$1 + (s-1)\max(1 - d/\ell, 0)$, with $s = 1.5$ and $\ell = 13$ µm: 1.5× at the soma, back to 1× at the second branch
point.

The **dendrogram** draws each branch at its path distance from the soma (x) and stacks the branches (y)."""),
code('''cells = {m: get_cell(m, cfg) for m in ("short", "long")}           # the two cells (NEURON objects)
mouth_cell = get_cell("long", cfg.replace(mouth_scale=1.5))         # long cell with the 1.5x mouth

show(plotting.dendrogram_figure(cells), "morphology/dendrogram")                 # Figure 1
show(plotting.diameter_figure(cells, mouth_cell=mouth_cell), "morphology/diameter")  # Supplementary Figure S1'''),
md("Interactive 3D view of the long cell (drag to rotate; click *grown extension* in the legend to hide it):"),
code('''viewer3d.morphology_3d(cells["long"], cells["short"].tip_distance).show()'''),

md(r"""## 2. Single synaptic events across the factorial design (Figures 2 and 3)

**Synapse.** One excitatory synapse at a time, a conductance with a double-exponential time course,

$$
g_{\mathrm{syn}}(t) = w\,f\,\big(e^{-t/\tau_2} - e^{-t/\tau_1}\big), \qquad I_{\mathrm{syn}} = g_{\mathrm{syn}}(t)\,(V - E_{\mathrm{syn}}),
$$

with $w = 5$ nS (peak), $\tau_1 = 0.3$ ms, $\tau_2 = 3$ ms, $E_{\mathrm{syn}} = 0$ mV ($f$ normalises the peak to $w$). It is
placed every 10 µm (20 µm in quick mode) along the target dendrite, at the same distances in both cells.

**Dendritic Ca<sub>LVA</sub>** (long cell only, whole target dendrite):

$$
I_{\mathrm{Ca}} = \bar g(d)\, m^2 h\, (V - E_{\mathrm{Ca}}),\qquad
\frac{dx}{dt} = \frac{x_\infty(V) - x}{\tau_x(V)}\ \ (x = m, h),
$$

$$
m_\infty = \frac{1}{1 + e^{-(V' - \Delta V + 30)/6}},\quad
\tau_m = \frac{5 + 20/(1 + e^{(V' - \Delta V + 25)/5})}{q},\quad
h_\infty = \frac{1}{1 + e^{(V' + 80)/6.4}},\quad
\tau_h = \frac{20 + 50/(1 + e^{(V' + 40)/7})}{q},
$$

with $V' = V + 10$ mV (junction-potential correction of the original channel), $q = 2.3^{(34-21)/10} = 2.95$, and the
activation shift $\Delta V = 0$ (original) or $-15$ mV (half-activation of $m$ moves from −40 to −55 mV; inactivation
unchanged). $E_{\mathrm{Ca}}$ follows the Nernst equation, with $[\mathrm{Ca}^{2+}]_i$ from a 0.1 µm shell,
$d[\mathrm{Ca}^{2+}]_i/dt = -10^4 \gamma I_{\mathrm{Ca}}/(2 F d) - ([\mathrm{Ca}^{2+}]_i - 100\ \mathrm{nM})/\tau_{\mathrm{Ca}}$.

**Density profiles**, with $g$ = 1 or 2.5 mS/cm² and $L = 400$ µm: *uniform* $\bar g(d) = g$; *increasing*
$\bar g(d) = 2g\,\min(d/L, 1)$ (0 at the soma, $g$ at 200 µm, $2g$ at the tip, about the same total conductance).

**Measures.** EPSP amplitude = peak minus the mean of the 5 ms before the input, at the soma and at the synapse. A
**Ca<sub>LVA</sub> event** is counted when Ca<sub>LVA</sub> adds more than 10 mV to the peak at the synapse *or at the tip*
(events triggered by synapses near the soma start at the distal end)."""),
code('''single = studies.grid("single", S, progress)     # {(shift, g, mouth): {"none"|"uniform"|"increasing": Result}}
print()

# Summary table: somatic EPSP (mV) at a few distances for every condition (mouth 1x).
rows = []
for (shift, g, mouth), res in single.items():
    if mouth != 1.0:
        continue
    for prof, r in res.items():
        s = r.summary.query("morphology == 'long'").set_index("site_um").peak_soma_mV
        rows.append({"kinetics": "original" if shift == 0 else f"shift {shift:g} mV", "g (mS/cm²)": g, "profile": prof,
                     **{f"{d:g} µm": round(s.get(d, np.nan), 2) for d in (100.0, 200.0, 300.0, 400.0)}})
display(pd.DataFrame(rows).drop_duplicates(subset=["profile", "100 µm", "200 µm"]))

show(plotting.factorial_figure(single), "single/factorial")      # Figure 3'''),

md(r"""**Electrotonic structure and EPSP time course (Figure 2).** The *input impedance* $Z_{\mathrm{in}}(x, f)$ is the voltage
response at point $x$ to a sinusoidal current injected at $x$, and the *transfer impedance* $Z_{\mathrm{tr}}(x, f)$ the
response at the soma to the same current; their ratio $|Z_{\mathrm{tr}}|/|Z_{\mathrm{in}}|$ is the voltage attenuation
from $x$ to the soma. They are computed at 0 and 100 Hz (NEURON `Impedance`, active conductances linearised at rest).

For each EPSP $\Delta V(t)$ (at the soma and at the synapse) we compute its integral $A = \int \Delta V\,dt$, the
*effective time constant* $\tau_{\mathrm{eff}} = A/\Delta V_{\mathrm{peak}}$, and the *decay time constant* from a fit of
$\Delta V(t) \propto e^{-t/\tau_{\mathrm{decay}}}$ between 80% and 20% of the peak."""),
code('''M = S["mechanism"]
mech_cfg = cfg.replace(name="mechanism", ca_profile=M["profile"], g_ca_mS_cm2=M["g_ca_mS_cm2"],
                       ca_act_shift_mV=M["ca_act_shift_mV"])
imp = {"short": mechanism.impedance_profile("short", cfg),
       "long": mechanism.impedance_profile("long", cfg),
       "long, 1.5× mouth": mechanism.impedance_profile("long", cfg.replace(mouth_scale=1.5))}
none = {"short": single[(0.0, 1.0, 1.0)]["none"], "long": single[(0.0, 1.0, 1.0)]["none"],
        "long, 1.5× mouth": single[(0.0, 1.0, 1.5)]["none"]}
with_ca = single[(mech_cfg.ca_act_shift_mV, mech_cfg.g_ca_mS_cm2, 1.0)][mech_cfg.ca_profile]
show(plotting.electrotonic_figure(imp, none, with_ca, "uniform, 2.5 mS/cm², shifted −15 mV"),
     "electrotonic/electrotonic")                               # Figure 2'''),

md("""**Wider mouth (Supplementary Figure S2)**: relative change of the somatic EPSP when the proximal diameter is 1.5×.
**Per-condition details**: one overview figure per density, kinetics and profile (traces, dendrograms coloured by
channel density, distance dependence)."""),
code('''show(plotting.mouth_figure(single), "single/mouth")             # Supplementary Figure S2

for (shift, g, mouth), res in single.items():
    if mouth != 1.0:
        continue
    for prof in ("uniform", "increasing"):
        fig = plotting.overview_figure(res[prof], {m: get_cell(m, res[prof].config) for m in ("short", "long")})
        show(fig, f"single/overview_{prof}_g{g:g}_s{shift:g}")'''),

md(r"""## 3. Mechanism (Figure 4)

For the condition with the most robust events (uniform Ca<sub>LVA</sub>, 2.5 mS/cm², shifted activation) we compute:
(a, b) the channel's steady states and time constants; (c) the local input impedance; (d, e) the peak and the time spent
above −50 mV of the passive local EPSP; (f) the smallest synaptic weight that triggers an event at each distance (weights
1–40 nS); (g) how the event onset moves when the synaptic decay $\tau_2$ is slowed to 6, 10 and 20 ms; (h, i) the local
potential and the gates $m^2$, $h$ during one event."""),
code('''D = mechanism.collect(mech_cfg)        # runs the scans (cached) and returns everything the figure needs
display(D["thresholds"].rename(columns={"distance_um": "synapse (µm)", "threshold_nS": "threshold weight (nS)"}))
show(plotting.mechanism_figure(D), "mechanism/mechanism")        # Figure 4'''),

md("""## 4. Trains of synaptic inputs (Supplementary Figure S3)

As in section 2, but each synapse is activated 5 times at 50 Hz; amplitudes are the largest depolarisation during the
train."""),
code('''train = studies.grid("train", S, progress)
print()
show(plotting.factorial_figure(train), "train/factorial")        # Supplementary Figure S3'''),

md(r"""## 5. Firing with a steady somatic current (Supplementary Figure S4)

The **rheobase** $I_{\mathrm{rh}}$ is the smallest 1 s current step at the soma that evokes a spike (bisection to 2 pA),
computed for every cell and channel configuration. A steady current $I = \alpha\, I_{\mathrm{rh}}$ ($\alpha$ = 0.5 or 0.95)
is then applied from the start, so the cell rests closer to threshold, and the single-synapse sweep is repeated. A spike is
a crossing of 0 mV at the soma."""),
code('''bias = studies.bias_grid(S, progress)     # {(shift, g): {fraction: {profile: Result}}}
print()
for (shift, g), study in bias.items():
    tag = f"g{g:g}_s{shift:g}"
    show(plotting.spike_traces_figure(study, max(study)), f"bias/spike_traces_{tag}")    # Supplementary Figure S4
    show(plotting.firing_figure(study, S["bias"]["example_site_um"]), f"bias/firing_{tag}")'''),

md(r"""## 6. Firing with a noisy somatic current (Figure 5)

The soma receives a fluctuating current, a mean plus an Ornstein–Uhlenbeck process,

$$
I(t) = \mu + \eta(t), \qquad d\eta = -\frac{\eta}{\tau}\,dt + \sigma\sqrt{\frac{2}{\tau}}\;dW,
$$

with $\mu = 0.8\,I_{\mathrm{rh}}$, $\sigma = 0.4\,I_{\mathrm{rh}}$ and $\tau = 3$ ms, so that the cell fires irregularly at a
few Hz. Each simulation (block) delivers 20 synaptic inputs, one every 250 ms, and is repeated with **the same noise and no
synapse**; the difference isolates the spikes added by the synapse. Blocks run in parallel on all processor cores.

From the spike times $t_k$ relative to each input we compute the **PSTH** and the **cumulative extra spikes per input**

$$
C(t) = \frac{1}{N}\sum_{j=1}^{N}\Big(\#\{t_k^{\,\mathrm{syn}} \le t\}_j - \#\{t_k^{\,\mathrm{no\,syn}} \le t\}_j\Big),
$$

whose value at 60 ms is the number of extra spikes per input (panel f); 95% confidence intervals come from 1000 bootstrap
resamples of the $N$ inputs. Panels a–b show one input, chosen automatically, where only the cell with dendritic
Ca<sub>LVA</sub> fires."""),
code('''N = S["noise"]
spikes, examples = studies.noise_study(S, progress)            # parallel; cached
print()
evoked = studies.evoked_spikes(spikes, N)
display(evoked.pivot(index="condition", columns="site_um", values="evoked").round(3))

contrast = studies.contrast_example(spikes, S)                 # representative input for panels a-b
fig = plotting.noise_figure(studies.psth(spikes, {**N, "psth_bin_ms": 5.0}), evoked, {**examples, "contrast": contrast},
                            {**N, "psth_bin_ms": 5.0}, example_conditions=list(studies.EXAMPLE_PAIR),
                            psth_sites=tuple(d for d in (100, 200, 300) if d in N["sites_um"]),
                            cumulative=studies.cumulative_extra(spikes, N))
show(fig, "noise/noise")                                        # Figure 5'''),

md("""## 7. (Optional) The HTML report

The cell below writes the complete report (all figures, tables, text and the draft manuscript material) from the results
computed above, to `report.html` in the figures folder. On Colab, download it from the file browser on the left."""),
code('''from pvdend import report
path = report.build(S=S)
print("Report written to", path)'''),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}, "colab": {"provenance": [], "name": "figures.ipynb"}})
nbf.write(nb, ROOT / "notebooks" / "figures.ipynb")
print("written notebooks/figures.ipynb")
