# PV_dendrites

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgiugliano/PV_dendrites/blob/main/notebooks/figures.ipynb) **Every figure, step by step** (`notebooks/figures.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgiugliano/PV_dendrites/blob/main/notebooks/explore.ipynb) **Interactive explorer** (`notebooks/explore.ipynb`)

Simulations of a human layer 2/3 parvalbumin-positive (PV+) interneuron model. They
compare how an excitatory synapse along one dendrite is felt at the soma when that
dendrite is short (100 µm, as reconstructed) or grown to 400 µm. In the long dendrite,
low-voltage-activated Ca²⁺ channels (Ca<sub>LVA</sub>) can be switched on with
different spatial distributions, densities and activation ranges.

Two notebooks run in the browser on Google Colab, with nothing to install:

* **`figures.ipynb`** regenerates every figure of the report in order, each preceded by an explanation of
  the methods with equations. Run it top to bottom. With `QUICK = True` (default) it takes about 40
  minutes on a laptop (1–1.5 hours on Colab); with `QUICK = False` it reproduces the report figures exactly (several hours).
* **`explore.ipynb`** has widgets (dropdowns, sliders, buttons) to run any condition interactively.

## The model

The cell model is the human L2/3 PV+ interneuron (HL23PV) of

> Yao HK, Guet-McCreight A, Mazza F, Moradi Chameh H, Prevot TD, Griffiths JD, Tripathy SJ,
> Valiante TA, Sibille E, Hay E (2022). Reduced inhibition in depression impairs stimulus
> processing in human cortical microcircuits. *Cell Reports* 38(2):110232.
> doi:[10.1016/j.celrep.2021.110232](https://doi.org/10.1016/j.celrep.2021.110232)

The model code and data come from the authors' release: *Human Cortical Layer 2/3 Microcircuits in Health and
Depression*, Zenodo, doi:[10.5281/zenodo.5771000](https://doi.org/10.5281/zenodo.5771000).
The ion channel files (`mod/`), the morphology (`morphologies/HL23PV.swc`) and the original HOC
code (`original_model/`) are copied unchanged from that release. `src/pvdend/cell.py` is a line-by-line
Python translation of `NeuronTemplate.hoc` and `biophys_HL23PV.hoc`. The tests check every
parameter against the original HOC file.

## What is simulated

| | Short morphology | Long morphology |
|---|---|---|
| File | `HL23PV.swc` (original) | `HL23PV_dend400.swc` (generated) |
| Target dendrite | `dend[27] → dend[29] → dend[30]` | same, with the tip grown |
| Tip distance from soma | 100.8 µm | 400.0 µm |
| Dendritic Ca<sub>LVA</sub> | never | selectable profile (below) |

* **Target dendrite**: the terminal path of the original cell whose tip is closest to
  100 µm from the soma. The path itself has no branch points. Two side branches leave it
  (`dend[28]`, tip at 31 µm, and `dend[31]`, tip at 186 µm). Both cells keep them
  unless `prune_side_branches` is on.
* **Growth**: the tip is extended by 299.19 µm with a random walk. Each step is 5 µm; at each
  step the direction is perturbed by Gaussian noise (s.d. 0.1) with seed 42. The
  diameter stays at the tip value of 0.43 µm. This is the method used earlier for the 1200 µm extension. See
  `scripts/make_morphology.py` and `morphologies/target_dendrite.json`.
* **Distances** are path distances from the centre of the soma, `soma[0](0.5)`.
* **Dendritic Ca<sub>LVA</sub>** (with `CaDynamics`, using the somatic γ and decay values) is placed only on the
  target path of the long cell, from 0 to 400 µm, with density g = `g_ca_mS_cm2`:

  | `ca_profile` | g(d) |
  |---|---|
  | `none` | 0 (passive dendrite, as in the original model) |
  | `uniform` | g |
  | `increasing` | 2g · d / `gradient_span_um` (0 at the soma, g at the midpoint, 2g at 400 µm) |

  The two profiles carry about the same total conductance. The dendrite uses `mod/Ca_LVA_dend.mod`, a copy of
  the model's `Ca_LVA.mod` whose activation gate can be shifted (`ca_act_shift_mV`; −15 mV moves the
  half-activation of m from −40 to −55 mV, inactivation unchanged). With no shift it is identical to the original.
* **Wider mouth**: `mouth_scale` multiplies the diameter of the target dendrite at the soma, fading linearly back
  to ×1 at `mouth_length_um` (13 µm, the second branch point).
* **Switches** (both cells): `somatic_ca_lva` (somatic Ca<sub>LVA</sub> on/off), `ttx` (NaTg and Nap set to 0
  everywhere), and `soma_bias_frac` (steady somatic current as a fraction of the cell's rheobase).
* **Input**: one `Exp2Syn` conductance synapse (τ<sub>rise</sub> 0.3 ms, τ<sub>decay</sub> 3 ms, E 0 mV,
  5 nS). It is activated by a single event or a regular train (`n_events`, `freq_hz`). The synapse is placed in turn at
  every `site_step_um` (10 µm by default) along the target dendrite, at the same absolute
  distances in both cells (10–100 µm short, 10–400 µm long), one simulation per site.
* **Noisy current**: `studies.noise_study` drives the soma with a mean current plus Ornstein–Uhlenbeck noise
  (fractions of rheobase) and measures the spike probability each synapse adds, with paired noise realisations.
* **Recorded**: membrane potential at the soma, at the synapse and at the tip; local [Ca²⁺]ᵢ and I<sub>Ca</sub>.
  Measured: somatic and local EPSP amplitude, attenuation, rise time, half-width, latency, EPSP integral,
  effective and decay time constants, train summation, somatic spikes, and local Δ[Ca²⁺]ᵢ.
* **Numerics**: 34 °C, dt 0.025 ms. The cell is brought to its resting state before each run.
  Segments are at most 2 µm long along the target dendrite (`max_seg_len_um`); everywhere else they follow
  the original rule, `nseg = 1 + 2·int(L/40)`.

## Running it

### In the browser (Google Colab)

Click the badge at the top. Run the cells in order. The first cell installs NEURON and compiles
the channels, which takes about 1–2 minutes. Then use the widgets.

### On your computer

You need Python ≥ 3.10 and a C compiler (on macOS, `xcode-select --install`).

```bash
git clone https://github.com/mgiugliano/PV_dendrites
cd PV_dendrites
uv sync --extra dev            # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/nrnivmodl mod        # compile the ion channels (once)
```

Interactive notebook (JupyterLab or VS Code with the Jupyter extension):

```bash
.venv/bin/jupyter lab notebooks/explore.ipynb
```

Command line:

```bash
.venv/bin/python scripts/run.py configs/default.json                                # one sweep + overview figure
.venv/bin/python scripts/run.py configs/default.json --set ca_profile=increasing g_ca_mS_cm2=2.5 ca_act_shift_mV=-15 name=incr_shift
.venv/bin/python scripts/make_all_figures.py                                       # every study in configs/studies.json
.venv/bin/python scripts/make_report.py                                            # all of them in one page, with explanations: figures/report.html
.venv/bin/python -m pytest                                                         # tests
```

The notebook's **Save config** button writes a JSON file to `configs/`, and
`scripts/run.py` reproduces it exactly. The notebook and the scripts call the same
functions, so a figure explored interactively and the one in a paper come from the same code.

### From Python

```python
from pvdend import Config, run_sweep, get_cell, plotting

cfg = Config(name="demo", ca_profile="increasing", g_ca_mS_cm2=2.5, ca_act_shift_mV=-15, n_events=5, freq_hz=50)
res = run_sweep(cfg)            # res.summary: pandas DataFrame, res.traces: numpy arrays
res.save()                      # results/demo/{config.json, summary.csv, traces.npz, ca_profile_*.csv}
fig = plotting.overview_figure(res, {m: get_cell(m, cfg) for m in cfg.morphologies})
plotting.save_figure(fig, "demo")   # figures/demo.{pdf,svg,png}
```

All parameters are fields of `Config` in `src/pvdend/config.py`, each with its default and a comment.

## Repository layout

```
mod/                    NEURON ion channels (unchanged, from Zenodo)
morphologies/           HL23PV.swc (unchanged), HL23PV_dend400.swc (generated), target_dendrite.json
original_model/         original HOC template and biophysics, for reference and tests
src/pvdend/
  config.py             Config: every parameter of an experiment
  cell.py               PVCell: Python translation of the HOC model
  morphology.py         target dendrite, path distances, tip growth
  calcium.py            dendritic Ca_LVA profiles, somatic Ca_LVA switch, TTX
  studies.py            the studies of the report (factorial grid, trains, bias, noise)
  protocols.py          synaptic sweep, recordings, measurements, saving/loading
  plotting.py           publication figures (used by scripts and notebook)
  viewer3d.py           interactive 3D view (plotly): rotate, zoom, toggle the grown extension
  mechanism.py          analyses explaining where the Ca_LVA event appears (impedance, EPSP duration, thresholds)
  report.py             single-page HTML report of the whole figure set, with the mechanism section
  manuscript.py         draft Methods, Results, figure legends and references (numbers filled in from the data)
  gui.py                ipywidgets explorer for the notebook
scripts/                make_morphology.py, run.py, make_all_figures.py, make_report.py
configs/                default.json, studies.json (+ configs saved from the notebook)
notebooks/figures.ipynb step-by-step notebook that regenerates every figure (local and Colab)
notebooks/explore.ipynb interactive notebook with widgets (local and Colab)
scripts/notebook_sources/  the scripts that generate the two notebooks
figures/                generated figures and report.html
tests/                  pytest suite
```

## License

BSD 3-Clause (see `LICENSE`). The files from the original Zenodo release keep the terms of
that release. If you use this code, please cite Yao *et al.* (2022) and the Zenodo record above.
