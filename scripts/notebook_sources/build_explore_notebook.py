"""Generate notebooks/explore.ipynb (the interactive explorer notebook).

Run:  python scripts/notebook_sources/build_explore_notebook.py
"""
from pathlib import Path

import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
BADGE = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgiugliano/PV_dendrites/blob/main/notebooks/explore.ipynb)"
cells = [
md(f"""# EPSPs along a short vs a grown dendrite of a human PV+ interneuron

{BADGE}

This notebook explores how an excitatory synapse placed along one dendrite of a
human layer 2/3 parvalbumin-positive (PV+) interneuron model is seen at the soma,
and how low-voltage-activated Ca²⁺ channels (Ca$_{{LVA}}$) in that dendrite change it.

* **Short morphology**: the original reconstruction `HL23PV.swc`. The target dendrite ends 100.8 µm from the soma and carries no Ca channels.
* **Long morphology**: the same cell, with the tip of the target dendrite grown (random walk, constant tip diameter) until it is 400 µm from the soma. Ca$_{{LVA}}$ can be placed along this dendrite.

The cell model (biophysics, ion channels, morphology) is that of Yao *et al.* (2022), *Cell Reports* 38, 110232, distributed on Zenodo (doi:[10.5281/zenodo.5771000](https://doi.org/10.5281/zenodo.5771000)).

Every button below calls the same Python functions as the command-line scripts in `scripts/`,
so whatever you explore here can be saved as a config file and reproduced exactly."""),
md("## 1. Setup\nOn Google Colab this cell downloads the repository, installs NEURON and compiles the ion channels (about 1–2 minutes the first time). Locally it only compiles the channels if needed."),
code('''import os, shutil, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/mgiugliano/PV_dendrites"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    ROOT = Path("/content/PV_dendrites")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuron"], check=True)
    from google.colab import output
    output.enable_custom_widget_manager()
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

def compiled():
    return any((ROOT / a / lib).exists() for a in ("x86_64", "arm64", "aarch64")
               for lib in (".libs/libnrnmech.so", "libnrnmech.so", "libnrnmech.dylib"))

if not compiled():
    # use the nrnivmodl of the NEURON installed for *this* Python, not another one on PATH
    local = Path(sys.executable).parent / "nrnivmodl"
    nrnivmodl = str(local) if local.exists() else shutil.which("nrnivmodl")
    r = subprocess.run([nrnivmodl, "mod"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not compiled():
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("Compiling the NEURON mechanisms (mod/) failed, see output above.")

os.environ["PVDEND_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src"))
print("Repository:", ROOT, "| Colab:", IN_COLAB)'''),
code('''%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display

from pvdend import Config, get_cell, run_sweep, plotting, gui
plotting.set_style()'''),
md("""## 2. The two morphologies

**Dendrograms**: every branch is drawn at its path distance from the soma, so the 100 µm and 400 µm versions of the target dendrite can be compared directly (the tick marks the original tip). **3D view**: drag to rotate, scroll to zoom; click *grown extension* in the legend to hide it and see the short cell."""),
code('''from pvdend import viewer3d
cfg = Config()
cells = {m: get_cell(m, cfg) for m in ("short", "long")}
plotting.dendrogram_figure(cells)
plt.show()
viewer3d.morphology_3d(cells["long"], cells["short"].tip_distance).show()'''),
md("""## 3. Interactive explorer

* **Ca channels**: the dendritic Ca$_{LVA}$ profile on the long dendrite (`none`, `uniform`, `increasing`), its density *g* in mS/cm² (uniform: g everywhere; increasing: 0 at the soma, g at the midpoint, 2g at the tip), and an optional shift of its activation curve (e.g. −15 mV: half-activation at −55 instead of −40 mV).
* **Switches**: somatic Ca$_{LVA}$ on/off, TTX, a steady somatic bias current (fraction of rheobase), which morphologies to simulate, the diameter of the dendrite's "mouth" at the soma, and optional pruning of the side branches leaving the target dendrite.
* **Synapse**: Exp2Syn conductance, single event (`Events` = 1) or a regular train.
* **Buttons**: *Run sweep* simulates every site of the chosen spacing. *Compare all profiles* runs the five profiles with the current settings. *Run single site* shows one location in detail. *Export figure* writes PDF/SVG/PNG to `figures/` (and downloads the PDF on Colab). *Save config* writes a JSON file that `scripts/run.py` reproduces exactly.

A sweep takes about 0.3 s per site and morphology: about 10 s at 20 µm spacing, and about 20 s at 10 µm."""),
code("ex = gui.explorer()"),
md("""## 4. The same thing without widgets

The *Run sweep* button runs exactly this. Change any field of `Config` (see `src/pvdend/config.py` for the full list)."""),
code('''cfg = Config(name="example", ca_profile="increasing", g_ca_mS_cm2=2.5, ca_act_shift_mV=-15, site_step_um=20)
res = run_sweep(cfg)
display(res.summary[["morphology", "site_um", "peak_soma_mV", "peak_syn_mV", "attenuation", "dcai_syn_uM"]].head(8).round(3))
fig = plotting.overview_figure(res, {m: get_cell(m, cfg) for m in cfg.morphologies})
plt.show()'''),
md("""## 5. Publication figures

The figures in `figures/` are produced by `python scripts/make_all_figures.py`, which runs the studies defined in `configs/studies.json`
and uses the same plotting functions as this notebook. Saved results in `results/` are reused when the configuration has not changed.
`python scripts/make_report.py` then collects all of them, with captions and summary tables, in one page: `figures/report.html`.
Running the full set takes a few minutes; uncomment the lines below to do it from here."""),
code('''# subprocess.run([sys.executable, "scripts/make_all_figures.py"], cwd=ROOT, check=True)
# subprocess.run([sys.executable, "scripts/make_report.py"], cwd=ROOT, check=True)'''),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "name": "explore.ipynb"}})
nbf.write(nb, Path(__file__).resolve().parents[2] / "notebooks" / "explore.ipynb")
print("written")
