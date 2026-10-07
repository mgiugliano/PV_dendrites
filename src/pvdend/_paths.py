"""Repository paths and one-time loading of the compiled NEURON mechanisms."""
import os
from pathlib import Path

from neuron import h, load_mechanisms

REPO_ROOT = Path(os.environ.get("PVDEND_ROOT", Path(__file__).resolve().parents[2]))
MOD_DIR = REPO_ROOT / "mod"
MORPH_DIR = REPO_ROOT / "morphologies"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CONFIG_DIR = REPO_ROOT / "configs"
TARGET_META = MORPH_DIR / "target_dendrite.json"


def ensure_mechanisms():
    """Load the mechanisms compiled by `nrnivmodl mod` in the repository root."""
    if not hasattr(h, "Ca_LVA"):  # NEURON auto-loads them only when run from REPO_ROOT
        load_mechanisms(str(REPO_ROOT), warn_if_already_loaded=False)
    if not hasattr(h, "Ca_LVA"):
        raise RuntimeError(
            f"NEURON mechanisms not found. Run `nrnivmodl mod` inside {REPO_ROOT}."
        )
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
