"""pvdend: EPSP integration along a short (100 um) vs grown (400 um) dendrite of a
human L2/3 PV+ interneuron model (Yao et al. 2022), with switchable dendritic Ca_LVA.

Typical use::

    from pvdend import Config, run_sweep, plotting
    res = run_sweep(Config(ca_profile="hotspot"))
"""
__version__ = "0.1.0"

from .config import CA_NORMS, CA_PROFILES, Config
from .protocols import Result, get_cell, load_or_run, run_sweep

__all__ = ["Config", "CA_PROFILES", "CA_NORMS", "Result", "run_sweep", "load_or_run", "get_cell"]
