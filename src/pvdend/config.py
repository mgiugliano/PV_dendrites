"""All parameters of one experiment, in one place.

A `Config` fully specifies a simulation sweep. The command-line scripts read it
from JSON, the notebook builds it from widgets; both then call the same code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CA_PROFILES = ("none", "uniform", "increasing")
MORPHOLOGIES = ("short", "long")

# Somatic Ca_LVA density and CaDynamics parameters of the original model
# (models/biophys_HL23PV.hoc, Yao et al. 2022).
SOMA_GBAR_CA_LVA = 0.09250008555398015
SOMA_CADYN_GAMMA = 0.0005
SOMA_CADYN_DECAY = 531.0255920416845


@dataclass
class Config:
    name: str = "default"
    morphologies: list = field(default_factory=lambda: ["short", "long"])

    # --- morphology ---------------------------------------------------------
    prune_side_branches: bool = False  # remove branches leaving the target path
    max_seg_len_um: float = 2.0  # spatial resolution along the target path
    mouth_scale: float = 1.0  # diameter factor at the soma end of the target dendrite (1: original)
    mouth_length_um: float = 13.0  # extra width fades linearly to 1x over this distance from the soma

    # --- dendritic Ca_LVA (target path of the 'long' morphology only) ---------
    ca_profile: str = "none"  # one of CA_PROFILES
    g_ca_mS_cm2: float = 1.0  # uniform density, and density of 'increasing' at the midpoint span/2
    gradient_span_um: float = 400.0  # 'increasing' goes from 0 at the soma to 2 g at span
    ca_act_shift_mV: float = 0.0  # shift of the dendritic Ca_LVA activation gate (negative: activates earlier)
    cadyn_gamma: float = SOMA_CADYN_GAMMA
    cadyn_decay_ms: float = SOMA_CADYN_DECAY

    # --- pharmacology / somatic switches (both morphologies) -----------------
    somatic_ca_lva: bool = True  # False: somatic Ca_LVA gbar = 0
    ttx: bool = False  # True: NaTg and Nap gbar = 0 everywhere

    # --- steady somatic bias current ----------------------------------------
    soma_bias_frac: float = 0.0  # DC at the soma, as a fraction of the cell's rheobase (0: none)

    # --- excitatory synapse (Exp2Syn conductance) ----------------------------
    syn_tau1_ms: float = 0.3
    syn_tau2_ms: float = 3.0
    syn_e_mV: float = 0.0
    syn_weight_uS: float = 0.005
    n_events: int = 1
    freq_hz: float = 20.0
    onset_ms: float = 50.0

    # --- input sites (absolute path distance from the soma centre) -----------
    site_start_um: float = 10.0
    site_step_um: float = 10.0
    sites_um: list | None = None  # explicit list overrides start/step

    # --- simulation ----------------------------------------------------------
    celsius: float = 34.0
    dt_ms: float = 0.025
    v_init_mV: float = -80.0
    t_post_ms: float = 150.0  # simulated time after the last event

    def __post_init__(self):
        if self.ca_profile not in CA_PROFILES:
            raise ValueError(f"ca_profile must be one of {CA_PROFILES}")
        if self.mouth_scale <= 0 or self.mouth_length_um <= 0:
            raise ValueError("mouth_scale and mouth_length_um must be positive")
        for m in self.morphologies:
            if m not in MORPHOLOGIES:
                raise ValueError(f"morphologies must be a subset of {MORPHOLOGIES}")
        if not 0 <= self.soma_bias_frac < 1:
            raise ValueError("soma_bias_frac must be in [0, 1): the cell must stay below rheobase")
        if self.soma_bias_frac > 0 and self.ttx:
            raise ValueError("a bias relative to rheobase is undefined with TTX (the cell cannot fire)")
        if self.n_events < 1 or self.freq_hz <= 0:
            raise ValueError("n_events must be >= 1 and freq_hz > 0")

    @property
    def interval_ms(self) -> float:
        return 1000.0 / self.freq_hz

    @property
    def tstop_ms(self) -> float:
        return self.onset_ms + (self.n_events - 1) * self.interval_ms + self.t_post_ms

    def replace(self, **changes) -> "Config":
        return Config(**{**asdict(self), **changes})

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(asdict(self), indent=2)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))
