"""Simulation protocols: an Exp2Syn input at a series of sites on the target dendrite.

Each site is simulated separately: one synapse, activated by a single event or
a regular train, while the membrane potential is recorded at the soma, at the
synapse and at the dendritic tip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from neuron import h

from . import calcium
from ._paths import RESULTS_DIR
from .cell import PVCell
from .config import Config
from .morphology import path_location

SPIKE_THRESHOLD_MV = 0.0
RHEOBASE_STEP_MS = 1000.0  # duration of the current step used to measure rheobase
ALWAYS_ON = dict(delay=-1e12, dur=1e13)  # IClamp active also during steady-state initialisation
TIP_TOL_UM = 1e-3  # sites this close beyond the tip are placed at the tip
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility
_CELLS: dict = {}


def get_cell(morph: str, cfg: Config) -> PVCell:
    """Cells are cached; channels are reconfigured from `cfg` on every run."""
    key = (morph, cfg.prune_side_branches, cfg.max_seg_len_um)
    if key not in _CELLS:
        _CELLS[key] = PVCell(morph, cfg.prune_side_branches, cfg.max_seg_len_um)
    return _CELLS[key]


def _rheobase_key(morph: str, cfg: Config) -> str:
    """Rheobase depends on the cell and its channels, not on synapse or site settings."""
    keep = ("prune_side_branches", "max_seg_len_um", "ca_profile", "ca_norm", "g_uniform", "g_peak",
            "g_total_equiv", "hotspot_center_um", "hotspot_width_um", "gradient_span_um", "cadyn_gamma",
            "cadyn_decay_ms", "somatic_ca_lva", "ttx", "celsius", "dt_ms", "v_init_mV")
    return json.dumps([morph] + [getattr(cfg, k) for k in keep])


def _spikes(v) -> int:
    v = np.asarray(v)
    return int(np.sum((v[1:] >= SPIKE_THRESHOLD_MV) & (v[:-1] < SPIKE_THRESHOLD_MV)))


def rheobase_nA(morph: str, cfg: Config, tol=0.002) -> float:
    """Smallest step current at the soma (RHEOBASE_STEP_MS long) that evokes a spike; cached on disk."""
    path = RESULTS_DIR / "rheobase.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = _rheobase_key(morph, cfg)
    if key in cache:
        return cache[key]
    cell = get_cell(morph, cfg)
    calcium.configure(cell, cfg)
    h.celsius, h.dt = cfg.celsius, cfg.dt_ms
    stim = h.IClamp(cell.soma[0](0.5))
    stim.delay, stim.dur = 50.0, RHEOBASE_STEP_MS
    v = h.Vector().record(cell.soma[0](0.5)._ref_v)

    def fires(amp):
        stim.amp = amp
        steady_state_init(cfg.v_init_mV)
        h.continuerun(stim.delay + stim.dur + 20)
        return _spikes(v) > 0

    lo, hi = 0.0, 0.5
    while not fires(hi):
        lo, hi = hi, hi * 2
        if hi > 20:
            raise RuntimeError("no spike up to 20 nA")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fires(mid) else (mid, hi)
    stim.amp = 0
    cache[key] = hi
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1))
    return hi


def bias_nA(morph: str, cfg: Config) -> float:
    return cfg.soma_bias_frac * rheobase_nA(morph, cfg) if cfg.soma_bias_frac > 0 else 0.0


def site_distances(cfg: Config, tip_um: float) -> np.ndarray:
    if cfg.sites_um is not None:
        d = np.asarray(cfg.sites_um, dtype=float)
    else:
        d = np.arange(cfg.site_start_um, tip_um + TIP_TOL_UM, cfg.site_step_um)
    return np.round(d[d <= tip_um + TIP_TOL_UM], 6)


def steady_state_init(v_init: float):
    """finitialize, then integrate to rest with large implicit steps (t < 0)."""
    h.finitialize(v_init)
    dt = h.dt
    h.t, h.dt = -1e10, 1e9
    while h.t < -1e9:
        h.fadvance()
    h.t, h.dt = 0.0, dt
    h.fcurrent()
    h.frecord_init()


def _rec(ref):
    return h.Vector().record(ref)


def run_site(cell: PVCell, cfg: Config, distance_um: float) -> dict:
    """Simulate one synapse at `distance_um`; returns traces (numpy arrays)."""
    amp = bias_nA(cell.label, cfg)  # first: may run its own (cached) rheobase simulations
    h.celsius, h.dt = cfg.celsius, cfg.dt_ms
    sec, x, actual = path_location(cell, cell.target_path, distance_um)
    syn = h.Exp2Syn(sec(x))
    syn.tau1, syn.tau2, syn.e = cfg.syn_tau1_ms, cfg.syn_tau2_ms, cfg.syn_e_mV
    stim = h.NetStim()
    stim.start, stim.number, stim.interval, stim.noise = (
        cfg.onset_ms, cfg.n_events, cfg.interval_ms, 0)
    nc = h.NetCon(stim, syn)
    nc.weight[0], nc.delay = cfg.syn_weight_uS, 0
    bias = h.IClamp(cell.soma[0](0.5))
    bias.delay, bias.dur = ALWAYS_ON["delay"], ALWAYS_ON["dur"]
    bias.amp = amp

    tip = cell.target_path[-1](1.0)
    recs = {"t": _rec(h._ref_t), "v_soma": _rec(cell.soma[0](0.5)._ref_v),
            "v_syn": _rec(sec(x)._ref_v), "v_tip": _rec(tip._ref_v),
            "g_syn": _rec(syn._ref_g)}
    has_ca = h.ismembrane("CaDynamics", sec=sec)
    if has_ca:
        recs["cai_syn"] = _rec(sec(x)._ref_cai)
        recs["ica_syn"] = _rec(sec(x)._ref_ica)
    recs["cai_soma"] = _rec(cell.soma[0](0.5)._ref_cai)

    steady_state_init(cfg.v_init_mV)
    h.continuerun(cfg.tstop_ms)

    out = {k: np.array(v) for k, v in recs.items()}
    out.update(site_um=distance_um, site_actual_um=actual, section=sec.name().split(".")[-1], x=x,
               bias_nA=bias.amp)
    bias.amp = 0
    return out


def measure(tr: dict, cfg: Config) -> dict:
    """Scalar features of one simulation."""
    t, on, isi = tr["t"], cfg.onset_ms, cfg.interval_ms
    pre = (t >= on - 5) & (t < on)
    post = t >= on
    m = {}
    for where in ("soma", "syn", "tip"):
        v = tr[f"v_{where}"]
        base = v[pre].mean()
        m[f"vrest_{where}_mV"] = base
        m[f"peak_{where}_mV"] = v[post].max() - base
    m["attenuation"] = m["peak_soma_mV"] / m["peak_syn_mV"]

    # first event at the soma (window: one inter-event interval, or t_post for a single
    # event): peak, latency, and for single events rise time 10-90 % and half-width
    v = tr["v_soma"] - m["vrest_soma_mV"]
    w = (t >= on) & (t < on + min(isi, cfg.t_post_ms))
    tw, vw = t[w], v[w]
    k = int(np.argmax(vw))
    pk = vw[k]
    m["peak1_soma_mV"] = pk
    m["latency_soma_ms"] = tw[k] - on
    if pk > 0 and cfg.n_events == 1:  # shape metrics are defined for single events only
        rise = vw[: k + 1]
        m["rise_10_90_soma_ms"] = (tw[np.argmax(rise >= 0.9 * pk)] - tw[np.argmax(rise >= 0.1 * pk)])
        above = np.nonzero(vw >= 0.5 * pk)[0]
        m["halfwidth_soma_ms"] = tw[above[-1]] - tw[above[0]]
    else:
        m["rise_10_90_soma_ms"] = m["halfwidth_soma_ms"] = np.nan

    # trains: summation = largest event-locked peak / first peak
    if cfg.n_events > 1:
        peaks = [v[(t >= on + i * isi) & (t < on + (i + 1) * isi)].max() for i in range(cfg.n_events)]
        m["summation_soma"] = max(peaks) / peaks[0] if peaks[0] > 0 else np.nan
    m["n_spikes_soma"] = _spikes(tr["v_soma"])
    up = np.nonzero((tr["v_soma"][1:] >= SPIKE_THRESHOLD_MV) & (tr["v_soma"][:-1] < SPIKE_THRESHOLD_MV))[0]
    m["first_spike_ms"] = float(t[up[0] + 1] - on) if len(up) else np.nan
    m["spikes_before_input"] = int(np.sum(t[up + 1] < on)) if len(up) else 0
    m["area_soma_mVms"] = float(_trapz(np.clip(v[post], 0, None), t[post]))
    if "cai_syn" in tr:
        c = tr["cai_syn"]
        m["dcai_syn_uM"] = (c[post].max() - c[pre].mean()) * 1e3
        m["ica_syn_min_mA_cm2"] = tr["ica_syn"][post].min()
    else:
        m["dcai_syn_uM"] = 0.0
        m["ica_syn_min_mA_cm2"] = 0.0
    return m


@dataclass
class Result:
    config: Config
    summary: pd.DataFrame
    traces: dict = field(default_factory=dict)  # (morph, site_um) -> trace dict
    ca_profiles: dict = field(default_factory=dict)  # morph -> DataFrame(distance, gbar)

    def save(self, folder: str | Path | None = None) -> Path:
        folder = Path(folder or RESULTS_DIR / self.config.name)
        folder.mkdir(parents=True, exist_ok=True)
        self.config.to_json(folder / "config.json")
        self.summary.to_csv(folder / "summary.csv", index=False)
        for morph, df in self.ca_profiles.items():
            df.to_csv(folder / f"ca_profile_{morph}.csv", index=False)
        arrays = {}
        for (morph, d), tr in self.traces.items():
            for k, v in tr.items():
                if isinstance(v, np.ndarray):
                    arrays[f"{morph}|{d:g}|{k}"] = v
        np.savez_compressed(folder / "traces.npz", **arrays)
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "Result":
        folder = Path(folder)
        cfg = Config.from_json(folder / "config.json")
        summary = pd.read_csv(folder / "summary.csv")
        traces: dict = {}
        with np.load(folder / "traces.npz") as z:
            for key in z.files:
                morph, d, k = key.split("|")
                traces.setdefault((morph, float(d)), {})[k] = z[key]
        prof = {m: pd.read_csv(folder / f"ca_profile_{m}.csv") for m in cfg.morphologies
                if (folder / f"ca_profile_{m}.csv").exists()}
        return cls(cfg, summary, traces, prof)


def run_sweep(cfg: Config, progress=None, keep_traces=True) -> Result:
    """Run every site of every morphology in `cfg`. `progress(i, n, text)` is optional."""
    jobs = []
    for morph in cfg.morphologies:
        cell = get_cell(morph, cfg)
        jobs += [(morph, d) for d in site_distances(cfg, cell.tip_distance)]

    rows, traces, profiles = [], {}, {}
    current = None
    for i, (morph, d) in enumerate(jobs):
        if morph != current:
            cell = get_cell(morph, cfg)
            g = calcium.configure(cell, cfg)
            profiles[morph] = pd.DataFrame({
                "distance_um": [dd for _, dd in cell.path_segments()],
                "gbar_Ca_LVA_S_cm2": g,
                "area_um2": [seg.area() for seg, _ in cell.path_segments()]})
            current = morph
        if progress:
            progress(i, len(jobs), f"{morph}: {d:g} um")
        tr = run_site(cell, cfg, d)
        rows.append({"morphology": morph, "site_um": d, "site_actual_um": tr["site_actual_um"],
                     "section": tr["section"], "bias_nA": tr["bias_nA"], **measure(tr, cfg)})
        if keep_traces:
            traces[(morph, float(d))] = tr
    if progress:
        progress(len(jobs), len(jobs), "done")
    return Result(cfg, pd.DataFrame(rows), traces, profiles)


def load_or_run(cfg: Config, folder=None, force=False, progress=None) -> Result:
    """Reuse saved results if the stored config is identical, otherwise run and save."""
    folder = Path(folder or RESULTS_DIR / cfg.name)
    if not force and (folder / "config.json").exists():
        if Config.from_json(folder / "config.json") == cfg:  # missing new fields take their defaults
            return Result.load(folder)
    res = run_sweep(cfg, progress=progress)
    res.save(folder)
    return res
