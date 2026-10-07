"""Analyses that explain where along the dendrite a Ca_LVA-dependent regenerative event appears.

1. Ca_LVA gating (steady states and time constants, from mod/Ca_LVA.mod).
2. Passive electrotonic profile of the target dendrite (input and transfer impedance).
3. Shape of the passive local EPSP (amplitude and duration) along the dendrite.
4. Threshold synaptic weight for the regenerative event, as a function of distance.
5. Gating of Ca_LVA during an event at one site.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from neuron import h

from . import calcium
from ._paths import RESULTS_DIR
from .config import Config
from .morphology import path_location
from .protocols import get_cell, run_site, steady_state_init

EVENT_CRITERION_MV = 10.0  # extra local depolarisation due to Ca_LVA that counts as an event


def ca_lva_rates(v, celsius=34.0):
    """mInf, mTau, hInf, hTau of Ca_LVA (numpy copy of PROCEDURE rates in mod/Ca_LVA.mod)."""
    qt = 2.3 ** ((celsius - 21) / 10)
    v = np.asarray(v, dtype=float) + 10
    m_inf = 1 / (1 + np.exp((v + 30) / -6))
    m_tau = (5 + 20 / (1 + np.exp((v + 25) / 5))) / qt
    h_inf = 1 / (1 + np.exp((v + 80) / 6.4))
    h_tau = (20 + 50 / (1 + np.exp((v + 40) / 7))) / qt
    return m_inf, m_tau, h_inf, h_tau


def impedance_profile(morph: str, cfg: Config, freqs=(0.0, 100.0)) -> pd.DataFrame:
    """|Z_in| at each target-path segment and |Z_transfer| to the soma (MOhm), at rest, no dendritic Ca."""
    cfg = cfg.replace(ca_profile="none")
    cell = get_cell(morph, cfg)
    calcium.configure(cell, cfg)
    h.celsius, h.dt = cfg.celsius, cfg.dt_ms
    steady_state_init(cfg.v_init_mV)
    imp = h.Impedance()
    imp.loc(0.5, sec=cell.soma[0])
    rows = {"distance_um": [d for _, d in cell.path_segments()]}
    for f in freqs:
        imp.compute(f, 1)  # 1: include linearised active conductances (Ih, somatic channels)
        rows[f"zin_{f:g}Hz_MOhm"] = [imp.input(seg.x, sec=seg.sec) for seg, _ in cell.path_segments()]
        rows[f"ztr_{f:g}Hz_MOhm"] = [imp.transfer(seg.x, sec=seg.sec) for seg, _ in cell.path_segments()]
    rows["zin_soma_0Hz_MOhm"] = imp.input(0.5, sec=cell.soma[0]) if 0.0 in freqs else np.nan
    return pd.DataFrame(rows)


def length_constant_um(cfg: Config, morph="long") -> float:
    """DC length constant (leak only) of the grown, constant-diameter part of the dendrite."""
    cell = get_cell(morph, cfg)
    sec = cell.target_path[-1]
    r_mem = 1 / sec(0.5).g_pas  # Ohm cm2
    return 1e4 * np.sqrt(r_mem * sec.diam * 1e-4 / (4 * sec.Ra))  # cm -> um


def local_epsp_shape(result, morph: str, threshold_mV=-50.0) -> pd.DataFrame:
    """Peak (absolute), half-width and time above `threshold_mV` of the local EPSP."""
    rows = []
    for (m, d), tr in sorted(result.traces.items()):
        if m != morph:
            continue
        t, v = tr["t"], tr["v_syn"]
        base = v[t < result.config.onset_ms].mean()
        dv = v - base
        pk = dv.max()
        above = np.nonzero(dv >= pk / 2)[0]
        rows.append(dict(distance_um=d, peak_abs_mV=v.max(),
                         halfwidth_ms=t[above[-1]] - t[above[0]],
                         t_above_ms=float(np.sum(v > threshold_mV) * result.config.dt_ms)))
    return pd.DataFrame(rows)


def threshold_scan(cfg: Config, sites_um, weights_nS, profile="uniform", folder=None) -> pd.DataFrame:
    """Local peak with and without dendritic Ca_LVA for every (site, weight); cached as CSV."""
    folder = Path(folder or RESULTS_DIR / "mechanism")
    tag = f"scan_{profile}_{cfg.ca_norm}_w{len(weights_nS)}_s{len(sites_um)}.csv"
    path = folder / tag
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for prof in ("none", profile):
        for w in weights_nS:
            c = cfg.replace(ca_profile=prof, syn_weight_uS=w / 1e3, morphologies=["long"],
                            sites_um=list(sites_um), n_events=1)
            cell = get_cell("long", c)
            calcium.configure(cell, c)
            for d in sites_um:
                tr = run_site(cell, c, d)
                pre = tr["t"] < c.onset_ms
                rows.append(dict(profile=prof, weight_nS=w, distance_um=d,
                                 peak_syn_mV=(tr["v_syn"] - tr["v_syn"][pre].mean()).max(),
                                 peak_soma_mV=(tr["v_soma"] - tr["v_soma"][pre].mean()).max()))
    df = pd.DataFrame(rows)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def thresholds(scan: pd.DataFrame, profile="uniform") -> pd.DataFrame:
    """Smallest weight at which Ca_LVA adds > EVENT_CRITERION_MV to the local peak."""
    a = scan[scan.profile == profile].set_index(["distance_um", "weight_nS"]).peak_syn_mV
    b = scan[scan.profile == "none"].set_index(["distance_um", "weight_nS"]).peak_syn_mV
    extra = (a - b).reset_index(name="extra_mV")
    out = []
    for d, grp in extra.groupby("distance_um"):
        hit = grp[grp.extra_mV > EVENT_CRITERION_MV].weight_nS
        out.append(dict(distance_um=d, threshold_nS=hit.min() if len(hit) else np.nan))
    return pd.DataFrame(out)


def gating_traces(cfg: Config, site_um: float) -> dict:
    """Local V, I_Ca and Ca_LVA gates (m, h) at `site_um`, with and without dendritic Ca_LVA."""
    out = {}
    for prof in ("none", cfg.ca_profile):
        c = cfg.replace(ca_profile=prof, morphologies=["long"], sites_um=[site_um])
        cell = get_cell("long", c)
        calcium.configure(cell, c)
        sec, x, _ = path_location(cell, cell.target_path, site_um)
        extra = {}
        if prof != "none":
            for name in ("m", "h"):
                extra[name] = h.Vector().record(getattr(sec(x), f"_ref_{name}_Ca_LVA"))
        tr = run_site(cell, c, site_um)  # finitialize inside also starts the extra recordings
        tr.update({k: np.array(v) for k, v in extra.items()})
        out[prof] = tr
    return out


def event_extra(cfg: Config, sites_um, tag: str, folder=None) -> pd.DataFrame:
    """Extra local depolarisation due to dendritic Ca_LVA at each site (long cell); cached."""
    path = Path(folder or RESULTS_DIR / "mechanism") / f"extra_{tag}.csv"
    if path.exists():
        return pd.read_csv(path)
    peaks = {}
    for prof in ("none", cfg.ca_profile):
        c = cfg.replace(ca_profile=prof, morphologies=["long"], n_events=1)
        cell = get_cell("long", c)
        calcium.configure(cell, c)
        for d in sites_um:
            tr = run_site(cell, c, d)
            pre = tr["t"] < c.onset_ms
            peaks[(prof, d)] = ((tr["v_syn"] - tr["v_syn"][pre].mean()).max(),
                                (tr["v_soma"] - tr["v_soma"][pre].mean()).max())
    df = pd.DataFrame([dict(distance_um=d,
                            extra_local_mV=peaks[(cfg.ca_profile, d)][0] - peaks[("none", d)][0],
                            extra_soma_mV=peaks[(cfg.ca_profile, d)][1] - peaks[("none", d)][1])
                       for d in sites_um])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


SCAN_SITES = (40, 80, 120, 140, 160, 180, 200, 250, 300, 350, 400)
SCAN_WEIGHTS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40)
TAU_SITES = tuple(range(20, 401, 20))
TAU2_VALUES = (3.0, 6.0, 10.0, 20.0)
EXAMPLE_SITE = 250.0


def collect(cfg: Config | None = None) -> dict:
    """Everything the mechanism figure and text need (uniform Ca_LVA, 'peak' densities)."""
    cfg = (cfg or Config()).replace(ca_profile="uniform", ca_norm="peak")
    from .protocols import load_or_run
    passive = load_or_run(cfg.replace(name="mechanism_passive", ca_profile="none", n_events=1))
    v = np.linspace(-100, 20, 241)
    m_inf, m_tau, h_inf, h_tau = ca_lva_rates(v, cfg.celsius)
    rest = float(passive.summary.vrest_syn_mV.mean())
    scan = threshold_scan(cfg, SCAN_SITES, SCAN_WEIGHTS)
    return dict(
        cfg=cfg, rest_mV=rest,
        gating=pd.DataFrame(dict(v=v, m_inf=m_inf, m_tau=m_tau, h_inf=h_inf, h_tau=h_tau)),
        h_rest=float(ca_lva_rates(rest, cfg.celsius)[2]),
        impedance={m: impedance_profile(m, cfg) for m in ("short", "long")},
        lam_um=length_constant_um(cfg),
        epsp={m: local_epsp_shape(passive, m) for m in ("short", "long")},
        scan=scan, thresholds=thresholds(scan),
        tau={t: event_extra(cfg.replace(syn_tau2_ms=t), TAU_SITES, f"uniform_tau2_{t:g}")
             for t in TAU2_VALUES},
        profiles={p: event_extra(cfg.replace(ca_profile=p), TAU_SITES, f"{p}_tau2_3")
                  for p in ("uniform", "hotspot", "increasing", "decreasing")},
        example=gating_traces(cfg, EXAMPLE_SITE), example_site=EXAMPLE_SITE,
    )


def onset_um(extra: pd.DataFrame, criterion=EVENT_CRITERION_MV) -> float:
    hit = extra[extra.extra_local_mV > criterion].distance_um
    return float(hit.min()) if len(hit) else np.nan
