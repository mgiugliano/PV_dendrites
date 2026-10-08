"""The studies of the report, defined in configs/studies.json.

A *condition* is one combination of the factors (Ca_LVA activation shift, density, mouth
scale). For each condition, the long cell is simulated with every Ca_LVA profile, and the
long cell without dendritic Ca_LVA and the short cell provide the references ('none').
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import calcium
from ._paths import CONFIG_DIR, RESULTS_DIR
from .config import Config
from .protocols import get_cell, load_or_run, rheobase_nA, run_noise_trial

KIN_LABEL = {0.0: "original kinetics", -15.0: "activation shifted by −15 mV"}


def spec(set_file=None) -> dict:
    return json.loads((set_file or CONFIG_DIR / "studies.json").read_text())


def base_config(S=None) -> Config:
    S = S or spec()
    return Config.from_json(CONFIG_DIR / S["base"])


def _name(kind, prof, shift=0.0, g=1.0, mouth=1.0, frac=0.0):
    if prof == "none":  # independent of Ca_LVA settings
        tag = f"m{mouth:g}"
    else:
        tag = f"g{g:g}_m{mouth:g}" + ("" if shift == 0 else f"_s{shift:g}")
    if frac:
        tag += f"_b{frac:g}"
    return f"{kind}_{tag}__{prof}"


def _cfg(base, kind_overrides, kind, prof, shift, g, mouth, frac=0.0):
    c = base.replace(**kind_overrides, ca_profile=prof, mouth_scale=mouth, soma_bias_frac=frac)
    if prof != "none":
        c = c.replace(ca_act_shift_mV=shift, g_ca_mS_cm2=g)
    return c.replace(name=_name(kind, prof, shift, g, mouth, frac))


def grid(kind: str, S=None, progress=None) -> dict:
    """{(shift, g, mouth): {'none': Result, profile: Result, ...}} for kind 'single' or 'train'."""
    S = S or spec()
    F, K = S["factors"], S[kind]
    base = base_config(S)
    out = {}
    for shift, g, mouth in itertools.product(F["ca_act_shift_mV"], K.get("g_ca_mS_cm2", F["g_ca_mS_cm2"]),
                                             K.get("mouth_scale", F["mouth_scale"])):
        res = {}
        for prof in ["none"] + F["profiles"]:
            c = _cfg(base, K["overrides"], kind, prof, shift, g, mouth)
            if progress:
                progress(c.name)
            res[prof] = load_or_run(c)
        out[(shift, g, mouth)] = res
    return out


def bias_grid(S=None, progress=None) -> dict:
    """{(shift, g): {fraction: {profile: Result}}}; fraction 0 is the single-event grid."""
    S = S or spec()
    F, B = S["factors"], S["bias"]
    base = base_config(S)
    out = {}
    for shift, g in itertools.product(F["ca_act_shift_mV"], B["g_ca_mS_cm2"]):
        out[(shift, g)] = {}
        for frac in B["fractions"]:
            kind = "single"
            res = {}
            for prof in ["none"] + F["profiles"]:
                c = _cfg(base, S["single"]["overrides"], kind, prof, shift, g, B["mouth_scale"], frac)
                if progress:
                    progress(c.name)
                res[prof] = load_or_run(c)
            out[(shift, g)][frac] = res
    return out


def noise_conditions(S=None) -> list:
    """(label, morphology, Config) for the noise study."""
    S = S or spec()
    F, N = S["factors"], S["noise"]
    base = base_config(S)
    conds = [("short", "short", _cfg(base, {}, "noise", "none", 0, 1, 1.0)),
             ("long, no Ca_LVA", "long", _cfg(base, {}, "noise", "none", 0, 1, 1.0))]
    for shift in F["ca_act_shift_mV"]:
        for prof in F["profiles"]:
            label = f"long, {prof}, {'shifted' if shift else 'original'}"
            conds.append((label, "long", _cfg(base, {}, "noise", prof, shift, N["g_ca_mS_cm2"], 1.0)))
    return conds


def _noise_block(job):
    """Worker: one long noisy simulation (one block); returns spike times. Runs in a separate process."""
    label, morph, cfg_json, site, block, N = job
    from .config import Config as _C
    cfg = _C.from_dict(json.loads(cfg_json))
    cell = get_cell(morph, cfg)
    calcium.configure(cell, cfg)
    rb = rheobase_nA(morph, cfg)
    on, k, period = N["warmup_ms"], N["inputs_per_block"], N["period_ms"]
    c = cfg.replace(n_events=k, freq_hz=1000.0 / period)
    keep = block == 0 and site in (None, N["example_site_um"])
    r = run_noise_trial(cell, c, site, 1000 + block, N["mu_frac"] * rb, N["sigma_frac"] * rb, N["tau_ms"],
                        on, on + k * period, keep)
    out = dict(label=label, morph=morph, site=site, block=block, spikes=r["spikes"])
    if keep:  # first two inputs only
        sel = r["t"] < on + 2 * period
        out.update(t=r["t"][sel], v=r["v_soma"][sel])
    return out


def noise_study(S=None, progress=None):
    """Paired noisy-current blocks, run in parallel; cached in results/noise_psth/.

    Returns (spikes, examples): spikes[(label, site)] = list of spike-time arrays relative to each input
    (site None = no synapse, same noise); examples[label] = traces of the first block.
    """
    import concurrent.futures as cf
    import multiprocessing as mp
    S = S or spec()
    N = S["noise"]
    folder = RESULTS_DIR / "noise_psth"
    folder.mkdir(parents=True, exist_ok=True)
    key = json.dumps(N, sort_keys=True)
    jobs, cached = [], {}
    for label, morph, cfg in noise_conditions(S):
        cell_tip = get_cell(morph, cfg).tip_distance
        for site in [None] + [d for d in N["sites_um"] if d <= cell_tip + 1e-3]:
            stem = f"{cfg.name}_{morph}_{'nosyn' if site is None else f'{site:g}'}"
            path, meta = folder / f"{stem}.npz", folder / f"{stem}.json"
            if path.exists() and meta.exists() and meta.read_text() == key + cfg.to_json():
                cached[(label, site)] = path
            else:
                jobs += [(label, morph, cfg.to_json(), site, b, N) for b in range(N["n_blocks"])]
                cached[(label, site)] = (path, meta, key + cfg.to_json())
    results = {}
    if jobs:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=N["workers"], mp_context=ctx) as ex:
            for i, res in enumerate(ex.map(_noise_block, jobs, chunksize=1)):
                results.setdefault((res["label"], res["site"]), []).append(res)
                if progress:
                    progress(f"noise block {i + 1}/{len(jobs)}")
    spikes, examples = {}, {}
    on, k, period = N["warmup_ms"], N["inputs_per_block"], N["period_ms"]
    w0, w1 = N["psth_window_ms"]
    for (label, site), where in cached.items():
        if isinstance(where, tuple):  # just simulated: save
            path, meta, stamp = where
            blocks = sorted(results[(label, site)], key=lambda r: r["block"])
            arrays = {f"b{r['block']}": r["spikes"] for r in blocks}
            for r in blocks:
                if "t" in r:
                    arrays["ex_t"], arrays["ex_v"] = r["t"], r["v"]
            np.savez_compressed(path, **arrays)
            meta.write_text(stamp)
        else:
            path = where
        with np.load(path) as z:
            rel = []
            for b in range(N["n_blocks"]):
                sp = z[f"b{b}"]
                for i in range(k):
                    t0 = on + i * period
                    rel.append(sp[(sp >= t0 + w0) & (sp < t0 + w1)] - t0)
            spikes[(label, site)] = rel
            if "ex_t" in z.files:
                examples.setdefault(label, {})["nosyn" if site is None else "syn"] = (z["ex_t"] - on, z["ex_v"])
    return spikes, examples


def psth(spikes: dict, N: dict) -> dict:
    """{(label, site): (bin_centres, rate_with_Hz, rate_without_Hz)} for each condition and synapse site."""
    w0, w1 = N["psth_window_ms"]
    edges = np.arange(w0, w1 + 1e-9, N["psth_bin_ms"])
    out = {}
    for (label, site), rel in spikes.items():
        if site is None:
            continue
        base = spikes[(label, None)]
        h1 = np.histogram(np.concatenate(rel), edges)[0] / len(rel) / (N["psth_bin_ms"] / 1000)
        h0 = np.histogram(np.concatenate(base), edges)[0] / len(base) / (N["psth_bin_ms"] / 1000)
        out[(label, site)] = ((edges[:-1] + edges[1:]) / 2, h1, h0)
    return out


def evoked_spikes(spikes: dict, N: dict) -> pd.DataFrame:
    """Extra spikes per input in the count window (paired difference) with bootstrap 95% CI, per condition and site."""
    c0, c1 = N["count_window_ms"]
    rng = np.random.default_rng(0)
    rows = []
    for (label, site), rel in spikes.items():
        if site is None:
            continue
        base = spikes[(label, None)]
        d = np.array([np.sum((a >= c0) & (a < c1)) - np.sum((b >= c0) & (b < c1)) for a, b in zip(rel, base)])
        boot = [rng.choice(d, d.size).mean() for _ in range(1000)]
        rows.append(dict(condition=label, site_um=site, evoked=d.mean(), ci_lo=np.percentile(boot, 2.5),
                         ci_hi=np.percentile(boot, 97.5), n_inputs=d.size,
                         p_base=np.mean([np.any((b >= c0) & (b < c1)) for b in base])))
    return pd.DataFrame(rows)


def early_late(spikes: dict, N: dict, split_ms=20.0) -> pd.DataFrame:
    """Extra spikes per input before and after `split_ms` (within the count window), per condition and site."""
    c0, c1 = N["count_window_ms"]
    rows = []
    for (label, site), rel in spikes.items():
        if site is None:
            continue
        base = spikes[(label, None)]
        cnt = lambda arrs, a, b: np.mean([np.sum((x >= a) & (x < b)) for x in arrs])  # noqa: E731
        rows.append(dict(condition=label, site_um=site,
                         early=cnt(rel, c0, split_ms) - cnt(base, c0, split_ms),
                         late=cnt(rel, split_ms, c1) - cnt(base, split_ms, c1)))
    return pd.DataFrame(rows)
