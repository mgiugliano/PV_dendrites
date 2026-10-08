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


def noise_study(S=None, progress=None) -> tuple[pd.DataFrame, dict]:
    """Paired noisy-current trials. Returns (spike table, example traces); cached in results/noise/."""
    S = S or spec()
    N = S["noise"]
    folder = RESULTS_DIR / "noise"
    folder.mkdir(parents=True, exist_ok=True)
    key = json.dumps(N, sort_keys=True)
    rows, examples = [], {}
    for label, morph, cfg in noise_conditions(S):
        stem = f"{cfg.name}_{morph}"
        path = folder / f"{stem}.csv"
        ex_path = folder / f"{stem}_example.npz"
        meta = folder / f"{stem}.json"
        if path.exists() and meta.exists() and meta.read_text() == key + cfg.to_json():
            df = pd.read_csv(path)
            with np.load(ex_path) as z:
                examples[label] = {k: z[k] for k in z.files}
        else:
            cell = get_cell(morph, cfg)
            calcium.configure(cell, cfg)
            rb = rheobase_nA(morph, cfg)
            mu, sigma = N["mu_frac"] * rb, N["sigma_frac"] * rb
            on, tstop = N["warmup_ms"], N["warmup_ms"] + N["window_ms"] + 20
            recs = []
            sites = [None] + [d for d in N["sites_um"] if d <= cell.tip_distance + 1e-3]
            for d in sites:
                if progress:
                    progress(f"{cfg.name}: {'no synapse' if d is None else f'{d:g} um'}")
                for trial in range(N["n_trials"]):
                    keep = d == N["example_site_um"] and trial < 5 or (d is None and trial < 5)
                    r = run_noise_trial(cell, cfg, d, 1000 + trial, mu, sigma, N["tau_ms"], on, tstop, keep)
                    sp = r["spikes"]
                    recs.append(dict(site_um=np.nan if d is None else d, trial=trial,
                                     n_window=int(np.sum((sp >= on) & (sp < on + N["window_ms"]))),
                                     n_before=int(np.sum((sp >= on - N["window_ms"]) & (sp < on))),
                                     first_ms=float(sp[sp >= on][0] - on) if np.any(sp >= on) else np.nan,
                                     mu_nA=mu, sigma_nA=sigma))
                    if keep:
                        tag = "nosyn" if d is None else "syn"
                        examples.setdefault(label, {})[f"{tag}_{trial}_t"] = r["t"]
                        examples[label][f"{tag}_{trial}_v"] = r["v_soma"]
            df = pd.DataFrame(recs)
            df.to_csv(path, index=False)
            np.savez_compressed(ex_path, **examples[label])
            meta.write_text(key + cfg.to_json())
        df = df.assign(condition=label, morphology=morph)
        rows.append(df)
    return pd.concat(rows, ignore_index=True), examples


def evoked_probability(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition and site: P(spike in window) with the synapse minus without (same noise seeds)."""
    out = []
    for cond, grp in df.groupby("condition", sort=False):
        base = grp[grp.site_um.isna()].set_index("trial").n_window > 0
        for d, g in grp[grp.site_um.notna()].groupby("site_um"):
            p = (g.set_index("trial").n_window > 0)
            out.append(dict(condition=cond, site_um=d, p_syn=p.mean(), p_base=base.mean(),
                            p_evoked=(p.astype(int) - base.reindex(p.index).astype(int)).mean(),
                            latency_ms=g.first_ms.median()))
    return pd.DataFrame(out)
