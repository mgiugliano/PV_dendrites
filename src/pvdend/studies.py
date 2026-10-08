"""Multi-configuration studies defined in configs/figure_set.json (beyond single figure sets)."""
from __future__ import annotations

import json

from ._paths import CONFIG_DIR
from .config import Config
from .protocols import load_or_run


def figure_set(set_file=None) -> dict:
    return json.loads((set_file or CONFIG_DIR / "figure_set.json").read_text())


def bias_study(spec: dict | None = None, set_file=None, progress=None) -> dict:
    """{fraction: {profile: Result}} for the somatic-bias study; fraction 0 reuses its base figure set."""
    fs = figure_set(set_file)
    spec = spec or fs["bias_study"]
    base_entry = next(e for e in fs["figures"] if e["name"] == spec["base_set"])
    base = Config.from_json(CONFIG_DIR / base_entry["base"]).replace(**base_entry.get("overrides", {}))
    out = {}
    for frac in spec["fractions"]:
        out[frac] = {}
        for prof in spec["profiles"]:
            name = (f"{spec['base_set']}__{prof}" if frac == 0 else f"{spec['name']}{frac:g}__{prof}")
            cfg = base.replace(name=name, ca_profile=prof, soma_bias_frac=frac)
            if progress:
                progress(name)
            out[frac][prof] = load_or_run(cfg)
    return out
