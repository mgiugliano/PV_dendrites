"""Run one sweep from a JSON config, save results/ and an overview figure.

A sweep places one synapse at a time at every site along the target dendrite (both morphologies,
unless the config says otherwise) and saves the measurements (summary.csv), the voltage traces
(traces.npz) and the config itself in results/<name>/, plus figures/overview_<name>.{pdf,svg,png}.
Every field of pvdend.config.Config can be overridden on the command line:

    python scripts/run.py configs/default.json
    python scripts/run.py configs/default.json --set ca_profile=increasing g_ca_mS_cm2=2.5 ca_act_shift_mV=-15

Configs saved from the explorer notebook ("Save config" button) are reproduced exactly this way.
"""
import argparse
import json

from _common import progress

from pvdend import Config, get_cell, load_or_run, plotting


def parse_value(text):
    """Interpret a command-line value as JSON (numbers, true/false, lists), else keep it as a string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="JSON config file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config fields (values parsed as JSON)")
    ap.add_argument("--force", action="store_true", help="re-run even if saved results match")
    args = ap.parse_args()

    cfg = Config.from_json(args.config)  # validated: unknown keys or invalid values raise an error
    overrides = dict(kv.split("=", 1) for kv in args.set)
    if overrides:
        cfg = cfg.replace(**{k: parse_value(v) for k, v in overrides.items()})
    print(cfg.to_json())

    res = load_or_run(cfg, force=args.force, progress=progress)  # reuses results/<name>/ if the config is identical
    cells = {m: get_cell(m, cfg) for m in cfg.morphologies}
    fig = plotting.overview_figure(res, cells)
    for p in plotting.save_figure(fig, f"overview_{cfg.name}"):
        print("saved", p)


if __name__ == "__main__":
    main()
