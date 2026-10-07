"""Regenerate every figure listed in configs/figure_set.json.

Each entry gives a base config, optional overrides, and the Ca profiles to
compare. For every entry one overview figure per profile and one comparison
figure are written to figures/. Results are cached in results/ and reused
when the config has not changed (use --force to re-run).
"""
import argparse
import json

from _common import progress

from pvdend import Config, get_cell, load_or_run, plotting
from pvdend._paths import CONFIG_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-file", default=str(CONFIG_DIR / "figure_set.json"))
    ap.add_argument("--only", nargs="*", help="names of figure-set entries to build")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    entries = json.loads(open(args.set_file).read())["figures"]
    for entry in entries:
        if args.only and entry["name"] not in args.only:
            continue
        base = Config.from_json(CONFIG_DIR / entry["base"]).replace(**entry.get("overrides", {}))
        results = {}
        for prof in entry["profiles"]:
            cfg = base.replace(name=f"{entry['name']}__{prof}", ca_profile=prof)
            print(f"{cfg.name}")
            res = load_or_run(cfg, force=args.force, progress=progress)
            results[prof] = res
            cells = {m: get_cell(m, cfg) for m in cfg.morphologies}
            fig = plotting.overview_figure(res, cells)
            plotting.save_figure(fig, f"{entry['name']}/overview_{prof}")
            plotting.plt.close(fig)
        fig = plotting.comparison_figure(results)
        for p in plotting.save_figure(fig, f"{entry['name']}/comparison"):
            print("saved", p)
        plotting.plt.close(fig)


if __name__ == "__main__":
    main()
