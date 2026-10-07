"""Build figures/report.html: every figure set of configs/figure_set.json in one page.

Uses the cached results in results/ (runs whatever is missing).
"""
import argparse

from _common import progress  # noqa: F401  (sets up the import path)

from pvdend import report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = report.build(args.set_file, args.out, progress=lambda name: print(" ", name))
    print("wrote", path)


if __name__ == "__main__":
    main()
