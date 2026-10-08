"""Build figures/report.html: every study of configs/studies.json in one self-contained page.

Runs whatever simulations are missing (results are cached in results/ and reused when their
parameters have not changed), saves every figure as PDF/PNG in figures/, and writes the report
with all figures, tables, the mechanism analysis and the draft manuscript text.

    python scripts/make_report.py                 # default studies and output
    python scripts/make_report.py --out my.html   # another output file

The full set of simulations takes 1-2 hours on a laptop the first time (the noise study runs in
parallel on several cores); afterwards, rebuilding the report takes about a minute.
"""
import argparse

from _common import progress  # noqa: F401  (importing _common sets up the import path)

from pvdend import report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-file", default=None, help="study definition (default: configs/studies.json)")
    ap.add_argument("--out", default=None, help="output HTML file (default: figures/report.html)")
    args = ap.parse_args()
    path = report.build(args.set_file, args.out, progress=lambda name: print(" ", name))
    print("wrote", path)


if __name__ == "__main__":  # required: the noise study starts worker processes that re-import this file
    main()
