"""Run every study in configs/studies.json and save all figures (PDF/PNG) in figures/.

Results are cached in results/ and reused when a configuration has not changed. This runs
exactly the same code as scripts/make_report.py, without writing the HTML report.
"""
from _common import progress  # noqa: F401  (sets up the import path)

from pvdend import report

if __name__ == "__main__":
    report.build(progress=lambda name: print(" ", name), write_html=False)
    print("figures written to figures/")
