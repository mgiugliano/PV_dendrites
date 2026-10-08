"""Shared helpers for the command-line scripts.

Importing this module puts `src/` on the Python path, so the scripts work from a plain clone of the
repository without installing the package (`pip install -e .` works too).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def progress(i, n, text):
    """Print a one-line progress counter, overwritten in place ('[ 12/40] long: 120 um')."""
    print(f"\r  [{i:>3}/{n}] {text:<30}", end="" if i < n else "\n", flush=True)
