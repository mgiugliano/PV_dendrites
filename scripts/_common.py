import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def progress(i, n, text):
    print(f"\r  [{i:>3}/{n}] {text:<30}", end="" if i < n else "\n", flush=True)
