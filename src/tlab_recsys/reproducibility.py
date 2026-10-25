from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import platform

from .config import ROOT
from .download import file_hash

DEPENDENCIES = ("catboost", "numpy", "pandas", "polars", "pyarrow", "scikit-learn", "scipy", "joblib", "threadpoolctl", "psutil")


def runtime_fingerprint() -> dict:
    package = Path(__file__).resolve().parent
    return {
        "code_sha256": {path.name: file_hash(path) for path in sorted(package.glob("*.py")) if path.name != "reporting.py"},
        "uv_lock_sha256": file_hash(ROOT / "uv.lock"),
        "dependencies": {name: version(name) for name in DEPENDENCIES},
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
