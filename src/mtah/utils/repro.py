"""Determinism helpers. Rule 3: running the repository must reproduce the submission exactly."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mtah.paths import REPO_ROOT

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "lightgbm",
    "xgboost",
    "opencv-python-headless",
    "pyyaml",
    "mtah",
)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch, and ask torch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG, seeded for third-party code
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if "torch" in sys.modules or _installed("torch"):
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except ImportError:  # pragma: no cover
            pass


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    lock = REPO_ROOT / "uv.lock"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
    }
