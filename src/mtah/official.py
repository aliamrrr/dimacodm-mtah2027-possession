"""Access to the vendored official code (vendor/official/), never modified.

The organisers state that the scorer in their repository is the one that decides the ranking.
We load *their* files from disk instead of re-implementing anything, so a metric can never drift
from the official definition. `scripts/sync_official.py --check` proves the bytes are untouched.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

from mtah.paths import OFFICIAL_EVAL_DIR, OFFICIAL_MANIFEST, VENDOR_DIR


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - only if the file vanished
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def metrics() -> Any:
    """The official possession_metrics module (spellF1, macroF1, validation)."""
    return _load("mtah_official_possession_metrics", OFFICIAL_EVAL_DIR / "possession_metrics.py")


@cache
def games() -> Any:
    """The official games module (match constants, frame alignment, asset paths)."""
    return _load("mtah_official_games", VENDOR_DIR / "games.py")


@cache
def manifest() -> dict[str, Any]:
    with open(OFFICIAL_MANIFEST, encoding="utf-8") as handle:
        return dict(json.load(handle))


def run_validator(csv_path: Path, rows: int | None = None) -> tuple[bool, str]:
    """Run the official validate_submission.py, unchanged, in a subprocess.

    This is the exact check the organisers' scorer applies. Returns (ok, printed output).
    """
    cmd = [sys.executable, str(OFFICIAL_EVAL_DIR / "validate_submission.py"), str(csv_path)]
    if rows is not None:
        cmd += ["--rows", str(rows)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_scoreboard(results_dir: Path, gt: str | Path, out_dir: Path, bootstrap: int = 0) -> int:
    """Run the official run_evaluation.py on a directory of CSVs (prints the official board)."""
    cmd = [
        sys.executable,
        str(OFFICIAL_EVAL_DIR / "run_evaluation.py"),
        "--results",
        str(results_dir),
        "--gt",
        str(gt),
        "--out",
        str(out_dir),
        "--bootstrap",
        str(bootstrap),
    ]
    return subprocess.run(cmd, check=False).returncode
