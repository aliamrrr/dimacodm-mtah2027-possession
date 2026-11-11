"""Every filesystem location the project uses, in one place.

Override with environment variables rather than editing code, or in a `.env` file at the root of
the repository (copy `.env.example`; the shell environment always wins over `.env`):

    MTAH_DATA_DIR    the unpacked data bundle (default: ./data)
    MTAH_CACHE_DIR   parsed-data cache (default: ./.cache)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

VENDOR_DIR = REPO_ROOT / "vendor" / "official"
OFFICIAL_EVAL_DIR = VENDOR_DIR / "evaluation"
OFFICIAL_MANIFEST = VENDOR_DIR / "MANIFEST.json"
PUBLIC_GT = OFFICIAL_EVAL_DIR / "gt" / "BAR-ATH_possession_gt_public.csv"

CONFIGS_DIR = REPO_ROOT / "configs"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
FINAL_DIR = REPO_ROOT / "final"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SOLUTIONS_PKG_DIR = REPO_ROOT / "src" / "mtah" / "solutions"
SYNTHETIC_DIR = REPO_ROOT / "data_synthetic"


def _load_dotenv(path: Path) -> None:
    """MTAH_* settings from a .env file, without overriding the real environment.

    A relative *_DIR path is taken from the repository root, whatever the current directory.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not sep or not key.startswith("MTAH_") or key in os.environ:
            continue
        if key.endswith("_DIR") and value and not value.startswith("~"):
            value = str((REPO_ROOT / value).resolve())
        os.environ[key] = value


_load_dotenv(REPO_ROOT / ".env")


def data_dir() -> Path:
    """The real data bundle, unless MTAH_DATA_DIR points elsewhere."""
    return Path(os.environ.get("MTAH_DATA_DIR") or REPO_ROOT / "data").expanduser().resolve()


def cache_dir() -> Path:
    path = Path(os.environ.get("MTAH_CACHE_DIR") or REPO_ROOT / ".cache").expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
