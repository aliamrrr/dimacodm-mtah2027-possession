"""Git provenance for manifests. Never fails: outside a repository every field is None."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from mtah.paths import REPO_ROOT

# Git exports these to its hooks (the pre-push hook runs the tests, `mtah` may run from one): they
# would override `-C REPO_ROOT` and point every command below at another repository.
REPO_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_PREFIX",
)

# Where an untracked file would change the output without being in the recorded commit (rule 3):
# a solution package or a config nobody committed yet must make the tree dirty.
CODE_PATHS = ("src", "configs", "scripts", "vendor", "pyproject.toml", "uv.lock")


def _git(*args: str, strip: bool = True) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            env={k: v for k, v in os.environ.items() if k not in REPO_LOCATION_VARS},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if strip else proc.stdout


def info() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "short": None, "branch": None, "dirty": None, "dirty_files": []}
    # porcelain v1: "XY path"; the leading space of " M path" is significant, never strip it.
    status = _git("status", "--porcelain", "--untracked-files=no", strip=False) or ""
    dirty_files = [line[3:] for line in status.splitlines() if line.strip()]
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS) or ""
    dirty_files += [path for path in untracked.splitlines() if path]
    return {
        "commit": commit,
        "short": commit[:7],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:20],
        "remote": _git("config", "--get", "remote.origin.url"),
    }
