"""Run configuration: configs/<solution>.yaml.

    solution: dummy        # must match the file name
    seed: 0                # seeds python, numpy (and torch if installed) before predict()
    params:                # passed to the Solution constructor, overriding its defaults
      window: 50

The full config and its sha256 are written into every run manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mtah.paths import CONFIGS_DIR


@dataclass(frozen=True)
class RunConfig:
    solution: str
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"solution": self.solution, "seed": self.seed, "params": self.params}

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def default_config_path(solution: str) -> Path:
    return CONFIGS_DIR / f"{solution}.yaml"


def load_config(
    solution: str, path: Path | str | None = None, overrides: dict[str, Any] | None = None
) -> RunConfig:
    config_path = Path(path) if path is not None else default_config_path(solution)
    data: dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    elif path is not None:
        raise FileNotFoundError(config_path)
    declared = data.get("solution", solution)
    if declared != solution:
        raise ValueError(f"{config_path} declares solution {declared!r}, expected {solution!r}")
    params = dict(data.get("params") or {})
    params.update(overrides or {})
    return RunConfig(
        solution=solution,
        seed=int(data.get("seed", 0)),
        params=params,
        path=str(config_path) if config_path.is_file() else None,
    )


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """['window=25', 'mode=fast'] -> {'window': 25, 'mode': 'fast'} (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out
