"""Solution registry. Each sub-package (not starting with '_') registers one solution.

from mtah.solutions import discover, get
get("dummy")(**params).predict(match)
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from typing import TypeVar

from mtah.solutions.base import NAME_PATTERN, Solution

_REGISTRY: dict[str, type[Solution]] = {}
_BROKEN: dict[str, str] = {}
_DISCOVERED = False

S = TypeVar("S", bound=type[Solution])


def register(cls: S) -> S:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"{cls.__qualname__}.name must match {NAME_PATTERN.pattern}, got {name!r}")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"two solutions are named {name!r}: {existing} and {cls}")
    _REGISTRY[name] = cls
    return cls


def discover() -> dict[str, type[Solution]]:
    """Import every solution package once. Packages that fail to import are reported, not fatal,
    so one solution's missing dependency never blocks the others (`mtah list` shows them)."""
    global _DISCOVERED
    if not _DISCOVERED:
        for info in pkgutil.iter_modules(__path__):
            if info.name.startswith("_") or info.name == "base" or not info.ispkg:
                continue
            try:
                importlib.import_module(f"{__name__}.{info.name}")
            except Exception:
                _BROKEN[info.name] = traceback.format_exc(limit=3)
        _DISCOVERED = True
    return dict(sorted(_REGISTRY.items()))


def broken() -> dict[str, str]:
    discover()
    return dict(_BROKEN)


def get(name: str) -> type[Solution]:
    solutions = discover()
    if name not in solutions:
        hint = f"\n{name} failed to import:\n{_BROKEN[name]}" if name in _BROKEN else ""
        raise KeyError(f"unknown solution {name!r}; available: {sorted(solutions)}{hint}")
    return solutions[name]


__all__ = ["Solution", "broken", "discover", "get", "register"]
