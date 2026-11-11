"""Shared fixtures. Everything runs on a SHORT synthetic match (no real data, ever, in tests)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mtah.data import MatchData
from mtah.synthetic import generate
from mtah.utils.gitinfo import REPO_LOCATION_VARS


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory: pytest.TempPathFactory) -> None:
    os.environ["MTAH_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="session", autouse=True)
def _no_inherited_git_repository() -> None:
    """The pre-push hook runs these tests with GIT_DIR exported by git: a test that creates a
    scratch repository would otherwise run `git init` / `git add` on the team's repository (seen:
    core.bare flipped to true and the index replaced). Every git command must find its own repo."""
    for name in REPO_LOCATION_VARS:
        os.environ.pop(name, None)


@pytest.fixture(scope="session")
def mini_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_mini")
    generate(root, seed=0, mini=True)
    return root


@pytest.fixture()
def mini_match(mini_bundle: Path) -> MatchData:
    return MatchData(mini_bundle)
