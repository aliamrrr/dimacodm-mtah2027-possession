"""Every registered solution honours the contract. New solutions are picked up automatically."""

import numpy as np
import pytest

from mtah.config import load_config
from mtah.data import MatchData
from mtah.solutions import broken, discover

SOLUTIONS = sorted(discover())


def test_every_solution_imports():
    assert broken() == {}, broken()


def test_every_solution_has_a_config():
    for name in SOLUTIONS:
        assert load_config(name).path is not None, f"configs/{name}.yaml is missing"


@pytest.mark.parametrize("name", SOLUTIONS)
def test_contract(name, mini_bundle):
    from mtah.pipeline import predict

    config = load_config(name)
    first = predict(config, MatchData(mini_bundle))
    second = predict(config, MatchData(mini_bundle, use_cache=False))
    assert first.dtype == np.int8
    assert first.shape == (36_000,)
    assert first.min() >= 0 and first.max() <= 3
    assert np.array_equal(first, second), "not deterministic (rule 3)"
