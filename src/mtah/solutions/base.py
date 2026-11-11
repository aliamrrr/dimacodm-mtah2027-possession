"""The contract every solution implements.

    @register
    class MySolution(Solution):
        name = "alice"                       # = folder name = config name = CLI name
        summary = "one line for the scoreboard"
        authors = ("Alice",)
        defaults = {"window": 50}            # overridden by configs/alice.yaml -> params

        def predict(self, match: MatchData) -> Codes:
            ...                              # one int8 code per video frame: H=0 A=1 D=2 X=3

Rules the pipeline enforces on every run (and the tests on the synthetic match):
  * len(output) == match.spec.n_frames, dtype integer, values in 0..3;
  * the same inputs + config + seed give the same bytes (rule 3 of the challenge);
  * inputs are read-only: copy before modifying.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mtah.data.match import MatchData
from mtah.labels import Codes

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


class Solution(ABC):
    name: ClassVar[str]
    summary: ClassVar[str] = ""
    authors: ClassVar[tuple[str, ...]] = ()
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = sorted(set(params) - set(self.defaults))
        if unknown and self.defaults:
            raise ValueError(
                f"{self.name}: unknown params {unknown}; known: {sorted(self.defaults)}"
            )
        self.params: dict[str, Any] = {**self.defaults, **params}

    @abstractmethod
    def predict(self, match: MatchData) -> Codes:
        """One label code per video frame of the tactical video."""
