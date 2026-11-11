"""dummy -- the official baseline, re-implemented on the shared toolbox.

Nearest tracked player to the ball gets the ball for his team; frames without a ball say D; then a
centred majority vote over +/- `window` frames. With window=50 it reproduces the organisers'
baseline_smooth50.csv byte for byte (checked by `mtah data check-loader` on the real data), which
makes it the reference point every team solution is compared against: published floor spellF1
0.2433 on the full match.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mtah.common.geometry import nearest_player
from mtah.common.smoothing import majority_vote
from mtah.data.match import MatchData
from mtah.labels import Codes
from mtah.solutions import register
from mtah.solutions.base import Solution


@register
class Dummy(Solution):
    name = "dummy"
    summary = "official baseline: nearest player to the ball + majority vote (+/- window frames)"
    authors = ("organisers' baseline",)
    defaults: ClassVar[dict[str, Any]] = {"window": 50}

    def predict(self, match: MatchData) -> Codes:
        team, _, _ = nearest_player(match.tracking, match.spec.pitch)
        return majority_vote(team, int(self.params["window"]))
