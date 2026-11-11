"""dimacodm_logistic: control-based possession with a dead-ball detector and a 4-state decoder.

Per frame, two pieces of evidence, both from positions only:

* who controls the ball: a logistic model of the ball-to-nearest-player distances of each team,
  whether the ball moves with that player and whether it was just kicked there (0 when the ball is
  unseen: possession then simply carries on, as the rules say);
* is the ball dead: a logistic model of how fast the players move (relative to this match's own
  level, and before / after the frame), the ball seen beyond a line, players off the pitch, the
  kick-off formation.

A Viterbi decoder over H, A, D-after-H, D-after-A turns that into possessions: changing team
costs `switch_cost`, a stoppage `stop_cost` (cheaper while the players slow down), a restart
`restart_cost` (cheaper while they speed up, dearer by `same_team_restart_cost` when the team
that had the ball restarts). Both models were trained on other games only (the benchmark games,
see bench.py / train.py) and every parameter here was chosen on them, never on BAR-ATH, whose
public 5 minutes only serve to check the pipeline.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from mtah.data.match import MatchData
from mtah.labels import Codes, X
from mtah.solutions import register
from mtah.solutions.base import Solution
from mtah.solutions.dimacodm.features import FrameFeatures, frame_features
from mtah.solutions.dimacodm.model import (
    Weights,
    controlled_possession,
    decode,
    logit,
    transition_costs,
)


def decode_features(
    features: FrameFeatures,
    weights: Weights,
    params: dict[str, Any],
    halves: list[tuple[int, int]],
) -> Codes:
    """Labels (H/A/D everywhere, the caller writes X outside the halves) from the features."""
    return decode_logits(
        logit(features.team, np.asarray(weights.team)),
        logit(features.dead, np.asarray(weights.dead)),
        features.speed_trend,
        params,
        halves,
    )


def decode_logits(
    team_logit: np.ndarray,
    dead_logit: np.ndarray,
    speed_trend: np.ndarray,
    params: dict[str, Any],
    halves: list[tuple[int, int]],
    team_change: np.ndarray | None = None,
    fps: int = 25,
) -> Codes:
    """Labels from per-frame log-odds (team H vs A, dead vs alive) and the players' speed trend.

    team_change: probability per frame that the team changes (ali_seg): changing team costs up to
    `boundary_gain` less there, never below zero.

    `min_spell_s` (0 = off): after decoding, a touch without control (a team spell shorter than
    this, between the other team and either that team again or a stoppage) does not change
    possession (model.controlled_possession, the written rule of CHALLENGE.md).

    Every decoder input is rounded to 1e-4 first. Rule 3 wants the organisers to regenerate the
    SAME file on their own machine, and BLAS / SIMD paths may differ there in the last bits of a
    dot product; rounding makes such a difference vanish before the (sequential, bit-exact)
    Viterbi recursion could turn a near-tie the other way.
    """
    costs = transition_costs(
        float(params["switch_cost"]),
        float(params["stop_cost"]),
        float(params["restart_cost"]),
        float(params["same_team_restart_cost"]),
    )
    trend = np.round(speed_trend, 4)
    stop_gain, start_gain = float(params["stop_trend_bonus"]), float(params["start_trend_bonus"])
    varying = bool(stop_gain or start_gain)
    gain = float(params.get("boundary_gain", 0.0))
    switch = None
    if team_change is not None and gain:
        switch = np.minimum(float(params["switch_cost"]), gain * np.round(team_change, 4))
    codes = decode(
        np.round(team_logit, 4),
        np.round(dead_logit, 4),
        halves,
        len(trend),
        float(params["team_scale"]),
        float(params["dead_scale"]),
        float(params["dead_bias"]),
        costs,
        stop_bonus=stop_gain * np.maximum(0.0, -trend) if varying else None,
        start_bonus=start_gain * np.maximum(0.0, trend) if varying else None,
        switch_bonus=switch,
    )
    min_frames = round(float(params.get("min_spell_s", 0.0)) * fps)
    return controlled_possession(codes, halves, min_frames)


@register
class DimacodmLogisticSolution(Solution):
    name = "dimacodm_logistic"
    summary = "ball control + dead-ball detector (trained on other games) + 4-state Viterbi"
    authors = ("DIMACodm",)
    # Every key can be overridden in configs/dimacodm_logistic.yaml -> params, or on the command line: -p key=value
    defaults: ClassVar[dict[str, Any]] = {
        # Chosen by `python -m mtah.solutions.dimacodm.train --search` on the benchmark of other games
        # (3 Metrica + 7 DFL, leave-one-game-out spellF1: mean of the per-source means 0.4602),
        # never on BAR-ATH.
        "ball_gap_frames": 25,  # bridge ball dropouts up to this long (frames): 1 s
        "team_scale": 0.6,  # weight of the per-frame team log-odds (frames are correlated)
        "dead_scale": 0.25,  # weight of the per-frame dead-ball log-odds
        "dead_bias": -1.5,  # shift of the dead-ball log-odds (higher = fewer stoppages)
        "switch_cost": 7.0,  # H <-> A in open play
        "stop_cost": 20.0,  # alive -> dead
        "restart_cost": 12.0,  # dead -> the other team
        "same_team_restart_cost": 1.0,  # extra cost when the same team restarts
        "stop_trend_bonus": 5.0,  # stoppage cheaper by this x how fast the players slow down
        "start_trend_bonus": 10.0,  # restart cheaper by this x how fast they speed up
    }

    def predict(self, match: MatchData) -> Codes:
        spec = match.spec
        tracking = match.tracking
        graded = spec.graded_mask()
        features = frame_features(
            np.asarray(tracking.ball),
            np.asarray(tracking.players),
            np.asarray(tracking.slot_team),
            spec.pitch,
            spec.fps,
            graded,
            int(self.params["ball_gap_frames"]),
        )
        codes = decode_features(
            features,
            Weights.load(),
            self.params,
            [(first, last + 1) for _, first, last in spec.halves],
        )
        codes[~graded] = X
        return codes
