"""ali_deep: the set-transformer + TCN model (trained on other games) and the Viterbi decoder.

Registered only once trained weights exist (models/ali_deep/fold*.npz, written by deep_train on a
GPU), so the rest of the repository works without them. Inference is numpy (deep_net): the
submission needs no torch. Per frame, the deep model's (team, dead) log-odds are averaged over the
fold models (5 folds + fold 5, trained on all ten games), the label conventions listed in
`conventions` and the team-swapped mirror. They can be blended with the logistic ones of `ali`
(`blend` = weight of the deep model; 1 in the submitted configuration, the deep model alone),
then they are decoded by the same 4-state Viterbi as `ali`.
Every parameter comes from out-of-fold predictions on other games (deep_tune).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from mtah.data.match import MatchData
from mtah.labels import Codes, X
from mtah.paths import REPO_ROOT
from mtah.solutions import register
from mtah.solutions.ali.deep_features import inputs
from mtah.solutions.ali.deep_net import NumpyNet, ensemble_outputs
from mtah.solutions.ali.features import frame_features
from mtah.solutions.ali.model import Weights, logit
from mtah.solutions.ali.solution import AliSolution, decode_logits
from mtah.solutions.base import Solution

MODELS = REPO_ROOT / "models" / "ali_deep"


def fold_models(folder: Path = MODELS) -> list[Path]:
    return sorted(folder.glob("fold*.npz"))


class AliDeepSolution(Solution):
    name = "ali_deep"
    summary = "set transformer + TCN on the tracking (trained on other games) + Viterbi"
    authors = ("DIMACodm",)
    defaults: ClassVar[dict[str, Any]] = {
        # Chosen out-of-fold by `deep_tune --convention dfl --objective balanced --profile
        # real-like-moving` on the 10 other games (mean of the per-source means 0.7515, labels
        # as published), never on BAR-ATH: see configs/ali_deep.yaml.
        **AliSolution.defaults,
        "team_scale": 0.3,
        "dead_scale": 0.35,
        "dead_bias": -2.5,
        "switch_cost": 5.0,
        "stop_cost": 5.0,
        "restart_cost": 30.0,
        "same_team_restart_cost": 1.0,
        "stop_trend_bonus": 0.0,
        "start_trend_bonus": 10.0,
        "blend": 1.0,  # weight of the deep log-odds (1 - blend: the logistic ones)
        "boundary_gain": 4.0,  # changing team costs up to this much less where a change is detected
        # Controlled-possession rule (model.controlled_possession), OFF. We read CHALLENGE.md as
        # "a touch without control is not a change of possession" and tried 2 s; the public
        # reference, which that same document says wins over any reading of the definition,
        # contains possessions that the rule deletes. So brief spells are kept, as in the labels
        # of the other games the model is trained on.
        "min_spell_s": 0.0,
        # label-convention head(s): 1 = DFL, flags of the corrected tracking, as the ground truth
        # is built (CHALLENGE.md); 0 = Metrica events
        "conventions": [1],
        "models": "models/ali_deep",
    }

    def predict(self, match: MatchData) -> Codes:
        p = self.params
        spec, tracking = match.spec, match.tracking
        graded = spec.graded_mask()
        players, ball = np.asarray(tracking.players), np.asarray(tracking.ball)
        slot_team = np.asarray(tracking.slot_team)
        features = frame_features(
            ball, players, slot_team, spec.pitch, spec.fps, graded, int(p["ball_gap_frames"])
        )
        weights = Weights.load()
        team = logit(features.team, np.asarray(weights.team))
        dead = logit(features.dead, np.asarray(weights.dead))
        team_change = None
        blend = float(p["blend"])
        if blend > 0:
            # imported here: `python -m mtah.solutions.ali.deep_weights` must not find it loaded
            from mtah.solutions.ali.deep_weights import verified

            nets = [NumpyNet.load(path) for path in verified(REPO_ROOT / str(p["models"]))]
            if not nets:
                raise FileNotFoundError(f"no fold*.npz under {p['models']}: run deep_train first")
            deep_team, deep_dead = np.zeros_like(team), np.zeros_like(dead)
            team_change = np.zeros_like(team)
            for _, first, last in spec.halves:
                tokens, mask, globals_ = inputs(
                    players[first : last + 1],
                    slot_team,
                    ball[first : last + 1],
                    spec.pitch,
                    spec.fps,
                    int(p["ball_gap_frames"]),
                )
                t, d, c = ensemble_outputs(nets, tokens, mask, globals_, tuple(p["conventions"]))
                deep_team[first : last + 1], deep_dead[first : last + 1] = t, d
                team_change[first : last + 1] = c
            team = blend * deep_team + (1 - blend) * team
            dead = blend * deep_dead + (1 - blend) * dead
        halves = [(first, last + 1) for _, first, last in spec.halves]
        codes = decode_logits(
            team, dead, features.speed_trend, p, halves, team_change, fps=spec.fps
        )
        codes[~graded] = X
        return codes


if fold_models():  # only once trained: the repository must work without the weights
    register(AliDeepSolution)
