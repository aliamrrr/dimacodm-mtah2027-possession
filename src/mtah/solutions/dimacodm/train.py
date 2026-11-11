"""Train the two logistic models and choose the decoder parameters, on OTHER games only.

    python -m mtah.solutions.dimacodm.bench          # once: the benchmark bundles
    python -m mtah.solutions.dimacodm.train          # leave-one-game-out report + weights.json
    python -m mtah.solutions.dimacodm.train --search # also search the decoder parameters (slow)

Leave-one-game-out: the models are fitted on every game but one and scored on that one, so every
number printed here is measured on a game the models never saw. weights.json is then fitted on
all of them.
Deterministic from start to end (no randomness at all), hence reproducible by the organisers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import typer
from rich.console import Console

from mtah.data.match import MatchData
from mtah.evaluation import evaluate, load_gt
from mtah.labels import A, D, H, X
from mtah.solutions.dimacodm.bench import iter_bundles
from mtah.solutions.dimacodm.features import (
    DEAD_FEATURES,
    TEAM_FEATURES,
    FrameFeatures,
    frame_features,
)
from mtah.solutions.dimacodm.model import Weights, fit_logistic, logit
from mtah.solutions.dimacodm.solution import DimacodmLogisticSolution, decode_features

console = Console(width=140)
app = typer.Typer(add_completion=False)
SUBSAMPLE = 5  # consecutive frames are near duplicates


@dataclass
class GameData:
    name: str
    features: FrameFeatures
    gt: npt.NDArray[np.int8]
    halves: list[tuple[int, int]]

    @property
    def graded(self) -> npt.NDArray[np.bool_]:
        return self.gt != X


def load(bundle: Path, ball_gap_frames: int) -> GameData:
    match = MatchData(bundle, use_cache=False)
    tr, spec = match.tracking, match.spec
    features = frame_features(
        np.asarray(tr.ball),
        np.asarray(tr.players),
        np.asarray(tr.slot_team),
        spec.pitch,
        spec.fps,
        spec.graded_mask(),
        ball_gap_frames,
    )
    gt = load_gt(bundle / "synthetic" / "gt_full.csv")
    return GameData(bundle.name, features, gt, [(a, b + 1) for _, a, b in spec.halves])


def fit(games: list[GameData]) -> Weights:
    dead_rows = [np.flatnonzero(g.graded)[::SUBSAMPLE] for g in games]
    dead_x = np.concatenate([g.features.dead[r] for g, r in zip(games, dead_rows, strict=True)])
    dead_y = np.concatenate([g.gt[r] == D for g, r in zip(games, dead_rows, strict=True)])
    team_rows = [np.flatnonzero(np.isin(g.gt, (H, A)))[::SUBSAMPLE] for g in games]
    team_x = np.concatenate([g.features.team[r] for g, r in zip(games, team_rows, strict=True)])
    team_y = np.concatenate([g.gt[r] == H for g, r in zip(games, team_rows, strict=True)])
    return Weights(
        dead=fit_logistic(dead_x, dead_y).round(6).tolist(),
        team=fit_logistic(team_x, team_y, intercept=False).round(6).tolist(),
        dead_features=list(DEAD_FEATURES),
        team_features=list(TEAM_FEATURES),
        trained_on=[g.name for g in games],
    )


def auc(score: npt.NDArray[np.float64], positive: npt.NDArray[np.bool_]) -> float:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def score_game(game: GameData, weights: Weights, params: dict[str, Any]) -> dict[str, float]:
    codes = decode_features(game.features, weights, params, game.halves)
    codes[~game.graded] = X
    result = evaluate(codes, game.gt)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
    }


def cross_validate(games: list[GameData], params: dict[str, Any]) -> list[dict[str, float]]:
    rows = []
    for held_out in games:
        weights = fit([g for g in games if g is not held_out])
        row = score_game(held_out, weights, params)
        graded = held_out.graded
        dead_logit = logit(held_out.features.dead, np.asarray(weights.dead))
        row["auc_dead"] = auc(dead_logit[graded], held_out.gt[graded] == D)
        alive = np.isin(held_out.gt, (H, A)) & held_out.features.seen
        team_logit = logit(held_out.features.team, np.asarray(weights.team))
        row["team_acc_seen"] = float(np.mean((team_logit[alive] > 0) == (held_out.gt[alive] == H)))
        rows.append(row)
    return rows


def balanced_mean(scores: list[tuple[str, float]]) -> float:
    """Mean of the per-source means: Metrica (labels from events) and DFL (labels from the
    tracking) count equally, whatever their number of games: we do not know which convention
    is closer to BAR-ATH's ground truth, so the parameters must hold for both."""
    groups: dict[str, list[float]] = {}
    for name, value in scores:
        groups.setdefault(name.split("_")[0], []).append(value)
    return float(np.mean([np.mean(v) for v in groups.values()]))


SEARCH_SPACE: dict[str, tuple[float, ...]] = {
    "team_scale": (0.2, 0.3, 0.4, 0.6, 0.8, 1.2),
    "dead_scale": (0.1, 0.15, 0.25, 0.35, 0.5),
    "dead_bias": (-1.5, -1.0, -0.5, 0.0, 0.5),
    "switch_cost": (2.0, 3.0, 4.0, 5.0, 7.0, 10.0),
    "stop_cost": (8.0, 12.0, 15.0, 20.0, 30.0),
    "restart_cost": (8.0, 12.0, 15.0, 20.0, 30.0),
    "same_team_restart_cost": (0.0, 1.0, 2.0, 3.0),
    "stop_trend_bonus": (0.0, 2.0, 5.0, 10.0),
    "start_trend_bonus": (0.0, 2.0, 5.0, 10.0),
}


def coordinate_search(
    games: list[GameData], start: dict[str, Any], rounds: int = 1
) -> tuple[dict[str, Any], float]:
    """Coordinate ascent of the mean leave-one-game-out spellF1, one parameter at a time."""
    folds = [(g, fit([o for o in games if o is not g])) for g in games]

    def objective(params: dict[str, Any]) -> float:
        return balanced_mean([(g.name, score_game(g, w, params)["spell_f1"]) for g, w in folds])

    best, best_value = dict(start), objective(start)
    console.print(f"  start: mean LOGO spellF1 {best_value:.4f}")
    for _ in range(rounds):
        improved = False
        for key, values in SEARCH_SPACE.items():
            for value in values:
                if value == best[key]:
                    continue
                trial = {**best, key: value}
                current = objective(trial)
                if current > best_value + 1e-4:
                    best, best_value, improved = trial, current, True
                    console.print(f"  {key} = {value}: mean LOGO spellF1 {current:.4f}")
        if not improved:
            break
    return best, best_value


@app.command()
def main(
    search: Annotated[bool, typer.Option(help="search the decoder parameters")] = False,
    root: Annotated[Path | None, typer.Option(help="default: <cache>/bench")] = None,
    write: Annotated[bool, typer.Option(help="write weights.json")] = True,
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="key=value")] = None,
    dataset: Annotated[str, typer.Option(help="all | metrica | dfl")] = "all",
) -> None:
    from mtah.config import parse_overrides

    params = {**DimacodmLogisticSolution.defaults, **parse_overrides(param or [])}
    games = [load(b, int(params["ball_gap_frames"])) for b in iter_bundles(root, dataset)]
    if len(games) < 2:
        raise typer.BadParameter(
            "need the benchmark bundles: python -m mtah.solutions.dimacodm.bench"
        )
    if search:
        params, value = coordinate_search(games, params)
        console.print(f"best mean LOGO spellF1 {value:.4f} with {params}")
    rows = cross_validate(games, params)
    for game, row in zip(games, rows, strict=True):
        console.print(
            f"{game.name}: spellF1 {row['spell_f1']:.4f}  macroF1 {row['macro_f1']:.4f}  "
            f"poss {row['n_pred']}/{row['n_gt']}  AUC dead {row['auc_dead']:.3f}  "
            f"team acc (ball seen) {row['team_acc_seen']:.3f}"
        )
    pairs = [(g.name, r["spell_f1"]) for g, r in zip(games, rows, strict=True)]
    console.print(
        f"mean LOGO spellF1 {np.mean([r['spell_f1'] for r in rows]):.4f}  "
        f"balanced (metrica / dfl) {balanced_mean(pairs):.4f}"
    )
    weights = fit(games)
    if write:
        weights.save()
        console.print(f"wrote weights.json (trained on {weights.trained_on})")
    console.print({"params": params})


if __name__ == "__main__":  # pragma: no cover
    app()
