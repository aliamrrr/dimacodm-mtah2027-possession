"""Decoder parameters and deep / logistic blend of dimacodm, on out-of-fold predictions.

    python -m mtah.solutions.dimacodm.deep_tune                          # models/dimacodm/oof/*.npz
    python -m mtah.solutions.dimacodm.deep_tune --convention dfl --objective dfl

Each game's deep log-odds come from the fold model that never saw it (deep_train), its logistic
log-odds from weights fitted on the other games, and both are decoded exactly as DimacodmSolution
decodes BAR-ATH (solution.decode_logits, speed-trend bonuses included). The parameters maximise
the mean of the per-source means of spellF1 (train.balanced_mean), by coordinate ascent. Other
games only, CPU only (no torch); the games are decoded in parallel.

--convention picks the label-convention head(s) of the deep model (both averaged, or one of them,
from the per-convention out-of-fold log-odds in oof_conv/), --objective the games whose mean
spellF1 is maximised (balanced, or one source). Each run writes decoder_<convention>_<objective>.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from mtah.evaluation import evaluate
from mtah.labels import X
from mtah.paths import REPO_ROOT
from mtah.solutions.dimacodm import train as logistic
from mtah.solutions.dimacodm.bench import default_out as bench_root
from mtah.solutions.dimacodm.model import controlled_possession, logit
from mtah.solutions.dimacodm.solution import DimacodmLogisticSolution, decode_logits

app = typer.Typer(add_completion=False)
OUT = REPO_ROOT / "models" / "dimacodm"
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOUNDARY_GAINS = (0.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0)
HEADS = {"both": (0, 1), "metrica": (0,), "dfl": (1,)}  # label-convention heads of the model
# train.SEARCH_SPACE widened: the deep log-odds are sharper than the logistic ones and their
# optimum sat on its edges (dead_scale, dead_bias, switch / stop / restart costs).
SPACE: dict[str, tuple[float, ...]] = {
    "team_scale": (0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0),
    "dead_scale": (0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0),
    "dead_bias": (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    "switch_cost": (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0),
    "stop_cost": (5.0, 6.0, 8.0, 12.0, 15.0, 20.0, 30.0, 40.0),
    "restart_cost": (8.0, 12.0, 15.0, 20.0, 30.0, 45.0, 60.0),
    "same_team_restart_cost": (0.0, 1.0, 2.0, 3.0, 5.0),
    "stop_trend_bonus": (0.0, 2.0, 5.0, 10.0, 15.0),
    "start_trend_bonus": (0.0, 2.0, 5.0, 10.0, 15.0),
}
_GAMES: dict[str, dict[str, Any]] = {}  # per worker process, set once by _init


def _init(games: dict[str, dict[str, Any]]) -> None:
    _GAMES.update(games)


def _score(job: tuple[str, dict[str, Any]]) -> tuple[str, float]:
    """spellF1 of one game, its log-odds blended and decoded as DimacodmSolution.predict does."""
    name, params = job
    game = _GAMES[name]
    w = float(params["blend"])
    team = w * game["team"] + (1 - w) * game["logistic_team"]
    dead = w * game["dead"] + (1 - w) * game["logistic_dead"]
    codes = decode_logits(
        team, dead, game["speed_trend"], params, game["halves"], game.get("team_change")
    )
    codes[game["gt"] == X] = X
    return name, float(evaluate(codes, game[params.get("_truth", "gt")])["spell_f1_mean"])


def load_games(
    out: Path,
    convention: str = "both",
    blend_with_logistic: bool = True,
    profile: str = "simulated",
) -> dict[str, dict[str, Any]]:
    """Out-of-fold deep log-odds + leave-one-game-out logistic log-odds of every game."""
    folder = out / ("oof" if convention == "both" else "oof_conv")
    oof = {p.stem: np.load(p) for p in sorted(folder.glob("*.npz"))}
    if not oof:
        raise typer.BadParameter(f"no out-of-fold predictions under {folder}")
    key = "" if convention == "both" else str(HEADS[convention][0])
    gap = int(DimacodmLogisticSolution.defaults["ball_gap_frames"])
    games = [logistic.load(bench_root(profile) / name, gap) for name in oof]
    payload = {}
    for game in games:
        data = oof[game.name]
        gt = data["gt"].astype(np.int8)  # a smoke run grades only part of the game
        graded = gt != X
        if not np.array_equal(gt[graded], game.gt[graded]):
            raise ValueError(f"{game.name}: out-of-fold ground truth differs from the bundle's")
        zeros = np.zeros(len(gt))
        team_lr, dead_lr = zeros, zeros
        if blend_with_logistic:
            weights = logistic.fit([g for g in games if g is not game])
            team_lr = logit(game.features.team, np.asarray(weights.team))
            dead_lr = logit(game.features.dead, np.asarray(weights.dead))
        payload[game.name] = {
            "team": data[f"team{key}"],
            "dead": data[f"dead{key}"],
            "logistic_team": team_lr,
            "logistic_dead": dead_lr,
            "speed_trend": game.features.speed_trend,
            "gt": gt,
            "halves": game.halves,
            "team_change": data.get(f"boundary{key}"),
        }
    return payload


def source_mean(scores: dict[str, float], objective: str) -> float:
    """balanced: mean of the per-source means; metrica / dfl: mean over that source's games."""
    if objective == "balanced":
        return logistic.balanced_mean(list(scores.items()))
    return float(np.mean([v for k, v in scores.items() if k.startswith(f"{objective}_")]))


def tune(
    out: Path,
    convention: str = "both",
    objective: str = "balanced",
    blend_with_logistic: bool = True,
    workers: int = 0,
    start: Path | None = None,
    min_spell_s: float = 0.0,
    profile: str = "simulated",
) -> dict[str, Any]:
    """`start`: a previous decoder_*.json to start the ascent from (same convention).

    `min_spell_s` > 0: the decoder applies the controlled-possession rule at that duration and is
    scored against the labels of the other games under the SAME rule (model.controlled_possession:
    a touch without control is not a change of possession, CHALLENGE.md). The score against their
    labels as published is reported beside it (`as_labelled`)."""
    games = load_games(out, convention, blend_with_logistic, profile)
    for game in games.values():
        game["gt_controlled"] = controlled_possession(
            game["gt"], game["halves"], round(min_spell_s * 25)
        )
    space = dict(SPACE)
    if any(g["team_change"] is not None and g["team_change"].any() for g in games.values()):
        space["boundary_gain"] = BOUNDARY_GAINS
    if blend_with_logistic:
        space["blend"] = BLENDS
    best: dict[str, Any] = {
        **DimacodmLogisticSolution.defaults,
        "blend": 0.5 if blend_with_logistic else 1.0,
        "conventions": list(HEADS[convention]),
        "boundary_gain": 0.0,
    }
    fixed = {"min_spell_s": min_spell_s, "_truth": "gt_controlled"}
    if start is not None:
        best.update(json.loads(start.read_text(encoding="utf-8"))["params"])
        if not blend_with_logistic:
            best["blend"] = 1.0
        if best["conventions"] != list(HEADS[convention]):
            raise typer.BadParameter(f"{start} was tuned for conventions {best['conventions']}")
    best.update(fixed)
    # The search only decodes the games its objective counts; the report covers them all.
    searched = [n for n in games if objective == "balanced" or n.startswith(f"{objective}_")]
    workers = workers or min(len(searched), os.cpu_count() or 1)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(games,)) as pool:

        def per_game(params: dict[str, Any], names: list[str] = searched) -> dict[str, float]:
            return dict(pool.map(_score, [(name, params) for name in names]))

        def target(params: dict[str, Any]) -> float:
            return source_mean(per_game(params), objective)

        best_value = target(best)
        typer.echo(f"start: {objective} spellF1 {best_value:.4f}")
        for _ in range(2):
            improved = False
            for key, values in space.items():
                for value in values:
                    if value == best.get(key):
                        continue
                    trial = {**best, key: value}
                    current = target(trial)
                    if current > best_value + 1e-4:
                        best, best_value, improved = trial, current, True
                        typer.echo(f"  {key} = {value}: {objective} spellF1 {current:.4f}")
            if not improved:
                break
        scores = per_game(best, list(games))
        as_labelled = per_game({**best, "_truth": "gt"}, list(games))
        result = {
            "convention": convention,
            "objective": objective,
            "min_spell_s": min_spell_s,
            "profile": profile,
            "spell_f1": round(best_value, 4),
            "by_source": {s: round(source_mean(scores, s), 4) for s in ("metrica", "dfl")},
            "as_labelled": {
                s: round(source_mean(as_labelled, s), 4) for s in ("balanced", "metrica", "dfl")
            },
            "params": {k: v for k, v in best.items() if not k.startswith("_")},
            "per_game": {k: round(v, 4) for k, v in scores.items()},
            "blend_curve": {
                str(w): round(target({**best, "blend": w}), 4)
                for w in (BLENDS if blend_with_logistic else ())
            },
        }
    suffix = f"_controlled{min_spell_s:g}s" if min_spell_s else ""
    path = out / f"decoder_{convention}_{objective}{suffix}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@app.command()
def main(
    out: Annotated[Path, typer.Option()] = OUT,
    workers: Annotated[int, typer.Option(help="0: one per CPU (at most one per game)")] = 0,
    convention: Annotated[str, typer.Option(help="both | metrica | dfl")] = "both",
    objective: Annotated[str, typer.Option(help="balanced | metrica | dfl")] = "balanced",
    deep_only: Annotated[bool, typer.Option(help="no blend with the logistic model")] = False,
    start: Annotated[Path | None, typer.Option(help="decoder_*.json to start from")] = None,
    min_spell_s: Annotated[
        float, typer.Option(help="controlled-possession rule: spells shorter than this (s)")
    ] = 0.0,
    profile: Annotated[
        str, typer.Option(help="benchmark of the out-of-fold games: simulated | real-like")
    ] = "simulated",
) -> None:
    result = tune(out, convention, objective, not deep_only, workers, start, min_spell_s, profile)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
