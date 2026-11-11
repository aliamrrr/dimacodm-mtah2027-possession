"""Train the deep model on OTHER games: 5 folds by game, out-of-fold predictions.

On a GPU machine (docs/TRAINING.md, train_gpu.sh):

    uv sync --locked --group train
    python -m mtah.solutions.dimacodm.bench --dataset all             # public data -> bundles + caches
    python -m mtah.solutions.dimacodm.deep_train --device cuda        # every fold, one after the other
    python -m mtah.solutions.dimacodm.deep_train --device cuda --fold 2   # or one fold per GPU
    python -m mtah.solutions.dimacodm.deep_tune                       # decoder + blend, out-of-fold

Locally, `--smoke` runs a tiny model for a few steps to check the whole chain on CPU.

Each fold holds two games out (both label conventions where possible), trains on the other eight,
then writes, under --out (default models/dimacodm/):
    fold{k}.npz        weights for deep_net.NumpyNet (what runs on BAR-ATH)
    fold{k}.pt         torch checkpoint (resume / inspection)
    oof/{game}.npz     out-of-fold team / dead log-odds on that game's benchmark bundle
    fold{k}.json       training curve and held-out scores
Fold 5 (`--fold 5`, after the others) holds nothing out: same recipe on all ten games, one more
member for the BAR-ATH ensemble; it has no out-of-fold predictions, so it plays no part in
choosing the decoder.
Nothing here reads BAR-ATH: training, model selection and decoder parameters all come from the
benchmark of other games.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mtah.data.match import MatchData
from mtah.evaluation import evaluate, load_gt
from mtah.labels import X
from mtah.paths import REPO_ROOT
from mtah.solutions.dimacodm.bench import default_out as bench_root
from mtah.solutions.dimacodm.bench import extra_games
from mtah.solutions.dimacodm.deep_data import (
    CLEAN,
    FPS,
    IGNORE,
    PITCH,
    Randomization,
    degrade_window,
    load_clean,
    make_sample,
    sample_window,
)
from mtah.solutions.dimacodm.deep_features import inputs, swap_teams
from mtah.solutions.dimacodm.deep_model import SetTCN, export_numpy
from mtah.solutions.dimacodm.deep_net import DeepConfig, log_odds
from mtah.solutions.dimacodm.model import decode, transition_costs
from mtah.solutions.dimacodm.solution import DimacodmLogisticSolution

app = typer.Typer(add_completion=False)
OUT = REPO_ROOT / "models" / "dimacodm"
FOLDS: list[list[str]] = [
    ["metrica_1", "dfl_J03WMX"],
    ["metrica_2", "dfl_J03WN1"],
    ["metrica_3", "dfl_J03WOH"],
    ["dfl_J03WOY", "dfl_J03WPY"],
    ["dfl_J03WQQ", "dfl_J03WR9"],
]
ALL_GAMES = [g for fold in FOLDS for g in fold]


class WindowStream(IterableDataset):
    """Endless degraded windows of the training games (one RNG stream per worker)."""

    def __init__(
        self,
        names: list[str],
        length: int,
        margin: int,
        seed: int,
        gap: int,
        spell_alpha: float = 0.0,
        rnd: Randomization | None = None,
    ) -> None:
        self.names, self.length, self.margin, self.seed, self.gap = names, length, margin, seed, gap
        self.spell_alpha = spell_alpha
        self.rnd = rnd or Randomization()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        games = [load_clean(name) for name in self.names]
        while True:
            game, start, stop = sample_window(games, rng, self.length, self.margin)
            yield make_sample(game, start, stop, rng, self.rnd, self.gap, self.spell_alpha)


def bundle_inputs(
    name: str, gap: int, limit: int | None = None, profile: str = "simulated"
) -> dict[str, Any]:
    """Model inputs of a benchmark bundle, per half (the SAME degraded data the logistic
    solution is scored on), plus its ground truth. `profile`: which benchmark (bench.PROFILES)."""
    bundle = bench_root(profile) / name
    match = MatchData(bundle, use_cache=False)
    tr = match.tracking
    halves = []
    for _, first, last in match.spec.halves:
        stop = last + 1 if limit is None else min(last + 1, first + limit)
        tokens, mask, globals_ = inputs(
            np.asarray(tr.players[first:stop]),
            np.asarray(tr.slot_team),
            np.asarray(tr.ball[first:stop]),
            match.spec.pitch,
            match.spec.fps,
            gap,
        )
        halves.append((first, stop, tokens, mask, globals_))
    gt = load_gt(bundle / "synthetic" / "gt_full.csv")
    if limit is not None:  # smoke test: grade only the frames we predict
        keep = np.zeros(len(gt), dtype=bool)
        for first, stop, *_ in halves:
            keep[first:stop] = True
        gt = np.where(keep, gt, X).astype(np.int8)
    return {"halves": halves, "gt": gt, "n": len(gt)}


def clean_inputs(name: str, gap: int) -> dict[str, Any]:
    """Like bundle_inputs, from the undegraded tracking (ceiling experiment)."""
    game = load_clean(name)
    rng = np.random.default_rng(0)
    still = replace(CLEAN, flip_p=0.0)
    halves = []
    for first, stop in game.halves:
        raw = degrade_window(game, first, stop, rng, still)
        tokens, mask, globals_ = inputs(
            raw["slots"], raw["slot_team"], raw["ball"], PITCH, FPS, gap
        )
        halves.append((first, stop, tokens, mask, globals_))
    return {"halves": halves, "gt": game.labels, "n": game.n_frames}


def torch_log_odds(
    model: SetTCN, data: dict[str, Any], device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Like deep_net.ensemble_logits for one model: both conventions, teams swapped and back."""
    team = np.zeros(data["n"], dtype=np.float32)
    dead = np.zeros(data["n"], dtype=np.float32)
    for first, stop, tokens, mask, globals_ in data["halves"]:
        st, sg = swap_teams(tokens, globals_)
        teams, deads = [], []
        for convention in (0, 1):
            for tok, glob, sign in ((tokens, globals_, 1.0), (st, sg, -1.0)):
                logits = model.state_logits_half(tok, mask, glob, convention, device)
                t, d = log_odds(logits)
                teams.append(sign * t)
                deads.append(d)
        team[first:stop] = np.mean(teams, axis=0)
        dead[first:stop] = np.mean(deads, axis=0)
    return team, dead


def per_convention(model: SetTCN, data: dict[str, Any], device: str) -> dict[str, np.ndarray]:
    """team{c}, dead{c}, boundary{c} (team-change probability) for conventions 0 and 1, each
    averaged with the team-swapped mirror: what deep_tune reads from oof_conv/."""
    arrays = {"gt": data["gt"]}
    for c in (0, 1):
        team = np.zeros(data["n"], dtype=np.float32)
        dead = np.zeros(data["n"], dtype=np.float32)
        change = np.zeros(data["n"], dtype=np.float32)
        for first, stop, tokens, mask, globals_ in data["halves"]:
            st, sg = swap_teams(tokens, globals_)
            logits, boundary = model.heads_half(tokens, mask, globals_, c, device)
            mirror_logits, mirror_boundary = model.heads_half(st, mask, sg, c, device)
            t1, d1 = log_odds(logits)
            t2, d2 = log_odds(mirror_logits)
            team[first:stop], dead[first:stop] = (t1 - t2) / 2, (d1 + d2) / 2
            if boundary is not None and mirror_boundary is not None:
                probability = (1 / (1 + np.exp(-boundary)) + 1 / (1 + np.exp(-mirror_boundary))) / 2
                change[first:stop] = probability
        arrays[f"team{c}"], arrays[f"dead{c}"], arrays[f"boundary{c}"] = team, dead, change
    return arrays


def decode_scores(
    team: np.ndarray, dead: np.ndarray, gt: np.ndarray, params: dict[str, Any]
) -> dict[str, float]:
    graded = gt != X
    edges = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    runs = [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]
    costs = transition_costs(
        params["switch_cost"],
        params["stop_cost"],
        params["restart_cost"],
        params["same_team_restart_cost"],
    )
    codes = decode(
        np.round(team, 4),
        np.round(dead, 4),
        runs,
        len(gt),
        params["team_scale"],
        params["dead_scale"],
        params["dead_bias"],
        costs,
    )
    codes[~graded] = X
    result = evaluate(codes, gt)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
    }


def train_fold(
    k: int,
    cfg: DeepConfig,
    *,
    steps: int,
    batch: int,
    lr: float,
    workers: int,
    device: str,
    out: Path,
    seed: int,
    eval_every: int,
    smoke: bool,
    centre: int = 768,
    compile_model: bool = True,
    init: Path | None = None,
    extra: bool = False,
    spell_alpha: float = 0.0,
    clean: bool = False,
    profile: str = "simulated",
) -> dict[str, Any]:
    held_out = FOLDS[k] if k < len(FOLDS) else []  # fold 5: every game
    train_names = [g for g in ALL_GAMES if g not in held_out]
    if extra:  # training-only games (SkillCorner): the benchmark scores stay comparable
        train_names += extra_games()
    margin = cfg.receptive_field + 1
    # Only the centre of a window (full receptive field on both sides) is supervised: a longer
    # centre spends less of each window (GPU time and degradation) on unlabelled context.
    length = 2 * margin + (64 if smoke else centre)
    gap = int(DimacodmLogisticSolution.defaults["ball_gap_frames"])
    torch.manual_seed(seed + k)
    model = SetTCN(cfg).to(device)
    if init is not None:  # continue from an exported fold (warm restart: new schedule, new data)
        with np.load(init / f"fold{k}.npz") as data:
            weights = {
                key: torch.from_numpy(data[key]) for key in data.files if key != "__config__"
            }
        model.load_state_dict(weights)
    # The training forward only (fixed shapes): evaluation and export use `model` itself, which
    # shares the parameters. ~2x on a GPU; skipped on CPU (needs a C++ toolchain).
    run = torch.compile(model) if compile_model and device.startswith("cuda") else model
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, steps // 20)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"fold{k}.pt"
    start_step, history = 0, []
    if checkpoint.is_file() and not smoke:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, history = state["step"], state["history"]
    loader = DataLoader(
        WindowStream(
            train_names, length, margin, seed * 100 + k, gap, spell_alpha, CLEAN if clean else None
        ),
        batch_size=batch,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    limit = 6000 if smoke else None
    val = {
        name: clean_inputs(name, gap) if clean else bundle_inputs(name, gap, limit, profile)
        for name in held_out
    }
    params = dict(DimacodmLogisticSolution.defaults)
    amp = device.startswith("cuda")
    supervised = slice(margin, length - margin)
    began, seen = time.perf_counter(), 0
    iterator = iter(loader)
    for step in range(start_step, steps):
        model.train()
        sample = next(iterator)
        tensors = {key: value.to(device, non_blocking=True) for key, value in sample.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            state_logits, ball, pointer, boundary = run(
                tensors["tokens"], tensors["mask"], tensors["globals"], tensors["convention"]
            )
        state_logits, ball, pointer = state_logits.float(), ball.float(), pointer.float()
        targets = tensors["state"][:, supervised].reshape(-1)
        frame_weight = tensors["state_weight"][:, supervised].reshape(-1)
        loss_state = (
            F.cross_entropy(
                state_logits[:, supervised].reshape(-1, 3),
                targets,
                ignore_index=IGNORE,
                reduction="none",
            )
            * frame_weight
        ).sum() / (frame_weight * (targets != IGNORE)).sum().clamp(min=1e-6)
        loss_pointer = F.cross_entropy(
            pointer[:, supervised].reshape(-1, pointer.shape[-1]),
            tensors["possessor"][:, supervised].reshape(-1),
            ignore_index=IGNORE,
        )
        weight = tensors["ball_weight"][:, supervised]
        loss_ball = (
            F.smooth_l1_loss(
                ball[:, supervised],
                tensors["ball_target"][:, supervised],
                reduction="none",
                beta=0.05,
            ).sum(-1)
            * weight
        ).sum() / weight.sum().clamp(min=1.0)
        loss = loss_state + 0.3 * torch.nan_to_num(loss_pointer) + 0.5 * loss_ball
        if boundary is not None:
            soft = tensors["boundary"][:, supervised]
            known = tensors["boundary_mask"][:, supervised]
            loss_boundary = F.binary_cross_entropy_with_logits(
                boundary.float()[:, supervised],
                soft,
                weight=known * (1 + 9 * soft),
                reduction="sum",
            ) / known.sum().clamp(min=1.0)
            loss = loss + loss_boundary
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        seen += batch
        if (step + 1) % max(1, eval_every // 10) == 0:
            rate = seen / (time.perf_counter() - began)
            typer.echo(
                json.dumps(
                    {
                        "fold": k,
                        "step": step + 1,
                        "loss": round(loss.item(), 4),
                        "state": round(loss_state.item(), 4),
                        "ball": round(loss_ball.item(), 4),
                        "lr": round(float(scheduler.get_last_lr()[0]), 7),
                        "windows_per_s": round(rate, 1),
                    }
                )
            )
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            scores = {}
            for name, data in val.items():
                team, dead = torch_log_odds(model, data, device)
                scores[name] = decode_scores(team, dead, data["gt"], params)
            history.append({"step": step + 1, "scores": scores})
            typer.echo(json.dumps({"fold": k, "step": step + 1, "held_out": scores}))
            if not smoke:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step + 1,
                        "history": history,
                        "config": asdict(cfg),
                    },
                    checkpoint,
                )
    export_numpy(model, out / f"fold{k}.npz")
    oof = out / "oof"
    oof.mkdir(exist_ok=True)
    for name, data in val.items():
        team, dead = torch_log_odds(model, data, device)
        np.savez_compressed(oof / f"{name}.npz", team=team, dead=dead, gt=data["gt"])
        by_convention = out / "oof_conv"
        by_convention.mkdir(exist_ok=True)
        arrays: dict[str, Any] = per_convention(model, data, device)
        np.savez_compressed(by_convention / f"{name}.npz", **arrays)
    summary = {
        "fold": k,
        "held_out": held_out,
        "config": asdict(cfg),
        "steps": steps,
        "batch": batch,
        "centre": centre,
        "lr": lr,
        "init": None if init is None else init.name,
        "extra": extra,
        "spell_alpha": spell_alpha,
        "clean": clean,
        "profile": profile,
        "history": history,
    }
    (out / f"fold{k}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@app.command()
def main(
    fold: Annotated[
        list[int] | None, typer.Option(help="fold(s) 0-4, default all; 5 = all ten games")
    ] = None,
    steps: Annotated[int, typer.Option()] = 12_000,
    batch: Annotated[int, typer.Option()] = 16,
    lr: Annotated[float, typer.Option()] = 5e-4,
    workers: Annotated[int, typer.Option()] = 8,
    device: Annotated[str, typer.Option(help="cuda | cpu")] = "cuda",
    out: Annotated[Path, typer.Option()] = OUT,
    seed: Annotated[int, typer.Option()] = 0,
    eval_every: Annotated[int, typer.Option()] = 2_000,
    centre: Annotated[int, typer.Option(help="supervised frames per training window")] = 768,
    compile_model: Annotated[bool, typer.Option("--compile/--no-compile")] = True,
    init: Annotated[Path | None, typer.Option(help="folder of fold*.npz to continue from")] = None,
    extra: Annotated[bool, typer.Option(help="also train on the SkillCorner games")] = False,
    boundary: Annotated[bool, typer.Option(help="ali_seg: predict the team changes too")] = False,
    spell_alpha: Annotated[float, typer.Option(help="frame weight = run length ** -alpha")] = 0.0,
    clean: Annotated[bool, typer.Option(help="ceiling experiment: no degradation at all")] = False,
    profile: Annotated[
        str, typer.Option(help="benchmark the held-out games are scored on: simulated | real-like")
    ] = "simulated",
    context: Annotated[
        int, typer.Option(help="extra dilations: 0 = +/-10 s of context, 1 = +/-20 s, 2 = +/-41 s")
    ] = 0,
    smoke: Annotated[bool, typer.Option(help="tiny model, few steps, CPU-friendly")] = False,
    do_tune: Annotated[bool, typer.Option("--tune", help="decoder search on out-of-fold")] = False,
) -> None:
    if do_tune:  # CPU only: deep_tune needs no torch
        from mtah.solutions.dimacodm.deep_tune import tune

        typer.echo(json.dumps(tune(out), indent=2))
        return
    cfg = DeepConfig(boundary_head=boundary)
    cfg = replace(cfg, dilations=cfg.dilations + tuple(256 * 2**i for i in range(context)))
    if smoke:
        cfg = replace(cfg, d_token=16, heads=2, set_layers=1, d_frame=32, dilations=(1, 2, 4, 8))
        steps, batch, workers, eval_every = min(steps, 60), min(batch, 4), 0, 30
    for k in fold if fold is not None else range(len(FOLDS)):
        summary = train_fold(
            k,
            cfg,
            steps=steps,
            batch=batch,
            lr=lr,
            workers=workers,
            device=device,
            out=out,
            seed=seed,
            eval_every=eval_every,
            smoke=smoke,
            centre=centre,
            compile_model=compile_model,
            init=init,
            extra=extra,
            spell_alpha=spell_alpha,
            clean=clean,
            profile=profile,
        )
        typer.echo(
            json.dumps({"fold": k, "final": summary["history"][-1] if summary["history"] else None})
        )


if __name__ == "__main__":  # pragma: no cover
    app()
