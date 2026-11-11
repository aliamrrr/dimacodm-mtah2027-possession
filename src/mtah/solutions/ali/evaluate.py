"""Score solutions on the benchmark of OTHER games (see bench.py) with the official metric.

This is where versions are compared and parameters chosen, legitimately, since the labels come
from other matches. Never on the public 5 minutes of BAR-ATH (RULES.md).

    python -m mtah.solutions.ali.evaluate dummy ali               # every bundle under <cache>/bench
    python -m mtah.solutions.ali.evaluate ali -p switch_cost=25   # one parameter changed
    python -m mtah.solutions.ali.evaluate ali --synthetic data_synthetic
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from mtah.config import load_config, parse_overrides
from mtah.data.match import MatchData
from mtah.evaluation import evaluate, load_gt
from mtah.pipeline import predict
from mtah.solutions.ali.bench import iter_bundles

console = Console(width=150)
app = typer.Typer(add_completion=False)


def score_bundle(solution: str, bundle: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    config = load_config(solution, overrides=overrides)
    match = MatchData(bundle, use_cache=False)  # 10 bundles x 140 MB of cache: no
    started = time.perf_counter()
    codes = predict(config, match)
    runtime = time.perf_counter() - started
    result = evaluate(codes, load_gt(bundle / "synthetic" / "gt_full.csv"))
    per_class = result["per_class"]
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "f1_d": per_class["D"]["f1"],
        "alive_dead_f1": result["alive_dead_f1"],
        "team_acc": result["team_acc_both_alive"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
        "runtime_s": runtime,
    }


def dataset_groups(bundles: list[Path]) -> list[tuple[str, list[int]]]:
    """Row indices per source, then the balanced mean's members (all)."""
    metrica = [i for i, b in enumerate(bundles) if b.name.startswith("metrica_")]
    dfl = [i for i, b in enumerate(bundles) if b.name.startswith("dfl_")]
    return [("mean metrica", metrica), ("mean dfl", dfl), ("mean all", list(range(len(bundles))))]


def run_table(
    solutions: list[str], bundles: list[Path], overrides: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    table = Table(title="benchmark (other games, official metric)", title_justify="left")
    for col in (
        "solution",
        "bundle",
        "spellF1",
        "macroF1",
        "F1 D",
        "alive/dead",
        "team",
        "poss",
        "s",
    ):
        table.add_column(col, justify="left" if col in ("solution", "bundle") else "right")
    results: dict[str, list[dict[str, Any]]] = {}
    for solution in solutions:
        rows = []
        for bundle in bundles:
            r = score_bundle(solution, bundle, overrides if solution != "dummy" else {})
            rows.append(r)
            table.add_row(
                solution,
                bundle.name,
                f"{r['spell_f1']:.4f}",
                f"{r['macro_f1']:.4f}",
                f"{r['f1_d']:.3f}",
                f"{r['alive_dead_f1']:.3f}",
                f"{r['team_acc']:.3f}",
                f"{r['n_pred']}/{r['n_gt']}",
                f"{r['runtime_s']:.0f}",
            )
        for label, picked in dataset_groups(bundles):
            if not picked:
                continue
            spell = float(np.mean([rows[i]["spell_f1"] for i in picked]))
            macro = float(np.mean([rows[i]["macro_f1"] for i in picked]))
            table.add_row(solution, f"[bold]{label}[/]", f"[bold]{spell:.4f}[/]", f"{macro:.4f}")
        results[solution] = rows
    console.print(table)
    return results


@app.command()
def main(
    solutions: Annotated[list[str], typer.Argument(help="solutions to score")],
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="key=value")] = None,
    root: Annotated[Path | None, typer.Option(help="default: <cache>/bench")] = None,
    synthetic: Annotated[Path | None, typer.Option(help="also score this synthetic bundle")] = None,
) -> None:
    bundles = list(iter_bundles(root))
    if synthetic is not None:
        bundles.append(synthetic)
    if not bundles:
        raise typer.BadParameter("no bundle: run `python -m mtah.solutions.ali.bench` first")
    run_table(solutions, bundles, parse_overrides(param or []))


if __name__ == "__main__":  # pragma: no cover
    app()
