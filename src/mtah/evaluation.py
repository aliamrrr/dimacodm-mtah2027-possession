"""Scoring with the official metric library, plus team-side helpers (comparisons, agreement)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mtah import official
from mtah.common.segments import fill_abstentions
from mtah.labels import Codes, X
from mtah.paths import PUBLIC_GT
from mtah.spec import MatchSpec

GT_CHOICES = ("public", "synthetic", "synthetic-public")


def resolve_gt(gt: str, data_root: Path | None) -> Path:
    """'public' (official 5 min), 'synthetic' / 'synthetic-public' (fake bundle), or a path."""
    if gt == "public":
        return PUBLIC_GT
    if gt in ("synthetic", "synthetic-public"):
        if data_root is None:
            raise ValueError(f"--gt {gt} needs a synthetic data directory")
        name = "gt_full.csv" if gt == "synthetic" else "gt_public.csv"
        path = Path(data_root) / "synthetic" / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: is {data_root} a synthetic bundle?")
        return path
    path = Path(gt)
    if not path.is_file():
        raise FileNotFoundError(f"ground truth not found: {gt} (choices: {GT_CHOICES} or a path)")
    if "possession_gt" in path.name and "public" not in path.name:
        raise PermissionError("refusing a file named like the PRIVATE ground truth (rule 2)")
    return path


def load_gt(path: Path) -> Codes:
    pm = official.metrics()
    return np.asarray(pm.label_codes(pm.read_label_csv(path)), dtype=np.int8)


def evaluate(codes: Codes, gt_codes: Codes, bootstrap: int = 0) -> dict[str, Any]:
    """The full official result dict (possession_metrics.evaluate)."""
    return dict(
        official.metrics().evaluate(np.asarray(gt_codes), np.asarray(codes), bootstrap_n=bootstrap)
    )


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    loose = result["transitions"][max(result["transitions"])]
    return {
        "spell_f1": round(result["spell_f1_mean"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "spell_ci": (
            None
            if result["spell_bootstrap_low"] is None
            else [round(result["spell_bootstrap_low"], 4), round(result["spell_bootstrap_high"], 4)]
        ),
        "n_pred_spells": result["n_pred_spells"],
        "n_gt_spells": result["n_gt_spells"],
        "mean_spell_s": round(result["mean_spell_duration_s"], 2),
        "gt_mean_spell_s": round(result["gt_mean_spell_duration_s"], 2),
        "alive_dead_f1": round(result["alive_dead_f1"], 4),
        "team_acc_both_alive": round(result["team_acc_both_alive"], 4),
        "trans_f1_25": round(loose["f1"], 4),
        "mean_latency": None if loose["mean_latency"] is None else round(loose["mean_latency"], 1),
        "x_on_graded": result["n_abstain_on_graded"],
        "n_graded": result["n_graded"],
    }


def score(codes: Codes, gt_path: Path, bootstrap: int = 0) -> dict[str, Any]:
    return summarize(evaluate(codes, load_gt(gt_path), bootstrap))


def agreement(reference: Codes, other: Codes, spec: MatchSpec) -> dict[str, float]:
    """How much two submissions agree on the graded frames, without any ground truth.

    `spell_f1` treats `reference` (abstentions filled, X outside the halves) as if it were the truth:
    two solutions at 0.9 make the same possessions; at 0.3 they disagree on most of them -- which is
    where watching the video pays off.
    """
    graded = spec.graded_mask()
    ref = fill_abstentions(reference, spec)
    oth = fill_abstentions(other, spec)
    as_gt = ref.copy()
    as_gt[~graded] = X
    result = official.metrics().evaluate(as_gt, oth)
    return {
        "frame_agreement": float(np.mean(ref[graded] == oth[graded])),
        "spell_f1": float(result["spell_f1_mean"]),
    }
