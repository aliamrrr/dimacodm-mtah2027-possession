"""Two small logistic models (numpy only) and the 4-state possession decoder.

The models are trained on the benchmark of OTHER games only (train.py), never on BAR-ATH: their
coefficients live in weights.json next to this file, so the submission run needs no training and
the training itself is reproducible from public data.

Decoder states: H, A, D after H, D after A. Remembering who had the ball before a stoppage lets
the restart prefer the other team, which is what football mostly does (a ball out goes to the
opponent of the last toucher: about 70 % of restarts on the benchmark).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mtah.common.smoothing import viterbi
from mtah.labels import A, D, H

FloatArray = npt.NDArray[np.float64]
WEIGHTS_PATH = Path(__file__).with_name("weights.json")
FORBIDDEN = 1e9

# Decoder states and the label each one writes.
S_H, S_A, S_DH, S_DA = 0, 1, 2, 3
STATE_LABEL = np.array([H, A, D, D], dtype=np.int8)


def fit_logistic(
    x: FloatArray, y: npt.NDArray[np.bool_], l2: float = 1.0, intercept: bool = True
) -> FloatArray:
    """L2-regularised logistic regression by Newton's method. Returns [w..., b] (b = 0 if no
    intercept). Deterministic: no random initialisation, no stochastic optimiser."""
    design = np.column_stack([x, np.ones(len(x))]) if intercept else x
    w = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], l2)
    if intercept:
        penalty[-1] = 0.0
    target = y.astype(np.float64)
    for _ in range(100):
        z = np.clip(design @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (p - target) + penalty * w
        hessian = (design * (p * (1 - p))[:, None]).T @ design + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return np.append(w, 0.0) if not intercept else w


def logit(x: FloatArray, coef: FloatArray) -> FloatArray:
    return np.asarray(x @ coef[:-1] + coef[-1], dtype=np.float64)


@dataclass(frozen=True)
class Weights:
    dead: list[float]  # coefficients of DEAD_FEATURES, then the intercept
    team: list[float]  # coefficients of TEAM_FEATURES, then 0 (no intercept)
    dead_features: list[str]
    team_features: list[str]
    trained_on: list[str]  # benchmark bundles (other games)
    note: str = "trained by `python -m mtah.solutions.ali.train` on other games only"

    def save(self, path: Path = WEIGHTS_PATH) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(self), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> Weights:
        with open(path, encoding="utf-8") as handle:
            return cls(**json.load(handle))


def transition_costs(
    switch_cost: float, stop_cost: float, restart_cost: float, same_team_restart_cost: float
) -> FloatArray:
    """cost[i, j] of going from state i to state j (0 on the diagonal)."""
    cost = np.full((4, 4), FORBIDDEN)
    np.fill_diagonal(cost, 0.0)
    cost[S_H, S_A] = cost[S_A, S_H] = switch_cost
    cost[S_H, S_DH] = cost[S_A, S_DA] = stop_cost
    cost[S_DH, S_A] = cost[S_DA, S_H] = restart_cost
    cost[S_DH, S_H] = cost[S_DA, S_A] = restart_cost + same_team_restart_cost
    return cost


def viterbi_varying(
    emission: FloatArray,
    costs: FloatArray,
    stop_bonus: FloatArray,
    start_bonus: FloatArray,
    switch_bonus: FloatArray | None = None,
) -> npt.NDArray[np.int64]:
    """Viterbi over the 4 states with costs that move in time: entering a stoppage at frame t
    costs `costs - stop_bonus[t]`, leaving one `costs - start_bonus[t]`, changing team in open
    play `costs - switch_bonus[t]`. Ties resolve to the lowest state index, deterministically
    (numpy argmax)."""
    n = len(emission)
    stop_mask = np.zeros((4, 4))
    stop_mask[S_H, S_DH] = stop_mask[S_A, S_DA] = 1.0
    start_mask = np.zeros((4, 4))
    start_mask[[S_DH, S_DH, S_DA, S_DA], [S_H, S_A, S_A, S_H]] = 1.0
    switch_mask = np.zeros((4, 4))
    switch_mask[S_H, S_A] = switch_mask[S_A, S_H] = 1.0
    switch = np.zeros(n) if switch_bonus is None else switch_bonus
    score = emission[0].copy()
    back = np.empty((n, 4), dtype=np.int8)
    back[0] = np.arange(4)
    columns = np.arange(4)
    for t in range(1, n):
        candidates = score[:, None] - (
            costs
            - stop_bonus[t] * stop_mask
            - start_bonus[t] * start_mask
            - switch[t] * switch_mask
        )
        best = np.argmax(candidates, axis=0)
        back[t] = best
        score = candidates[best, columns] + emission[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def controlled_possession(
    codes: npt.NDArray[np.int8], halves: list[tuple[int, int]], min_frames: int
) -> npt.NDArray[np.int8]:
    """The labels with every touch without control given back to the team that had the ball.

    CHALLENGE.md: a change of possession needs CONTROLLED possession; "a simple touch of the ball
    without control", "a deflection of a pass that does not intercept it" and "an aerial duel won
    without subsequent control" are not one. The rule here: a team spell shorter than
    `min_frames` that follows a spell of the other team (ball alive in between) and is followed
    either by that other team again or by a stoppage did not establish control, so its frames
    stay with the other team. A short spell that follows a stoppage is kept: a restart has a
    clear possessor. Spells are examined shortest first (the earliest on a tie), each half on its
    own; integers only, hence deterministic. `min_frames` <= 0 returns the labels unchanged.
    """
    out = codes.copy()
    if min_frames <= 0:
        return out
    for start, stop in halves:
        window = out[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)])).tolist()
        runs = [[bounds[i], bounds[i + 1], int(window[bounds[i]])] for i in range(len(bounds) - 1)]
        while True:
            best = -1
            for i in range(1, len(runs) - 1):
                first, last, code = runs[i]
                if code not in (H, A) or last - first >= min_frames:
                    continue
                other = A if code == H else H
                if runs[i - 1][2] != other or runs[i + 1][2] not in (other, D):
                    continue
                if best < 0 or last - first < runs[best][1] - runs[best][0]:
                    best = i
            if best < 0:
                break
            runs[best - 1][1] = runs[best][1]  # the previous team keeps the ball
            del runs[best]
            if runs[best][2] == runs[best - 1][2]:  # ... and its next spell is the same one
                runs[best - 1][1] = runs[best][1]
                del runs[best]
        for first, last, code in runs:
            window[first:last] = code
    return out


def decode(
    team_logit: FloatArray,
    dead_logit: FloatArray,
    halves: list[tuple[int, int]],
    n_frames: int,
    team_scale: float,
    dead_scale: float,
    dead_bias: float,
    costs: FloatArray,
    stop_bonus: FloatArray | None = None,
    start_bonus: FloatArray | None = None,
    switch_bonus: FloatArray | None = None,
) -> npt.NDArray[np.int8]:
    """Labels per frame (H/A/D inside the graded halves, D elsewhere: the caller writes X there).

    Emissions are scaled log-odds: +-team_scale * logit / 2 for H / A, and
    dead_scale * (dead logit - dead_bias) for both dead states. Each half is decoded on its own,
    so nothing carries over half-time. With bonuses, the stoppage, restart and team-change costs
    vary in time.
    """
    out = np.full(n_frames, D, dtype=np.int8)
    half_team = 0.5 * team_scale * team_logit
    dead = dead_scale * (dead_logit - dead_bias)
    for start, stop in halves:
        emission = np.column_stack(
            [half_team[start:stop], -half_team[start:stop], dead[start:stop], dead[start:stop]]
        )
        if stop_bonus is None and start_bonus is None and switch_bonus is None:
            path = viterbi(emission, costs)
        else:
            none = np.zeros(stop - start)
            path = viterbi_varying(
                emission,
                costs,
                none if stop_bonus is None else stop_bonus[start:stop],
                none if start_bonus is None else start_bonus[start:stop],
                None if switch_bonus is None else switch_bonus[start:stop],
            )
        out[start:stop] = STATE_LABEL[path]
    return out
