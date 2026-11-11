"""Temporal decoding: from noisy per-frame evidence to possession-shaped label sequences."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mtah.labels import A, Codes, D, H

NOBODY = -1


def majority_vote(team: npt.NDArray[np.integer], window: int) -> Codes:
    """Centred majority vote over +/- `window` frames -- exactly the official baseline's `smooth`.

    `team` holds H (0), A (1) or NOBODY (-1) per frame. Output: H/A where any vote exists in the
    window (ties -> H), D where the window has no vote at all. window=0 means no smoothing.
    """
    team = np.asarray(team)
    if window <= 0:
        return np.where(team == H, H, np.where(team == A, A, D)).astype(np.int8)
    home_cum = np.concatenate(([0], np.cumsum(team == H)))
    away_cum = np.concatenate(([0], np.cumsum(team == A)))
    index = np.arange(len(team))
    low = np.maximum(0, index - window)
    high = np.minimum(len(team), index + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]
    out = np.full(len(team), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def viterbi(
    log_emission: npt.NDArray[np.floating],
    switch_cost: float | npt.NDArray[np.floating],
    log_prior: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.int64]:
    """Most likely state path of a first-order chain.

    log_emission: (n, K) per-frame log-likelihood of each state (NaN treated as 0 = no evidence).
    switch_cost:  scalar penalty for any state change, or a (K, K) matrix of penalties
                  (cost[i, j] = penalty for going from i to j; diagonal should be 0).
    Larger penalties -> fewer, longer segments. Returns (n,) state indices.
    """
    emission = np.nan_to_num(np.asarray(log_emission, dtype=np.float64), nan=0.0)
    n, k = emission.shape
    if np.isscalar(switch_cost):
        cost = np.full((k, k), float(switch_cost))  # type: ignore[arg-type]
        np.fill_diagonal(cost, 0.0)
    else:
        cost = np.asarray(switch_cost, dtype=np.float64)
    score = emission[0] + (np.zeros(k) if log_prior is None else np.asarray(log_prior))
    back = np.empty((n, k), dtype=np.int8)
    back[0] = np.arange(k)
    for t in range(1, n):
        candidates = score[:, None] - cost  # (from, to)
        best_from = np.argmax(candidates, axis=0)
        back[t] = best_from
        score = candidates[best_from, np.arange(k)] + emission[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
