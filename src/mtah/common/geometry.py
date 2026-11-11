"""Distances between the ball and the players, in metres."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mtah.data.atd import Tracking

NOBODY = -1


def nearest_player(
    tracking: Tracking, pitch: tuple[float, float], chunk: int = 50_000
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Per frame: (team of the nearest tracked player, distance in m, slot index).

    team is NOBODY (-1), distance NaN and slot -1 when the ball or every player is missing.
    The arithmetic mirrors the official baseline exactly (same float64 operations, first slot wins
    ties), which is what lets the dummy solution reproduce baseline_smooth50.csv byte for byte.
    """
    length, width = pitch
    n = tracking.n_frames
    team = np.full(n, NOBODY, dtype=np.int8)
    dist = np.full(n, np.nan)
    slot = np.full(n, -1, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        ball = np.asarray(tracking.ball[start:stop])
        players = np.asarray(tracking.players[start:stop])
        ball_x = ball[:, 0] * length
        ball_y = ball[:, 1] * width
        dx = players[:, :, 0] * length - ball_x[:, None]
        dy = players[:, :, 1] * width - ball_y[:, None]
        d2 = dx * dx + dy * dy
        d2 = np.where(np.isnan(players[:, :, 0]), np.inf, d2)
        best = np.argmin(d2, axis=1)
        best_d2 = d2[np.arange(stop - start), best]
        ok = ~np.isnan(ball[:, 0]) & np.isfinite(best_d2)
        rows = np.arange(start, stop)[ok]
        team[rows] = tracking.slot_team[best[ok]]
        dist[rows] = np.sqrt(best_d2[ok])
        slot[rows] = best[ok]
    return team, dist, slot


def team_min_distance(tracking: Tracking, pitch: tuple[float, float]) -> npt.NDArray[np.float64]:
    """(n, 2) distance in metres from the ball to the nearest home / away player (NaN if unknown)."""
    scale = np.asarray(pitch, dtype=np.float64)
    ball = np.asarray(tracking.ball) * scale
    players = np.asarray(tracking.players) * scale
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    out = np.full((tracking.n_frames, 2), np.nan)
    for side in (0, 1):
        cols = tracking.slot_team == side
        if cols.any():
            sub = d[:, cols]
            has = ~np.isnan(sub).all(axis=1)
            out[has, side] = np.nanmin(sub[has], axis=1)
    return out
