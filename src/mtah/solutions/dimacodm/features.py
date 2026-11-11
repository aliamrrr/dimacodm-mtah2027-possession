"""Per-frame evidence for the `dimacodm_logistic` solution. Pure functions: arrays in, arrays out, no state.

Everything is computed from positions only (metres), never from a field whose definition we cannot
check on the real data (the ATD speed column, the third ball field): the same code then means the
same thing on BAR-ATH, on the benchmark of other games and on the synthetic match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mtah.common.ball import interpolate_gaps
from mtah.labels import A, H

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def rolling_mean(x: FloatArray, half_window: int) -> FloatArray:
    """Centred mean over +/- half_window frames, ignoring NaN (NaN where the window is empty)."""
    x = np.asarray(x, dtype=np.float64)
    if half_window <= 0:
        return x.copy()
    finite = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(x))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(x), index + half_window + 1)
    n = count[high] - count[low]
    out = np.full(len(x), np.nan)
    ok = n > 0
    out[ok] = (total[high] - total[low])[ok] / n[ok]
    return out


def nan_quantiles(x: FloatArray, quantiles: tuple[float, ...]) -> FloatArray:
    """Row-wise quantiles of a 2-D array ignoring NaN: (n, m) -> (n, len(quantiles)).

    Bit-for-bit what np.nanquantile(x, quantiles, axis=1).T returns (linear method, numpy's
    two-sided lerp), but vectorised: np.nanquantile loops over the rows in Python, which cost
    ~145 000 calls per match. A row without any value gives NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    n, m = x.shape
    count = (~np.isnan(x)).sum(axis=1)
    ordered = np.sort(np.where(np.isnan(x), np.inf, x), axis=1)  # NaN last
    rows = np.arange(n)
    out = np.full((n, len(quantiles)), np.nan)
    has = count > 0
    for j, q in enumerate(quantiles):
        virtual = q * (count - 1).astype(np.float64)
        previous = np.floor(virtual)
        gamma = virtual - previous
        lo = np.clip(previous.astype(np.int64), 0, m - 1)
        hi = np.clip(lo + 1, 0, np.maximum(count - 1, 0))
        a, b = ordered[rows, lo], ordered[rows, hi]
        with np.errstate(invalid="ignore"):
            diff = b - a
            value = np.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)
        out[has, j] = value[has]
    return out


def ball_track(ball_norm: FloatArray, pitch: tuple[float, float], max_gap: int) -> FloatArray:
    """Ball in metres, gaps of at most `max_gap` frames bridged linearly."""
    return interpolate_gaps(np.asarray(ball_norm, dtype=np.float64), max_gap) * np.asarray(pitch)


def velocity(xy_m: FloatArray, fps: int, half_window: int = 2) -> FloatArray:
    """(n, 2) m/s by a centred difference over +/- half_window frames (NaN near gaps)."""
    out = np.full_like(xy_m, np.nan)
    k = half_window
    out[k:-k] = (xy_m[2 * k :] - xy_m[: -2 * k]) * fps / (2.0 * k)
    return out


def out_of_play(ball_m: FloatArray, pitch: tuple[float, float], margin: float) -> BoolArray:
    """Ball seen beyond a touchline or goal line by more than `margin` metres."""
    x, y = ball_m[:, 0], ball_m[:, 1]
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def halves_separation(
    players_m: FloatArray, slot_team: npt.NDArray[np.int8], length: float
) -> FloatArray:
    """How cleanly the two teams sit in opposite halves (1.0 = every player on his own side).

    That is the kick-off formation, which no phase of play reproduces: it marks the dead ball
    before each kick-off, the second half's included, with no frame number written anywhere.
    """
    left = players_m[:, :, 0] < length / 2.0
    seen = ~np.isnan(players_m[:, :, 0])
    out = np.zeros(len(players_m))
    for home_left in (True, False):
        score = np.ones(len(players_m))
        for side in (H, A):
            cols = slot_team == side
            wanted = left[:, cols] if (side == H) == home_left else ~left[:, cols]
            n_seen = seen[:, cols].sum(axis=1)
            share = np.where(
                n_seen > 0, (wanted & seen[:, cols]).sum(axis=1) / np.maximum(n_seen, 1), 0.0
            )
            score *= share
        out = np.maximum(out, score)
    return out


# -- the feature matrices the models read ---------------------------------------------------

DEAD_FEATURES = (
    "log_v50_s25",  # players' median speed, +/- 1 s, relative to this match's own median
    "log_v75_s50",  # 75th percentile, +/- 2 s
    "log_v90_s25",  # 90th percentile, +/- 1 s
    "log_v75_past2s",  # 75th percentile over the 2 s before the frame
    "log_v75_next2s",  # ... and the 2 s after it (players stop after a whistle, start at a restart)
    "out_of_play",  # share of +/- 0.5 s with the ball seen beyond a line
    "players_off_pitch",  # players more than 0.5 m beyond a line, +/- 1 s (throw-ins, fetching)
    "kickoff_formation",  # max over +/- 1 s of the halves separation, when it is >= 0.9
    "log_v75_past1s",  # the same speed trend at a shorter scale ...
    "log_v75_next1s",
    "log_v75_past4s",  # ... and a longer one
    "log_v75_next4s",
    "out_of_play_2s",  # out_of_play over +/- 2 s
)
# Deliberately NOT dead-ball evidence: whether the ball is seen and whether it lies still. On the
# benchmark of other games the ball had to be simulated during stoppages, so a model leaning on
# those cues would learn our simulation rather than football (ablation: they add 0.005 spellF1
# there), and on BAR-ATH a goalkeeper holding the ball would look like a stoppage.
TEAM_FEATURES = (
    "near_0.5m",  # exp(-d_home / 0.5) - exp(-d_away / 0.5), d = ball to nearest player
    "near_1m",
    "near_2m",
    "near_4m",
    "count_2m",  # home players within 2 m - away players within 2 m
    "count_5m",
    "control_1m",  # exp(-d / 1) * exp(-|v_ball - v_player| / 3): the ball moves WITH the player
    "touch",  # exp(-d / 1.5) * min(1, ball acceleration / 60 m/s2): a kick or deflection there
    "near_1m_past",  # near_1m over the 0.5 s before the frame
    "near_1m_next",  # ... and the 0.5 s after it
    "near_1m_past_0.25s",  # the same context at other scales
    "near_1m_next_0.25s",
    "near_1m_past_1s",
    "near_1m_next_1s",
    "control_past_0.5s",  # control_1m over the 0.5 s before / after
    "control_next_0.5s",
)


@dataclass(frozen=True)
class FrameFeatures:
    dead: FloatArray  # (n, len(DEAD_FEATURES))
    team: FloatArray  # (n, len(TEAM_FEATURES)), antisymmetric in the two teams
    seen: BoolArray  # (n,) ball seen (after bridging short gaps)
    speed_trend: FloatArray  # (n,) log(v75 next 2 s / v75 past 2 s): < 0 stopping, > 0 starting


def _relative_log(x: FloatArray, graded: BoolArray) -> FloatArray:
    """log(x / median of x over the graded frames): removes the level of play of a given match."""
    finite = np.isfinite(x[graded])
    reference = float(np.median(x[graded][finite])) if finite.any() else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(np.maximum(x, 1e-3) / max(reference, 1e-3))


def rolling_max(x: FloatArray, half_window: int) -> FloatArray:
    """Centred max over +/- half_window frames (x must be finite)."""
    padded = np.pad(x, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    return windows.max(axis=1)


def past_mean(x: FloatArray, window: int) -> FloatArray:
    """Mean over the `window` frames just before each frame (NaN ignored, NaN if none)."""
    x = np.asarray(x, dtype=np.float64)
    finite = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(x))
    low = np.maximum(0, index - window)
    n = count[index] - count[low]
    out = np.full(len(x), np.nan)
    ok = n > 0
    out[ok] = (total[index] - total[low])[ok] / n[ok]
    return out


def next_mean(x: FloatArray, window: int) -> FloatArray:
    """Mean over the `window` frames just after each frame (NaN ignored, NaN if none)."""
    return past_mean(np.asarray(x)[::-1], window)[::-1]


def _nearest_per_team(
    dist: FloatArray, slot_team: npt.NDArray[np.int8]
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """(distance (n, 2), slot index (n, 2) or -1) of the nearest home / away player."""
    n = len(dist)
    near = np.full((n, 2), np.inf)
    index = np.full((n, 2), -1, dtype=np.int64)
    for side in (H, A):
        sub = np.where(slot_team[None, :] == side, dist, np.nan)
        has = ~np.isnan(sub).all(axis=1)
        if has.any():
            best = np.nanargmin(sub[has], axis=1)
            index[has, side] = best
            near[has, side] = sub[has][np.arange(int(has.sum())), best]
    return near, index


def frame_features(
    ball_norm: FloatArray,
    players_norm: FloatArray,
    slot_team: npt.NDArray[np.int8],
    pitch: tuple[float, float],
    fps: int,
    graded: BoolArray,
    ball_gap_frames: int = 6,
) -> FrameFeatures:
    """Every per-frame feature of the two models, from positions only (see the name tuples).

    Team features are antisymmetric (swapping the teams negates them), so the team model needs
    no intercept and cannot learn a preference for one side; without a ball they are 0, except
    the past / next context, which carries the last and next sighting into short gaps.
    """
    ball = ball_track(ball_norm, pitch, ball_gap_frames)
    players = np.asarray(players_norm, dtype=np.float64) * np.asarray(pitch)
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    half_s, one_s, two_s = fps // 2, fps, 2 * fps

    slot_velocity = velocity(players.reshape(n, -1), fps).reshape(players.shape)
    slot_speed = np.linalg.norm(slot_velocity, axis=2)
    slot_speed[slot_speed > 12.0] = np.nan  # identity switches and duplicates, not people
    has_speed = ~np.isnan(slot_speed).all(axis=1)
    speeds = np.full((n, 3), np.nan)
    if has_speed.any():
        speeds[has_speed] = nan_quantiles(slot_speed[has_speed], (0.5, 0.75, 0.9))
    v75 = speeds[:, 1]
    past_v75, next_v75 = past_mean(v75, two_s), next_mean(v75, two_s)
    length, width = pitch
    with np.errstate(invalid="ignore"):
        off_pitch = (
            (players[:, :, 0] < -0.5)
            | (players[:, :, 0] > length + 0.5)
            | (players[:, :, 1] < -0.5)
            | (players[:, :, 1] > width + 0.5)
        ).sum(axis=1)
    separation = halves_separation(players, slot_team, length)
    dead = np.column_stack(
        [
            _relative_log(rolling_mean(speeds[:, 0], one_s), graded),
            _relative_log(rolling_mean(v75, two_s), graded),
            _relative_log(rolling_mean(speeds[:, 2], one_s), graded),
            _relative_log(past_v75, graded),
            _relative_log(next_v75, graded),
            np.nan_to_num(
                rolling_mean(np.where(seen, out_of_play(ball, pitch, 0.0), np.nan), half_s)
            ),
            rolling_mean(off_pitch.astype(np.float64), one_s),
            rolling_max(np.where(separation >= 0.9, separation, 0.0), one_s),
            _relative_log(past_mean(v75, one_s), graded),
            _relative_log(next_mean(v75, one_s), graded),
            _relative_log(past_mean(v75, 2 * two_s), graded),
            _relative_log(next_mean(v75, 2 * two_s), graded),
            np.nan_to_num(
                rolling_mean(np.where(seen, out_of_play(ball, pitch, 0.0), np.nan), two_s)
            ),
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = np.log(np.maximum(next_v75, 1e-3) / np.maximum(past_v75, 1e-3))

    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    near, index = _nearest_per_team(dist, slot_team)
    ball_velocity = velocity(ball, fps)
    ball_acceleration = np.linalg.norm(velocity(ball_velocity, fps), axis=1)
    rows = np.arange(n)
    control, touch = np.zeros((n, 2)), np.zeros((n, 2))
    for side in (H, A):
        ok = index[:, side] >= 0
        player_v = np.full((n, 2), np.nan)
        player_v[ok] = slot_velocity[rows[ok], index[ok, side]]
        relative = np.linalg.norm(ball_velocity - player_v, axis=1)
        closeness = np.exp(-near[:, side] / 1.0)
        control[:, side] = np.nan_to_num(closeness * np.exp(-relative / 3.0))
        touch[:, side] = np.nan_to_num(
            np.exp(-near[:, side] / 1.5) * np.minimum(1.0, ball_acceleration / 60.0)
        )
    team_cols = [np.exp(-near[:, 0] / s) - np.exp(-near[:, 1] / s) for s in (0.5, 1.0, 2.0, 4.0)]
    for radius in (2.0, 5.0):
        close = dist < radius  # NaN compares False
        team_cols.append(
            close[:, slot_team == H].sum(axis=1) - close[:, slot_team == A].sum(axis=1)
        )
    team_cols += [control[:, 0] - control[:, 1], touch[:, 0] - touch[:, 1]]
    team = np.nan_to_num(np.column_stack(team_cols).astype(np.float64))
    team[~seen] = 0.0
    near_1m = np.where(seen, team[:, 1], np.nan)
    ctrl = np.where(seen, team[:, 6], np.nan)
    quarter_s = max(1, fps // 4)
    context = np.column_stack(
        [
            past_mean(near_1m, half_s),
            next_mean(near_1m, half_s),
            past_mean(near_1m, quarter_s),
            next_mean(near_1m, quarter_s),
            past_mean(near_1m, one_s),
            next_mean(near_1m, one_s),
            past_mean(ctrl, half_s),
            next_mean(ctrl, half_s),
        ]
    )
    return FrameFeatures(
        dead=np.nan_to_num(dead, nan=0.0),
        team=np.column_stack([team, np.nan_to_num(context)]),
        seen=seen,
        speed_trend=np.nan_to_num(trend),
    )
