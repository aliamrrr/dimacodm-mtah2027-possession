"""Inputs of the deep model, computed the same way in training (other games) and on BAR-ATH.

numpy only: the submission never imports torch. Per frame:

* one token per tracked slot (46 at most, anonymous) and one ball token, see TOKEN_FEATURES:
  the network sees the set of players, not an ordering of them;
* GLOBAL_FEATURES: collective speeds before / after the frame, ball beyond a line, players off the
  pitch, kick-off formation, how many players each team has tracked, and the antisymmetric team
  features of the logistic solution (features.TEAM_FEATURES).

Everything is local in time (no statistic of the whole match), so a 30 s training window and a
full half at inference get the same numbers. `swap_teams` mirrors the two teams exactly: it is the
training augmentation and the test-time symmetrisation (no side can be preferred).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mtah.labels import A, H
from mtah.solutions.ali.features import (
    _nearest_per_team,
    ball_track,
    halves_separation,
    nan_quantiles,
    next_mean,
    out_of_play,
    past_mean,
    rolling_max,
    rolling_mean,
    velocity,
)

FloatArray = npt.NDArray[np.float32]

TOKEN_FEATURES = (
    "x",  # pitch coordinates mapped to [-1, 1]
    "y",
    "vx",  # m/s / 8 (ball: / 30)
    "vy",
    "speed",
    "team",  # +1 home, -1 away, 0 ball
    "dx_ball",  # player: (ball - player) / 20 m, clipped ; ball: raw observation this frame
    "dy_ball",  # player: idem ; ball: position bridged over a short gap
    "near",  # player: exp(-distance / 1 m) ; ball: log1p(frames since the last raw sighting) / 5
    "dist",  # player: distance / 10 m, clipped at 4 ; ball: log1p(frames to the next sighting) / 5
    "rel_speed",  # player: |v_ball - v_player| / 8 ; ball: acceleration / 100, clipped
    "ball_known",  # the ball position is available at this frame
    "is_ball",
)
GLOBAL_FEATURES = (
    "v50",  # players' speed quantiles, +/- 1 s, m/s / 4
    "v75",
    "v90",
    "v75_past2s",
    "v75_next2s",
    "v75_past4s",
    "v75_next4s",
    "out_of_play_0.5s",
    "out_of_play_2s",
    "off_pitch_1s",  # players beyond a line, / 3
    "kickoff_formation",
    "n_home",  # tracked players, / 11; swapped with n_away by swap_teams
    "n_away",
    "ball_known_1s",
    "near_0.5m",  # the 16 antisymmetric team features (negated by swap_teams)
    "near_1m",
    "near_2m",
    "near_4m",
    "count_2m",
    "count_5m",
    "control_1m",
    "touch",
    "near_1m_past",
    "near_1m_next",
    "near_1m_past_0.25s",
    "near_1m_next_0.25s",
    "near_1m_past_1s",
    "near_1m_next_1s",
    "control_past_0.5s",
    "control_next_0.5s",
)
N_TOKEN = len(TOKEN_FEATURES)
N_GLOBAL = len(GLOBAL_FEATURES)
TEAM_COLUMN = TOKEN_FEATURES.index("team")
COUNT_COLUMNS = (GLOBAL_FEATURES.index("n_home"), GLOBAL_FEATURES.index("n_away"))
ANTISYMMETRIC_GLOBALS = slice(GLOBAL_FEATURES.index("near_0.5m"), N_GLOBAL)


def _gap_clock(observed: npt.NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
    """Frames since the last / until the next observation (0 on an observed frame)."""
    n = len(observed)
    index = np.arange(n)
    last = np.where(observed, index, -1)
    last = np.maximum.accumulate(last)
    since = np.where(last >= 0, index - last, n).astype(np.float32)
    following = np.where(observed, index, n)
    following = np.minimum.accumulate(following[::-1])[::-1]
    until = np.where(following < n, following - index, n).astype(np.float32)
    return since, until


def inputs(
    slots_norm: npt.NDArray[np.floating],
    slot_team: npt.NDArray[np.int8],
    ball_norm: npt.NDArray[np.floating],
    pitch: tuple[float, float],
    fps: int,
    ball_gap_frames: int = 25,
) -> tuple[FloatArray, npt.NDArray[np.bool_], FloatArray]:
    """(tokens (n, S + 1, N_TOKEN), mask (n, S + 1), globals (n, N_GLOBAL)); token S is the ball."""
    scale = np.asarray(pitch, dtype=np.float64)
    raw_ball = np.asarray(ball_norm, dtype=np.float64)
    observed = ~np.isnan(raw_ball[:, 0])
    ball = ball_track(raw_ball, pitch, ball_gap_frames)  # metres
    players = np.asarray(slots_norm, dtype=np.float64) * scale
    n, n_slots = players.shape[:2]
    known = ~np.isnan(ball[:, 0])
    present = ~np.isnan(players[:, :, 0])

    slot_v = velocity(players.reshape(n, -1), fps).reshape(players.shape)
    slot_speed = np.linalg.norm(slot_v, axis=2)
    slot_v[slot_speed > 12.0] = np.nan  # identity switches and duplicates, not people
    slot_speed[slot_speed > 12.0] = np.nan
    ball_v = velocity(ball, fps)
    ball_acc = np.linalg.norm(velocity(ball_v, fps), axis=1)

    tokens = np.zeros((n, n_slots + 1, N_TOKEN), dtype=np.float32)
    rel = ball[:, None, :] - players
    dist = np.linalg.norm(rel, axis=2)
    rel_speed = np.linalg.norm(ball_v[:, None, :] - slot_v, axis=2)
    side = np.where(slot_team == H, 1.0, -1.0)
    p = tokens[:, :n_slots]
    p[..., 0] = players[..., 0] / scale[0] * 2 - 1
    p[..., 1] = players[..., 1] / scale[1] * 2 - 1
    p[..., 2] = slot_v[..., 0] / 8
    p[..., 3] = slot_v[..., 1] / 8
    p[..., 4] = slot_speed / 8
    p[..., 5] = side[None, :]
    p[..., 6] = np.clip(rel[..., 0] / 20, -3, 3)
    p[..., 7] = np.clip(rel[..., 1] / 20, -3, 3)
    p[..., 8] = np.exp(-dist / 1.0)
    p[..., 9] = np.minimum(dist / 10, 4)
    p[..., 10] = np.minimum(rel_speed / 8, 4)
    p[..., 11] = known[:, None]
    since, until = _gap_clock(observed)
    b = tokens[:, n_slots]
    b[:, 0] = ball[:, 0] / scale[0] * 2 - 1
    b[:, 1] = ball[:, 1] / scale[1] * 2 - 1
    b[:, 2] = ball_v[:, 0] / 30
    b[:, 3] = ball_v[:, 1] / 30
    b[:, 4] = np.linalg.norm(ball_v, axis=1) / 30
    b[:, 6] = observed
    b[:, 7] = known & ~observed
    b[:, 8] = np.log1p(since) / 5
    b[:, 9] = np.log1p(until) / 5
    b[:, 10] = np.minimum(ball_acc / 100, 4)
    b[:, 11] = known
    b[:, 12] = 1.0
    np.nan_to_num(tokens, copy=False)
    tokens[:, :n_slots][~present] = 0.0
    mask = np.concatenate([present, np.ones((n, 1), dtype=bool)], axis=1)
    return tokens, mask, _globals(players, slot_team, ball, known, slot_v, ball_v, dist, pitch, fps)


def _globals(
    players: npt.NDArray[np.float64],
    slot_team: npt.NDArray[np.int8],
    ball: npt.NDArray[np.float64],
    known: npt.NDArray[np.bool_],
    slot_v: npt.NDArray[np.float64],
    ball_v: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    pitch: tuple[float, float],
    fps: int,
) -> FloatArray:
    n = len(players)
    half_s, one_s, two_s = fps // 2, fps, 2 * fps
    quarter_s = max(1, fps // 4)
    speed = np.linalg.norm(slot_v, axis=2)
    has = ~np.isnan(speed).all(axis=1)
    q = np.full((n, 3), np.nan)
    if has.any():
        q[has] = nan_quantiles(speed[has], (0.5, 0.75, 0.9))
    v75 = q[:, 1]
    length, width = pitch
    with np.errstate(invalid="ignore"):
        off = (
            (players[:, :, 0] < -0.5)
            | (players[:, :, 0] > length + 0.5)
            | (players[:, :, 1] < -0.5)
            | (players[:, :, 1] > width + 0.5)
        ).sum(axis=1)
    out_line = np.where(known, out_of_play(ball, pitch, 0.0), np.nan)
    separation = halves_separation(players, slot_team, length)
    present = ~np.isnan(players[:, :, 0])

    near, index = _nearest_per_team(dist, slot_team)
    rows = np.arange(n)
    control, touch = np.zeros((n, 2)), np.zeros((n, 2))
    ball_acc = np.linalg.norm(velocity(ball_v, fps), axis=1)
    for s in (H, A):
        ok = index[:, s] >= 0
        pv = np.full((n, 2), np.nan)
        pv[ok] = slot_v[rows[ok], index[ok, s]]
        rel = np.linalg.norm(ball_v - pv, axis=1)
        control[:, s] = np.nan_to_num(np.exp(-near[:, s]) * np.exp(-rel / 3.0))
        touch[:, s] = np.nan_to_num(np.exp(-near[:, s] / 1.5) * np.minimum(1.0, ball_acc / 60.0))
    team_cols = [np.exp(-near[:, 0] / s) - np.exp(-near[:, 1] / s) for s in (0.5, 1.0, 2.0, 4.0)]
    for radius in (2.0, 5.0):
        close = dist < radius
        team_cols.append(
            close[:, slot_team == H].sum(axis=1) - close[:, slot_team == A].sum(axis=1)
        )
    team_cols += [control[:, 0] - control[:, 1], touch[:, 0] - touch[:, 1]]
    team = np.nan_to_num(np.column_stack(team_cols))
    team[~known] = 0.0
    near_1m = np.where(known, team[:, 1], np.nan)
    ctrl = np.where(known, team[:, 6], np.nan)
    context = [
        past_mean(near_1m, half_s),
        next_mean(near_1m, half_s),
        past_mean(near_1m, quarter_s),
        next_mean(near_1m, quarter_s),
        past_mean(near_1m, one_s),
        next_mean(near_1m, one_s),
        past_mean(ctrl, half_s),
        next_mean(ctrl, half_s),
    ]
    columns = [
        rolling_mean(q[:, 0], one_s) / 4,
        rolling_mean(v75, one_s) / 4,
        rolling_mean(q[:, 2], one_s) / 4,
        past_mean(v75, two_s) / 4,
        next_mean(v75, two_s) / 4,
        past_mean(v75, 2 * two_s) / 4,
        next_mean(v75, 2 * two_s) / 4,
        rolling_mean(out_line, half_s),
        rolling_mean(out_line, two_s),
        rolling_mean(off.astype(np.float64), one_s) / 3,
        rolling_max(np.where(separation >= 0.9, separation, 0.0), one_s),
        present[:, slot_team == H].sum(axis=1) / 11,
        present[:, slot_team == A].sum(axis=1) / 11,
        rolling_mean(known.astype(np.float64), one_s),
    ]
    out = np.column_stack([*columns, team, *context]).astype(np.float32)
    return np.nan_to_num(out, copy=False)


def swap_teams(tokens: FloatArray, globals_: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The same frames with home and away exchanged (copies)."""
    tokens = tokens.copy()
    globals_ = globals_.copy()
    tokens[..., TEAM_COLUMN] *= -1
    home, away = COUNT_COLUMNS
    globals_[..., [home, away]] = globals_[..., [away, home]]
    globals_[..., ANTISYMMETRIC_GLOBALS] *= -1
    return tokens, globals_
