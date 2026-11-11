"""Training samples for the deep model: CLEAN tracking of other games, degraded on the fly.

Nothing here ever reads BAR-ATH. The clean games are the benchmark's (bench.py): Metrica's three
public sample games and the seven IDSSE / DFL matches. Each sample is a window of a half that is
degraded with magnitudes drawn at random every time (domain randomisation): how long a slot
follows a player, gaps, position noise, duplicates, how often and how long the ball goes missing.
The ranges bracket the published figures of BAR-ATH's automatic tracking (DATA.md) instead of
pretending to know them, so the model has to work across the whole range.

Targets per frame: the label (H / A / D), the possessor slot (the possessing team's player nearest
to the clean ball, or "none" = the ball token, while the ball travels) and the clean ball
position, which the model must infer where the degraded ball is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from mtah.labels import A, D, H, X
from mtah.paths import cache_dir
from mtah.solutions.dimacodm import bench
from mtah.solutions.dimacodm.deep_features import inputs, swap_teams

N_SLOTS = bench.N_SLOTS
FPS = bench.FPS
PITCH = bench.PITCH
IGNORE = -100  # target value the losses skip
BOUNDARY_SIGMA = 3.0  # frames: soft target around a change of team (0.12 s)
CONVENTIONS = {"metrica": 0, "dfl": 1, "sc": 2}  # how each source defines possession


@dataclass(frozen=True)
class CleanGame:
    name: str
    convention: int
    players: npt.NDArray[np.float32]  # (n, p, 2) normalised, NaN off the pitch
    player_team: npt.NDArray[np.int8]  # (p,)
    ball: npt.NDArray[np.float32]  # (n, 2) normalised, NaN where unknown
    labels: npt.NDArray[np.int8]  # (n,) H / A / D, X outside the halves
    halves: tuple[tuple[int, int], ...]  # graded rows, half-open
    # False for SkillCorner: its broadcast camera shows replays during stoppages (few players,
    # rarely the ball), a cue BAR-ATH's wide tactical camera does not have. Its dead frames are
    # therefore never a target: it teaches which team has the ball, DFL / Metrica the stoppages.
    dead_targets: bool = True
    run_length: npt.NDArray[np.int64] | None = None  # (n,) length of the label run of each frame

    @property
    def n_frames(self) -> int:
        return len(self.labels)


def run_lengths(labels: npt.NDArray[np.int8]) -> npt.NDArray[np.int64]:
    """For each frame, the length of the maximal run of its label."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    sizes = np.diff(np.concatenate(([0], edges, [len(labels)])))
    return np.repeat(sizes, sizes)


def boundary_targets(labels: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
    """Soft target per frame: 1 where the team in possession changes in open play (H <-> A), a
    Gaussian of BOUNDARY_SIGMA frames around it. Restarts are not boundaries here."""
    teams = np.isin(labels, (H, A))
    changes = np.flatnonzero(teams[1:] & teams[:-1] & (labels[1:] != labels[:-1])) + 1
    if not len(changes):
        return np.zeros(len(labels), dtype=np.float32)
    rows = np.arange(len(labels))
    right = np.searchsorted(changes, rows).clip(0, len(changes) - 1)
    left = (right - 1).clip(0, None)
    nearest = np.minimum(np.abs(rows - changes[right]), np.abs(rows - changes[left]))
    return np.exp(-(nearest.astype(np.float64) ** 2) / (2 * BOUNDARY_SIGMA**2)).astype(np.float32)


def convention_of(name: str) -> int:
    source = name.split("_", 1)[0]
    return CONVENTIONS[source if source in CONVENTIONS else "metrica"]


def load_clean(name: str, root: Path | None = None) -> CleanGame:
    """One game of the benchmark, clean, cached as float32 under <cache>/deep/."""
    cache = (root or cache_dir() / "deep") / f"{name}.npz"
    if not cache.is_file():
        game = bench.load_game(name)
        spec = bench.bundle_spec(game)
        labels = game.labels.copy() if game.labels is not None else bench.derive_labels(game)
        labels[~spec.graded_mask()] = X
        ball = game.ball.copy()
        players = game.players.copy()
        if game.labels is None or name.startswith("sc_"):
            # Metrica hides the ball during stoppages, SkillCorner's broadcast shows replays then:
            # the benchmark's plausible path, never lost.
            rng = np.random.default_rng(bench.game_seed(name))
            settings = bench.Degradation(ball_lost_stoppage=0.0)
            for start, stop in bench._gap_rows(labels == D):
                bench._dead_ball_path(ball, start, stop, rng, settings)
        if name.startswith("sc_"):
            # ... and its players off camera: bridged linearly where both ends are known.
            dead = labels == D
            for p in range(players.shape[1]):
                for axis in (0, 1):
                    track = pd.Series(players[:, p, axis])
                    filled = track.interpolate(limit_area="inside").to_numpy()
                    players[dead, p, axis] = filled[dead]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            players=players.astype(np.float32),
            player_team=game.player_team,
            ball=ball.astype(np.float32),
            labels=labels,
            halves=np.array([(a, b + 1) for _, a, b in spec.halves]),
        )
    with np.load(cache) as data:
        labels = data["labels"].astype(np.int8)
        return CleanGame(
            name=name,
            convention=convention_of(name),
            players=data["players"],
            player_team=data["player_team"].astype(np.int8),
            ball=data["ball"],
            labels=labels,
            run_length=run_lengths(labels),
            halves=tuple((int(a), int(b)) for a, b in data["halves"]),
            dead_targets=not name.startswith("sc_"),
        )


@dataclass(frozen=True)
class Randomization:
    """Ranges the degradation of every sample is drawn from.

    The ranges cover what an automatic tracking of this kind really looks like, as measured
    WITHOUT any label (coarse bins of jitter and run lengths, docs/TECHNICAL_DOCUMENT.md): positions
    can be smooth (no frame-to-frame jitter at all), a slot can follow a player for minutes, the
    ball is seen or missing in runs of tens of seconds rather than one or two, and for a share
    of the time a large part of the players is not tracked. Durations are drawn log-uniformly so
    that the short and the long regimes are seen as often as each other.
    """

    tracklet_s: tuple[float, float] = (8.0, 200.0)
    gap_s: tuple[float, float] = (0.4, 10.0)
    position_noise_m: tuple[float, float] = (0.0, 0.40)
    tracklet_bias_m: tuple[float, float] = (0.0, 0.40)
    duplicates_per_min: tuple[float, float] = (0.0, 2.0)
    ball_missing_live: tuple[float, float] = (0.25, 0.55)
    ball_missing_dead: tuple[float, float] = (0.05, 0.95)  # from always seen to never seen
    # share of the stoppages whose ball keeps moving (dead_ball_motion): it glides to the restart
    # spot, wanders, or is carried by a player, instead of lying still as in the source data
    dead_ball_motion: tuple[float, float] = (0.0, 1.0)
    ball_gap_s: tuple[float, float] = (0.6, 30.0)
    ball_noise_m: tuple[float, float] = (0.0, 0.60)
    crowd_radius_m: float = 2.0
    fast_ball_ms: float = 15.0
    flip_p: float = 0.5
    swap_p: float = 0.5
    log_durations: bool = True  # tracklet_s, gap_s and ball_gap_s are drawn log-uniformly
    smooth_p: float = 0.5  # share of the windows whose tracking is smooth ...
    smooth_m: float = 0.02  # ... noise and bias under this (players and ball)
    blackout_share: tuple[float, float] = (0.0, 0.2)  # share of the time with few players tracked
    blackout_s: tuple[float, float] = (2.0, 14.0)  # how long such an episode lasts


# No degradation at all (ceiling experiment: how far can the model go with perfect tracking?).
CLEAN = Randomization(
    tracklet_s=(1e6, 1e6),
    gap_s=(0.04, 0.04),
    position_noise_m=(0.0, 0.0),
    tracklet_bias_m=(0.0, 0.0),
    duplicates_per_min=(0.0, 0.0),
    ball_missing_live=(0.0, 0.0),
    ball_missing_dead=(0.0, 0.0),
    ball_noise_m=(0.0, 0.0),
    smooth_p=0.0,
    blackout_share=(0.0, 0.0),
    dead_ball_motion=(0.0, 0.0),
)


def _draw(rng: np.random.Generator, bounds: tuple[float, float], log: bool = False) -> float:
    low, high = bounds
    if log and 0 < low < high:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def blackouts(
    slots: npt.NDArray[np.float64],
    ball_x: npt.NDArray[np.float64],
    rng: np.random.Generator,
    share: float,
    seconds: tuple[float, float],
) -> None:
    """Episodes, about `share` of the time, where a large part of the slots is not tracked (in
    place). Either a random 30-80 % of them, or every slot farther along the pitch than
    15-40 m from the ball: a camera that follows the play loses the players it leaves behind."""
    n = len(slots)
    if share <= 0 or n == 0:
        return
    mean_length = 0.5 * (seconds[0] + seconds[1]) * FPS
    t = 0
    while True:
        t += int(rng.exponential(mean_length * (1 - share) / share))
        if t >= n:
            return
        stop = min(n, t + max(1, int(rng.uniform(*seconds) * FPS)))
        centre = ball_x[t] if not np.isnan(ball_x[t]) else 0.5
        if rng.random() < 0.5:
            gone = rng.random(slots.shape[1]) < rng.uniform(0.3, 0.8)
            slots[t:stop, gone] = np.nan
        else:
            reach = rng.uniform(15.0, 40.0) / PITCH[0]
            with np.errstate(invalid="ignore"):
                far = np.abs(slots[t:stop, :, 0] - centre) > reach
            slots[t:stop][far] = np.nan
        t = stop


def _approach(path: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> None:
    """End `path` (in place) on `target`: a straight run over its last frames, long enough to
    stay under about 12 m/s (at least 1 s, at most the whole path)."""
    steps = len(path)
    tail = min(steps - 1, FPS)
    scale = np.asarray(PITCH)
    for _ in range(3):  # the start of the run moves as the run gets longer
        last = path[steps - tail - 1]
        need = int(np.linalg.norm((target - last) * scale) / 12.0 * FPS)
        longer = min(steps - 1, max(tail, need))
        if longer == tail:
            break
        tail = longer
    if tail > 0:
        last = path[steps - tail - 1].copy()
        path[steps - tail :] = last + np.linspace(0.0, 1.0, tail)[:, None] * (target - last)


def dead_ball_motion(
    ball: npt.NDArray[np.float64],
    players: npt.NDArray[np.float64],
    dead: npt.NDArray[np.bool_],
    rng: np.random.Generator,
    share: float,
) -> npt.NDArray[np.bool_]:
    """For about `share` of the stoppages, rewrite the ball's path (in place, normalised
    coordinates) so that it keeps moving on the pitch. Returns the frames rewritten.

    The source data shows a dead ball lying still (DFL) or fetched then still (our own path for
    the sources that hide it). A label-free look at an automatic tracking shows the opposite: a
    ball that is seen during a stoppage is almost never still. Four ways it keeps moving, drawn
    at random: it glides in a straight line to where play restarts (what a tracker that bridges
    its own gaps produces), it wanders between a few points (thrown back, a second ball), a
    player carries it, or it is passed around from player to player whatever their team (given
    back to the taker, a kick-about before a kick-off). The model then has to tell a stoppage
    from the players, not from a ball that happens to lie still."""
    n = len(ball)
    changed = np.zeros(n, dtype=bool)
    if share <= 0:
        return changed
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    scale = np.asarray(PITCH)
    spans = zip(
        np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True
    )
    for a, b in spans:
        if b - a < FPS or rng.random() >= share:
            continue
        before = ball[max(0, a - 2 * FPS) : a]
        before = before[~np.isnan(before[:, 0])]
        after = ball[b : b + 2 * FPS]
        after = after[~np.isnan(after[:, 0])]
        p0 = before[-1] if len(before) else rng.uniform(0.05, 0.95, 2)
        p1 = after[0] if len(after) else rng.uniform(0.05, 0.95, 2)
        steps = b - a
        mode = int(rng.integers(4))
        points = [p0, p1]
        if mode == 1:  # wander through one to three points, a little beyond the lines at times
            points = [p0, *rng.uniform(-0.03, 1.03, (int(rng.integers(1, 4)), 2)), p1]
        elif mode == 2:  # carried: towards the player who is there the most, then with him
            seen = ~np.isnan(players[a:b, :, 0])
            if seen.any():
                who = int(np.argmax(seen.sum(axis=0)))
                rows = np.flatnonzero(seen[:, who])
                track = np.column_stack(
                    [np.interp(np.arange(steps), rows, players[a:b, who, k][rows]) for k in (0, 1)]
                )
                track += rng.normal(0.0, 0.4, 2) / scale
                lead = max(1, min(steps - 1, int(rng.uniform(1.0, 3.0) * FPS)))
                metres = float(np.linalg.norm((track[lead] - p0) * scale))
                lead = max(lead, min(steps - 1, int(metres / 12.0 * FPS)))
                weight = np.ones(steps)
                weight[:lead] = np.linspace(0.0, 1.0, lead)
                path = p0 + weight[:, None] * (track - p0)
                _approach(path, p1)
                ball[a:b] = path
                changed[a:b] = True
                continue
        elif mode == 3:  # passed around: from player to player at 5-12 m/s, held a moment
            path = np.empty((steps, 2))
            t, here = 0, np.asarray(p0, dtype=np.float64)
            while t < steps:
                cols = np.flatnonzero(~np.isnan(players[a + t, :, 0]))
                if not len(cols):
                    path[t:] = here
                    break
                who = int(rng.choice(cols))
                metres = float(np.linalg.norm((players[a + t, who] - here) * scale))
                end = min(steps, t + max(5, int(metres / rng.uniform(5.0, 12.0) * FPS)))
                arrive = players[a + end - 1, who]
                if np.isnan(arrive[0]):
                    arrive = players[a + t, who]
                path[t:end] = here + np.linspace(0.0, 1.0, end - t)[:, None] * (arrive - here)
                t, here = end, arrive.astype(np.float64)
                hold = min(steps - t, int(rng.uniform(0.3, 2.5) * FPS))
                if hold > 0:
                    with_him = players[a + t : a + t + hold, who]
                    kept = ~np.isnan(with_him[:, 0])
                    path[t : t + hold] = np.where(kept[:, None], with_him, here)
                    here = path[t + hold - 1].copy()
                    t += hold
            _approach(path, p1)
            ball[a:b] = path
            changed[a:b] = True
            continue
        corners = np.asarray(points, dtype=np.float64)
        legs = np.linalg.norm(np.diff(corners, axis=0) * scale, axis=1)
        travelled = np.concatenate(([0.0], np.cumsum(legs)))
        if travelled[-1] > 15.0 * steps / FPS:  # too far for this stoppage: straight, or as is
            corners = np.asarray([p0, p1], dtype=np.float64)
            travelled = np.array([0.0, float(np.linalg.norm((p1 - p0) * scale))])
        if travelled[-1] <= 0 or travelled[-1] > 20.0 * steps / FPS:
            continue
        at = np.linspace(0.0, travelled[-1], steps)  # constant speed along the whole path
        ball[a:b] = np.column_stack([np.interp(at, travelled, corners[:, k]) for k in (0, 1)])
        changed[a:b] = True
    return changed


def _on_runs(
    rng: np.random.Generator, n: int, mean_on: float, mean_off: float
) -> list[tuple[int, int]]:
    runs, t, on = [], 0, bool(rng.random() < mean_on / (mean_on + mean_off))
    while t < n:
        length = max(1, int(rng.exponential(mean_on if on else mean_off)))
        if on:
            runs.append((t, min(n, t + length)))
        t += length
        on = not on
    return runs


def _window(array: npt.NDArray, start: int, stop: int, fill: float) -> npt.NDArray:
    """array[start:stop] padded with `fill` where the window leaves the array."""
    n = len(array)
    out = np.full((stop - start, *array.shape[1:]), fill, dtype=array.dtype)
    lo, hi = max(0, start), min(n, stop)
    if hi > lo:
        out[lo - start : hi - start] = array[lo:hi]
    return out


def degrade_window(
    game: CleanGame, start: int, stop: int, rng: np.random.Generator, rnd: Randomization
) -> dict[str, npt.NDArray]:
    """One degraded window [start, stop) with its targets (rows outside the game are empty)."""
    scale = np.asarray(PITCH)
    clean_players = _window(game.players, start, stop, np.nan).astype(np.float64)
    clean_ball = _window(game.ball, start, stop, np.nan).astype(np.float64)
    labels = _window(game.labels, start, stop, X)
    w, n_players = clean_players.shape[:2]
    if rng.random() < rnd.flip_p:
        clean_players[..., 0], clean_ball[:, 0] = 1 - clean_players[..., 0], 1 - clean_ball[:, 0]
    if rng.random() < rnd.flip_p:
        clean_players[..., 1], clean_ball[:, 1] = 1 - clean_players[..., 1], 1 - clean_ball[:, 1]

    slot_team = np.array([H] * (N_SLOTS // 2) + [A] * (N_SLOTS // 2), dtype=np.int8)
    rng.shuffle(slot_team)
    slots = np.full((w, N_SLOTS, 2), np.nan)
    slot_of = np.full((w, n_players), -1, dtype=np.int64)  # which slot shows a player, per frame
    free_at = np.zeros(N_SLOTS, dtype=np.int64)
    log = rnd.log_durations
    mean_on, mean_off = _draw(rng, rnd.tracklet_s, log) * FPS, _draw(rng, rnd.gap_s, log) * FPS
    noise_m, bias_m = _draw(rng, rnd.position_noise_m), _draw(rng, rnd.tracklet_bias_m)
    ball_noise_m = _draw(rng, rnd.ball_noise_m)
    if rng.random() < rnd.smooth_p:  # smooth tracking: (almost) no jitter, players and ball
        noise_m, bias_m = rng.uniform(0.0, rnd.smooth_m, 2).tolist()
        ball_noise_m = float(rng.uniform(0.0, rnd.smooth_m))
    tracklets = []
    for player in range(n_players):
        on_pitch = ~np.isnan(clean_players[:, player, 0])
        if not on_pitch.any():
            continue
        for a, b in _on_runs(rng, w, mean_on, mean_off):
            rows = np.flatnonzero(on_pitch[a:b]) + a
            if len(rows):
                tracklets.append((int(rows[0]), int(rows[-1]) + 1, player))
    tracklets.sort()
    for a, b, player in tracklets:
        candidates = np.flatnonzero((slot_team == game.player_team[player]) & (free_at <= a))
        if not len(candidates):
            continue
        slot = int(rng.choice(candidates))
        free_at[slot] = b
        offset = rng.normal(0.0, bias_m, 2) / scale
        slots[a:b, slot] = clean_players[a:b, player] + offset
        slot_of[a:b, player] = slot
    for _ in range(int(rng.poisson(_draw(rng, rnd.duplicates_per_min) * w / FPS / 60))):
        player, a = int(rng.integers(n_players)), int(rng.integers(w))
        b = min(w, a + int(rng.uniform(1, 3) * FPS))
        candidates = np.flatnonzero((slot_team == game.player_team[player]) & (free_at <= a))
        if len(candidates) and not np.isnan(clean_players[a, player, 0]):
            slot = int(rng.choice(candidates))
            free_at[slot] = b
            slots[a:b, slot] = clean_players[a:b, player] + rng.normal(0.0, 0.7, 2) / scale
    slots += rng.normal(0.0, noise_m, slots.shape) / scale
    blackouts(slots, clean_ball[:, 0], rng, _draw(rng, rnd.blackout_share), rnd.blackout_s)
    rows, cols = np.nonzero(slot_of >= 0)
    lost = np.isnan(slots[rows, slot_of[rows, cols], 0])
    slot_of[rows[lost], cols[lost]] = -1  # a player nobody tracks cannot be named possessor

    # Stoppages: the ball of a share of them keeps moving. Then a two-state dropout chain, more
    # likely to drop when crowded or fast.
    seen_ball = clean_ball.copy()
    moved = dead_ball_motion(
        seen_ball, clean_players, labels == D, rng, _draw(rng, rnd.dead_ball_motion)
    )
    ball_m = seen_ball * scale
    near = np.linalg.norm(clean_players * scale - ball_m[:, None, :], axis=2)
    crowded = np.nansum(near < rnd.crowd_radius_m, axis=1) >= 2
    speed = np.zeros(w)
    speed[1:] = np.nan_to_num(np.linalg.norm(np.diff(ball_m, axis=0), axis=1) * FPS)
    modulation = 1.0 + 2.0 * crowded + 2.0 * (speed > rnd.fast_ball_ms)
    recover = 1.0 / (_draw(rng, rnd.ball_gap_s, log) * FPS)
    dead = labels == D
    hazard = np.zeros(w)
    for state, bounds in ((False, rnd.ball_missing_live), (True, rnd.ball_missing_dead)):
        rows = dead == state
        if rows.any():
            target = _draw(rng, bounds)
            hazard[rows] = (
                target * recover / (1 - target) * modulation[rows] / modulation[rows].mean()
            )
    draws = rng.random(size=(w,))
    missing = np.zeros(w, dtype=bool)
    off = bool(rng.random() < 0.4)
    for i in range(w):
        off = draws[i] >= recover if off else draws[i] < hazard[i]
        missing[i] = off
    ball = seen_ball + rng.normal(0.0, ball_noise_m, clean_ball.shape) / scale
    ball[missing] = np.nan

    # Possessor: the possessing team's nearest player to the clean ball (2 m), else "none".
    possessor = np.full(w, IGNORE, dtype=np.int64)
    alive = np.isin(labels, (H, A)) & ~np.isnan(clean_ball[:, 0])
    for team in (H, A):
        rows = np.flatnonzero(alive & (labels == team))
        if not len(rows):
            continue
        cols = np.flatnonzero(game.player_team == team)
        d = np.where(np.isnan(near[np.ix_(rows, cols)]), np.inf, near[np.ix_(rows, cols)])
        best = np.argmin(d, axis=1)
        close = d[np.arange(len(rows)), best] <= 2.0
        slot = slot_of[rows, cols[best]]
        possessor[rows] = np.where(close & (slot >= 0), slot, N_SLOTS)
    return {
        "slots": slots.astype(np.float32),
        "slot_team": slot_team,
        "ball": ball.astype(np.float32),
        "clean_ball": clean_ball.astype(np.float32),
        "ball_moved": moved,  # frames whose ball path is ours: no position target there
        "labels": labels,
        "possessor": possessor,
    }


def make_sample(
    game: CleanGame,
    start: int,
    stop: int,
    rng: np.random.Generator,
    rnd: Randomization,
    ball_gap_frames: int = 25,
    spell_alpha: float = 0.0,
) -> dict[str, npt.NDArray]:
    """Model inputs and targets for one window (team swap applied at random).

    spell_alpha weighs each frame by (length of its run in s) ** -alpha, floored at 0.5 s. spellF1
    counts every possession once while a per-frame loss counts it by its length: 1 makes every
    possession weigh the same in the loss, 0 is the plain per-frame loss."""
    raw = degrade_window(game, start, stop, rng, rnd)
    lengths = game.run_length if game.run_length is not None else run_lengths(game.labels)
    run = _window(lengths, start, stop, 1).astype(np.float64)
    tokens, mask, globals_ = inputs(
        raw["slots"], raw["slot_team"], raw["ball"], PITCH, FPS, ball_gap_frames
    )
    labels = raw["labels"].astype(np.int64)
    if rng.random() < rnd.swap_p:
        tokens, globals_ = swap_teams(tokens, globals_)
        labels = np.where(labels == H, A, np.where(labels == A, H, labels))
    state = np.where(labels == X, IGNORE, labels)
    if not game.dead_targets:
        state = np.where(labels == D, IGNORE, state)
    ball_target = raw["clean_ball"] * 2 - 1
    state_weight = (np.maximum(run / FPS, 0.5) ** -spell_alpha).astype(np.float32)
    return {
        "tokens": tokens,
        "mask": mask,
        "globals": globals_,
        "state": state,
        "state_weight": state_weight,
        "boundary": boundary_targets(labels),
        "boundary_mask": (state != IGNORE).astype(np.float32),
        "possessor": raw["possessor"],
        "ball_target": np.nan_to_num(ball_target),
        "ball_weight": (~np.isnan(ball_target[:, 0]) & ~raw["ball_moved"]).astype(np.float32)
        * np.where(np.isnan(raw["ball"][:, 0]), 2.0, 1.0).astype(np.float32),
        "convention": np.int64(game.convention),
    }


def sample_window(
    games: list[CleanGame], rng: np.random.Generator, length: int, margin: int
) -> tuple[CleanGame, int, int]:
    """A window whose central part [start + margin, stop - margin) lies in a graded half.

    Both conventions are drawn equally often, whatever their number of games."""
    by_convention: dict[int, list[CleanGame]] = {}
    for game in games:
        by_convention.setdefault(game.convention, []).append(game)
    pool = by_convention[sorted(by_convention)[int(rng.integers(len(by_convention)))]]
    game = pool[int(rng.integers(len(pool)))]
    sizes = np.array([b - a for a, b in game.halves], dtype=np.float64)
    a, b = game.halves[int(rng.choice(len(sizes), p=sizes / sizes.sum()))]
    centre = length - 2 * margin
    first = int(rng.integers(a, max(a + 1, b - centre + 1)))
    return game, first - margin, first - margin + length
