"""Automatic Tracking Data (ATD): FIFA/EPTS raw text + metadata XML -> numpy arrays.

Row i of every array is VIDEO FRAME i, i.e. submission row i (ATD frame_id = video frame + 1).
Missing observations are NaN, for the ball and for players alike.

Conventions are those of the official starter/baseline.py, on purpose: slot order is the order
of <Player> elements in the metadata, the first <Team> is home. `mtah data check-loader` proves on
the real data that this loader + the dummy solution reproduce the official baseline byte for byte.

Parsing strategy
----------------
A raw line is ``<frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<ball fields>``. When every line has the same
number of fields (the normal case) we replace the separators by commas and let numpy parse whole
chunks at C speed (~10x faster than a Python loop). Each chunk is cross-checked line by line against
the slow reference parser on a sample; any disagreement falls back to the slow parser for the whole
file, so the fast path can never silently change a value.

Parsed arrays are cached under .cache/ as .npy files and memory-mapped READ-ONLY on the next load:
solutions must never modify their inputs (``.copy()`` first if you need to).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mtah.paths import cache_dir

PARSER_VERSION = 1
HOME, AWAY = 0, 1
_CHUNK_BYTES = 32 << 20
_SAMPLE_PER_CHUNK = 64
_SEPARATORS = str.maketrans({":": ",", ";": ",", "\n": ",", "\r": None})

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Tracking:
    """Tracking for one recording. All positions normalised 0-1 (multiply by the pitch for metres)."""

    ball: FloatArray  # (n_frames, 2) x, y
    ball_extra: npt.NDArray[np.float32]  # (n_frames, k) remaining ball fields, k may be 0
    players: FloatArray  # (n_frames, n_slots, 2) x, y
    player_extra: npt.NDArray[np.float32]  # (n_frames, n_slots, k), [..., 0] is speed in m/s
    slot_team: npt.NDArray[np.int8]  # (n_slots,) 0 = home, 1 = away
    team_names: tuple[str, str]
    source: str = "tactical"

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.players.shape[1])

    @property
    def ball_visible(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self.ball[:, 0])

    @property
    def player_visible(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self.players[:, :, 0])

    @property
    def player_speed(self) -> npt.NDArray[np.float32]:
        if self.player_extra.shape[2] == 0:
            raise AttributeError("this ATD export has no speed field")
        return self.player_extra[:, :, 0]

    def ball_m(self, pitch: tuple[float, float]) -> FloatArray:
        return self.ball * np.asarray(pitch, dtype=np.float64)

    def players_m(self, pitch: tuple[float, float]) -> FloatArray:
        return self.players * np.asarray(pitch, dtype=np.float64)


# -- metadata ---------------------------------------------------------------------------------


def read_metadata(meta_path: Path) -> tuple[npt.NDArray[np.int8], tuple[str, str]]:
    """(slot_team, (home_name, away_name)), with the official baseline's conventions."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 <Team> elements, found {len(teams)}")
    home_id = teams[0].get("id")
    names = (teams[0].findtext("Name") or "home", teams[1].findtext("Name") or "away")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no <Player> entries")
    slot_team = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return slot_team, names


# -- raw parsing ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    n_slots: int
    k_player: int
    k_ball: int

    @property
    def width(self) -> int:
        return 1 + self.n_slots * self.k_player + self.k_ball


class _FastPathError(Exception):
    pass


def _detect_layout(raw_path: Path) -> _Layout:
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                parts = line.strip().split(":")
                if len(parts) != 3:
                    raise ValueError(f"{raw_path}: unexpected line format {line[:80]!r}")
                groups = parts[1].split(";")
                k_player = max(len(group.split(",")) for group in groups)
                return _Layout(len(groups), k_player, len(parts[2].split(",")))
    raise ValueError(f"{raw_path}: empty file")


def _parse_line(line: str, layout: _Layout) -> tuple[int, FloatArray] | None:
    """Reference parser for one line (generalised from the official baseline). None = skipped."""
    parts = line.strip().split(":")
    if len(parts) != 3:
        return None
    row = np.full(layout.width, np.nan)
    row[0] = int(parts[0])
    groups = parts[1].split(";")
    for slot, group in enumerate(groups[: layout.n_slots]):
        fields = group.split(",")[: layout.k_player]
        start = 1 + slot * layout.k_player
        row[start : start + len(fields)] = [float(v) for v in fields]
    ball = parts[2].split(",")[: layout.k_ball]
    start = 1 + layout.n_slots * layout.k_player
    row[start : start + len(ball)] = [float(v) for v in ball]
    return int(parts[0]), row


def _parse_slow(raw_path: Path, layout: _Layout, n_frames: int) -> FloatArray:
    table = np.full((n_frames, layout.width), np.nan)
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_line(line, layout)
            if parsed is None:
                continue
            frame_id, row = parsed
            if 1 <= frame_id <= n_frames:
                table[frame_id - 1] = row
    return table


def _parse_fast(raw_path: Path, layout: _Layout, n_frames: int, seed: int = 0) -> FloatArray:
    table = np.full((n_frames, layout.width), np.nan)
    rng = np.random.default_rng(seed)
    with open(raw_path, encoding="utf-8") as handle:
        while True:
            lines = handle.readlines(_CHUNK_BYTES)
            if not lines:
                break
            lines = [line for line in lines if line.strip()]
            if not lines:
                continue
            flat = np.fromstring("".join(lines).translate(_SEPARATORS), sep=",")
            if flat.size != len(lines) * layout.width:
                raise _FastPathError("inconsistent field count")
            block = flat.reshape(len(lines), layout.width)
            frame_ids = block[:, 0]
            if not np.all(frame_ids == np.round(frame_ids)):
                raise _FastPathError("non-integer frame id")
            # Cross-check a sample of lines against the reference parser.
            sample = np.unique(
                np.concatenate(
                    [[0, len(lines) - 1], rng.integers(0, len(lines), _SAMPLE_PER_CHUNK)]
                )
            )
            for index in sample:
                parsed = _parse_line(lines[index], layout)
                if parsed is None or not np.array_equal(parsed[1], block[index], equal_nan=True):
                    raise _FastPathError(f"sample mismatch on line {lines[index][:40]!r}")
            rows = frame_ids.astype(np.int64) - 1
            keep = (rows >= 0) & (rows < n_frames)
            table[rows[keep]] = block[keep]
    return table


def parse_raw(raw_path: Path, n_slots: int, n_frames: int) -> tuple[FloatArray, _Layout, str]:
    """(table, layout, parser_used). table[i] is video frame i: [frame_id, players..., ball...]."""
    layout = _detect_layout(raw_path)
    if layout.n_slots != n_slots:
        raise ValueError(
            f"{raw_path}: {layout.n_slots} player groups per line but the metadata lists "
            f"{n_slots} track slots"
        )
    try:
        return _parse_fast(raw_path, layout, n_frames), layout, "fast"
    except _FastPathError:
        return _parse_slow(raw_path, layout, n_frames), layout, "slow"


def count_records(raw_path: Path) -> tuple[int, int | None, int | None, bool]:
    """(records, first frame_id, last frame_id, contiguous) in one streaming pass."""
    count, first, last, contiguous, previous = 0, None, None, True, None
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame_id = int(line[: line.index(":")])
            first = frame_id if first is None else first
            if previous is not None and frame_id != previous + 1:
                contiguous = False
            previous = last = frame_id
            count += 1
    return count, first, last, contiguous


# -- cache ------------------------------------------------------------------------------------


def _cache_key(meta_path: Path, raw_path: Path, n_frames: int) -> str:
    parts = [str(PARSER_VERSION), str(n_frames)]
    for path in (meta_path, raw_path):
        stat = path.stat()
        parts += [str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


_ARRAYS = ("ball", "ball_extra", "players", "player_extra", "slot_team")


def load_tracking(
    meta_path: Path,
    raw_path: Path,
    n_frames: int,
    source: str = "tactical",
    use_cache: bool = True,
) -> Tracking:
    meta_path, raw_path = Path(meta_path), Path(raw_path)
    target = cache_dir() / f"atd_{source}_{_cache_key(meta_path, raw_path, n_frames)}"
    if use_cache and (target / "meta.json").is_file():
        with open(target / "meta.json", encoding="utf-8") as handle:
            info = json.load(handle)
        arrays = {name: np.load(target / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
        return Tracking(
            team_names=(info["team_names"][0], info["team_names"][1]), source=source, **arrays
        )

    slot_team, team_names = read_metadata(meta_path)
    table, layout, parser = parse_raw(raw_path, len(slot_team), n_frames)
    start = 1
    end_players = start + layout.n_slots * layout.k_player
    players_all = table[:, start:end_players].reshape(n_frames, layout.n_slots, layout.k_player)
    ball_all = table[:, end_players:]
    tracking = Tracking(
        ball=np.ascontiguousarray(ball_all[:, :2]),
        ball_extra=np.ascontiguousarray(ball_all[:, 2:], dtype=np.float32),
        players=np.ascontiguousarray(players_all[:, :, :2]),
        player_extra=np.ascontiguousarray(players_all[:, :, 2:], dtype=np.float32),
        slot_team=slot_team,
        team_names=team_names,
        source=source,
    )
    if use_cache:
        tmp = target.with_name(target.name + f".tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name in _ARRAYS:
            np.save(tmp / f"{name}.npy", getattr(tracking, name))
        with open(tmp / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "team_names": list(team_names),
                    "parser": parser,
                    "layout": [layout.n_slots, layout.k_player, layout.k_ball],
                    "raw": str(raw_path),
                },
                handle,
            )
        if _publish(tmp, target):
            return load_tracking(meta_path, raw_path, n_frames, source, use_cache=True)
        for name in _ARRAYS:  # same contract as the memory-mapped cache: read-only inputs
            getattr(tracking, name).flags.writeable = False
    return tracking


def _publish(tmp: Path, target: Path) -> bool:
    """Move a freshly written cache folder into place; False if it could not be done.

    On Windows an antivirus or the search indexer may hold a new file for a moment, and renaming
    the folder then fails with PermissionError. Retry briefly, then give up: the caller still has
    the arrays in memory, so the run goes on without the cache instead of crashing.
    """
    for delay in (0.0, 0.1, 0.3, 1.0):
        time.sleep(delay)
        shutil.rmtree(target, ignore_errors=True)
        try:
            tmp.rename(target)
            return True
        except OSError:
            continue
    shutil.rmtree(tmp, ignore_errors=True)
    return False
