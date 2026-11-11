"""A fake match in the EXACT file formats of the real bundle, with its own ground truth.

Why it exists: the real data is gated and must never reach GitHub, yet the tests have to run
every solution end to end. The synthetic bundle is generated from a seed (no real data
involved at all), so it is safe to build anywhere and lets us test loaders, solutions, determinism,
the official validator and the scorer on a realistic 199,319-frame match.

It is NOT a substitute for the real match: scores on it are a smoke test, never a ranking.

Layout written (same names as the real bundle, plus synthetic/), with LF line endings on every OS:

    <out>/tactical/BAR-ATH_tactical_atd.txt            FIFA/EPTS raw, 46 slots, x,y,speed + ball x,y,z
    <out>/tactical/BAR-ATH_tactical_atd.xml            metadata (teams, track slots)
    <out>/tactical/BAR-ATH_tactical_homography.csv
    <out>/tactical/BAR-ATH_tactical_smart_tagging.xml  SportsCode XML
    <out>/synthetic/gt_full.csv                         ground truth, every graded frame
    <out>/synthetic/gt_public.csv                       first 5 minutes only (mimics the public GT)
    <out>/synthetic/meta.json                           spec, seed, generator version, statistics

What is simulated: possessions (long spells and short touches), turnovers, stoppages with
restarts, passes between teammates, a camera that follows play, ball dropouts (~41 % of graded
frames), player dropouts, track-slot identity switches, interleaved team slots, and Smart Tagging
with jittered boundaries, N/A throw-ins and false positives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import numpy.typing as npt

from mtah.labels import A, D, H, X
from mtah.spec import MatchSpec

GENERATOR_VERSION = 1
N_SLOTS = 46
N_PLAYERS = 11
TEAM_TAGS = ("Barcelona", "Atletic Club")
TEAM_NAMES = ("FC Barcelona (synthetic)", "Athletic Club (synthetic)")
FORMATION = np.array(  # attacking towards +x, normalised pitch
    [
        [0.05, 0.50],
        [0.20, 0.15],
        [0.20, 0.38],
        [0.20, 0.62],
        [0.20, 0.85],
        [0.40, 0.15],
        [0.40, 0.38],
        [0.40, 0.62],
        [0.40, 0.85],
        [0.60, 0.38],
        [0.60, 0.62],
    ]
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    label: int


def mini_spec() -> MatchSpec:
    """A 24-minute bundle with two 10-minute halves, for fast unit tests."""
    return MatchSpec(
        game="BAR-ATH",
        n_frames=36_000,
        fps=25,
        pitch=(105.3, 68.0),
        halves=((1, 1_500, 16_499), (2, 19_000, 33_999)),
        synthetic=True,
    )


def full_spec() -> MatchSpec:
    official = MatchSpec.official()
    return MatchSpec(
        official.game,
        official.n_frames,
        official.fps,
        official.pitch,
        official.halves,
        synthetic=True,
    )


# -- helpers ----------------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, n: int, k: int, corr_frames: float) -> FloatArray:
    """(n, k) unit-variance Gaussian noise, low-passed with a Gaussian of `corr_frames` width."""
    white = rng.standard_normal((n, k))
    freqs = np.fft.rfftfreq(n)
    gain = np.exp(-0.5 * (freqs * 2 * np.pi * corr_frames) ** 2)
    smooth = np.fft.irfft(np.fft.rfft(white, axis=0) * gain[:, None], n=n, axis=0)
    return smooth / (smooth.std(axis=0, keepdims=True) + 1e-12)


def _lowpass(signal: FloatArray, corr_frames: float) -> FloatArray:
    n = len(signal)
    freqs = np.fft.rfftfreq(n)
    gain = np.exp(-0.5 * (freqs * 2 * np.pi * corr_frames) ** 2)
    return np.fft.irfft(
        np.fft.rfft(signal, axis=0) * gain.reshape(-1, *([1] * (signal.ndim - 1))), n=n, axis=0
    )


def _renewal_mask(
    rng: np.random.Generator, n: int, mean_on: float, mean_off: float
) -> npt.NDArray[np.bool_]:
    """Alternating on/off durations (exponential, in frames). True = observed."""
    out = np.empty(n, dtype=bool)
    t, on = 0, bool(rng.random() < mean_on / (mean_on + mean_off))
    while t < n:
        length = max(1, int(rng.exponential(mean_on if on else mean_off)))
        out[t : t + length] = on
        t += length
        on = not on
    return out


def _attack_dir(team: int, half: int) -> float:
    """+1 if `team` attacks towards x=1 in this half (home attacks +x in the first half)."""
    return 1.0 if (team == H) == (half == 1) else -1.0


# -- simulation -------------------------------------------------------------------------------


def _timeline(
    rng: np.random.Generator, spec: MatchSpec, fps: int
) -> tuple[npt.NDArray[np.int8], list[Segment], dict[int, int]]:
    gt = np.full(spec.n_frames, X, dtype=np.int8)
    segments: list[Segment] = []
    kickoffs: dict[int, int] = {}
    for half, first, last in spec.halves:
        stop_half = last + 1
        kickoff = first if half == 1 else first + min(int(117.4 * fps), (stop_half - first) // 10)
        kickoffs[half] = kickoff
        if kickoff > first:
            segments.append(Segment(first, kickoff, D))
        team = H if half == 1 else A
        t = kickoff
        while t < stop_half:
            if rng.random() < 0.22:
                seconds = rng.lognormal(np.log(1.6), 0.5)
            else:
                seconds = rng.lognormal(np.log(11.0), 0.75)
            length = int(np.clip(seconds, 0.4, 80.0) * fps)
            segments.append(Segment(t, min(stop_half, t + length), team))
            t += length
            if t >= stop_half:
                break
            if rng.random() < 0.28:
                dead = int(np.clip(rng.lognormal(np.log(13.0), 0.5), 3.0, 50.0) * fps)
                segments.append(Segment(t, min(stop_half, t + dead), D))
                t += dead
                team = 1 - team if rng.random() < 0.6 else team
            else:
                team = 1 - team
    for seg in segments:
        gt[seg.start : seg.stop] = seg.label
    return gt, segments, kickoffs


def _players(
    rng: np.random.Generator, spec: MatchSpec, gt: npt.NDArray[np.int8]
) -> tuple[FloatArray, FloatArray]:
    """(positions (n, 2 teams, 11, 2), play centre (n, 2)) in normalised coordinates."""
    n = spec.n_frames
    half = spec.half_of_frame()
    # Where play is drifting: towards the goal of the team in possession.
    target_x = np.full(n, 0.5)
    for team in (H, A):
        for h in (1, 2):
            mask = (gt == team) & (half == h)
            target_x[mask] = 0.5 + 0.18 * _attack_dir(team, h)
    centre = np.empty((n, 2))
    centre[:, 0] = _lowpass(target_x, 120.0) + 0.12 * _smooth_noise(rng, n, 1, 90.0)[:, 0]
    centre[:, 1] = 0.5 + 0.18 * _smooth_noise(rng, n, 1, 70.0)[:, 0]
    centre = np.clip(centre, [0.15, 0.2], [0.85, 0.8])

    positions = np.empty((n, 2, N_PLAYERS, 2))
    jitter = _smooth_noise(rng, n, 2 * N_PLAYERS * 2, 40.0).reshape(n, 2, N_PLAYERS, 2)
    for team in (H, A):
        for h in (1, 2):
            rows = half == h if h == 2 else half != 2
            direction = _attack_dir(team, h)
            base = FORMATION.copy()
            if direction < 0:
                base[:, 0] = 1.0 - base[:, 0]
            offset = (base - 0.5) * np.array([0.75, 0.85])
            push = np.where(gt[rows] == team, 0.04 * direction, -0.02 * direction)
            pos = centre[rows, None, :] + offset[None, :, :]
            pos[:, :, 0] += push[:, None]
            pos[:, 0, 0] = base[0, 0] + 0.03 * (centre[rows, 0] - 0.5)  # goalkeeper stays home
            pos[:, 0, 1] = 0.5 + 0.2 * (centre[rows, 1] - 0.5)
            positions[rows, team] = pos
    positions += 0.018 * jitter
    return np.clip(positions, 0.005, 0.995), centre


def _ball(
    rng: np.random.Generator, spec: MatchSpec, segments: list[Segment], positions: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """True ball (n, 2) and height (n,), driven by touches and passes between teammates."""
    fps = spec.fps
    n = spec.n_frames
    ball = np.full((n, 2), 0.5)
    height = np.zeros(n)
    last = np.array([0.5, 0.5])
    for seg in segments:
        if seg.label == D:
            ball[seg.start : seg.stop] = last
            continue
        team = seg.label
        t = seg.start
        dists = np.linalg.norm(positions[t, team, 1:] - last, axis=1)
        carrier = 1 + int(np.argmin(dists))
        # The ball travels to the new possessor first (interception / restart pass).
        flight = min(seg.stop - t, int(0.3 * fps))
        if flight > 0:
            w = np.linspace(0.0, 1.0, flight, endpoint=False)[:, None]
            ball[t : t + flight] = last + w * (positions[t + flight - 1, team, carrier] - last)
            t += flight
        while t < seg.stop:
            touch = min(seg.stop - t, max(3, int(rng.uniform(0.6, 2.5) * fps)))
            ball[t : t + touch] = positions[t : t + touch, team, carrier] + 0.004
            t += touch
            if t >= seg.stop:
                break
            here = positions[t, team, 1:]
            weights = np.exp(-np.linalg.norm(here - here[carrier - 1], axis=1) / 0.15)
            weights[carrier - 1] = 0.0
            nxt = 1 + int(rng.choice(N_PLAYERS - 1, p=weights / weights.sum()))
            flight = min(seg.stop - t, max(2, int(rng.uniform(0.3, 1.0) * fps)))
            start_xy = ball[t - 1]
            end_xy = positions[min(n - 1, t + flight), team, nxt]
            w = np.linspace(0.0, 1.0, flight, endpoint=False)[:, None]
            ball[t : t + flight] = start_xy + w * (end_xy - start_xy)
            if flight > 0.7 * fps:
                height[t : t + flight] = 4 * 3.0 * w[:, 0] * (1 - w[:, 0])
            t += flight
            carrier = nxt
        last = ball[seg.stop - 1].copy()
    return np.clip(ball, 0.0, 1.0), height


def _slots(
    rng: np.random.Generator, spec: MatchSpec, positions: FloatArray
) -> tuple[FloatArray, npt.NDArray[np.int8]]:
    """Observed player tracks in 46 slots with dropouts and identity switches. (n, 46, 2), slot_team."""
    n = spec.n_frames
    graded = spec.graded_mask()
    slot_team = np.array([H] * 23 + [A] * 23, dtype=np.int8)
    rng.shuffle(slot_team)  # team slots are interleaved, never assume contiguity
    tracks = np.full((n, N_SLOTS, 2), np.nan)
    for team in (H, A):
        team_slots = np.flatnonzero(slot_team == team)
        visible = np.stack([_renewal_mask(rng, n, 45.0 * 25, 1.2 * 25) for _ in range(N_PLAYERS)])
        visible[:, ~graded] &= rng.random((N_PLAYERS, int((~graded).sum()))) < 0.03
        mapping = list(team_slots[:N_PLAYERS])
        events = []
        for player in range(N_PLAYERS):
            starts = np.flatnonzero(np.diff(visible[player].astype(np.int8)) == 1) + 1
            events += [(int(f), player) for f in starts]
        events.sort()
        boundaries: list[tuple[int, list[int]]] = [(0, list(mapping))]
        for frame, player in events:
            if rng.random() < 0.6:
                spare = [s for s in team_slots if s not in mapping]
                mapping[player] = int(rng.choice(spare))
                boundaries.append((frame, list(mapping)))
        boundaries.append((n, list(mapping)))
        noise = 0.0015 * rng.standard_normal((n, N_PLAYERS, 2))
        for (start, current), (stop, _) in pairwise(boundaries):
            for player, slot in enumerate(current):
                rows = np.arange(start, stop)
                rows = rows[visible[player, start:stop]]
                tracks[rows, slot] = positions[rows, team, player] + noise[rows, player]
    return np.clip(tracks, 0.0, 1.0), slot_team


def _speed(tracks: FloatArray, spec: MatchSpec) -> FloatArray:
    metres = tracks * np.asarray(spec.pitch)
    speed = np.full(tracks.shape[:2], np.nan)
    speed[1:] = np.linalg.norm(np.diff(metres, axis=0), axis=2) * spec.fps
    return np.clip(speed, 0.0, 12.0)


# -- writers ----------------------------------------------------------------------------------


def _write_atd(
    path: Path, tracks: FloatArray, speed: FloatArray, ball: FloatArray, height: FloatArray
) -> None:
    n = tracks.shape[0]
    table = np.empty((n, 1 + N_SLOTS * 3 + 3))
    table[:, 0] = np.arange(1, n + 1)
    groups = np.concatenate([tracks, speed[:, :, None]], axis=2).reshape(n, -1)
    groups[np.repeat(np.isnan(tracks[:, :, 0]), 3, axis=1)] = np.nan
    table[:, 1 : 1 + N_SLOTS * 3] = groups
    table[:, -3:-1] = ball
    table[:, -1] = np.where(np.isnan(ball[:, 0]), np.nan, height)
    fmt = "%d:" + ";".join(["%.5f,%.5f,%.2f"] * N_SLOTS) + ":%.5f,%.5f,%.2f\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, n, 10_000):
            chunk = "".join(fmt % tuple(row) for row in table[start : start + 10_000])
            handle.write(chunk.replace("nan", "NaN"))


def _write_metadata(path: Path, slot_team: npt.NDArray[np.int8], spec: MatchSpec) -> None:
    ids = ("FIFATMA", "FIFATMB")
    players = "\n".join(
        f'      <Player id="P{i + 1}" teamId="{ids[int(t)]}"><Name>Track {i + 1}</Name>'
        f"<ShirtNumber>{i + 1}</ShirtNumber></Player>"
        for i, t in enumerate(slot_team)
    )
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FrameRate>{spec.fps}</FrameRate>
      <ProviderParameters>
        <ProviderParameter><Name>pitch_length</Name><Value>{spec.pitch[0]}</Value></ProviderParameter>
        <ProviderParameter><Name>pitch_width</Name><Value>{spec.pitch[1]}</Value></ProviderParameter>
        <ProviderParameter><Name>synthetic</Name><Value>true</Value></ProviderParameter>
      </ProviderParameters>
    </GlobalConfig>
    <Teams>
      <Team id="{ids[0]}"><Name>{escape(TEAM_NAMES[0])}</Name></Team>
      <Team id="{ids[1]}"><Name>{escape(TEAM_NAMES[1])}</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
  </Metadata>
</main>
"""
    path.write_text(xml, encoding="utf-8", newline="\n")


def _write_homography(
    path: Path, centre: FloatArray, spec: MatchSpec, rng: np.random.Generator
) -> None:
    n = spec.n_frames
    cx = np.clip(_lowpass(centre[:, 0], 50.0), 0.28, 0.72)
    sx, sy, persp = 1920 / 0.55, 1080 / 0.95, 0.06
    h = np.zeros((n, 9))
    h[:, 0] = sx
    h[:, 2] = 960 - sx * cx
    h[:, 4] = sy
    h[:, 5] = 540 - sy * 0.5
    h[:, 7] = persp
    h[:, 8] = 1.0
    error = np.round(0.2 + 0.08 * np.abs(rng.standard_normal(n)), 4)
    lines = ["frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"]
    lines += [f"{i}," + ",".join(f"{v:.7g}" for v in h[i]) + f",{error[i]},1" for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _tagging_xml(
    rng: np.random.Generator,
    spec: MatchSpec,
    segments: list[Segment],
    kickoffs: dict[int, int],
    centre: FloatArray,
) -> str:
    fps = spec.fps
    half_of = spec.half_of_frame()
    instances: list[tuple[float, float, str, dict[str, str]]] = []

    def add(start: int, stop: int, code: str, **tags: str) -> None:
        start_s, stop_s = max(0.0, start / fps), max(start / fps + 0.4, stop / fps)
        instances.append((start_s, stop_s, code, tags))

    def half_tag(frame: int) -> str:
        return "1st Half" if half_of[min(len(half_of) - 1, frame)] == 1 else "2nd Half"

    for half, frame in kickoffs.items():
        team = H if half == 1 else A
        add(
            frame - int(3.5 * fps),
            frame + int(1.5 * fps),
            "SET PIECES",
            Team=TEAM_TAGS[team],
            Half=half_tag(frame),
            Type="Kick Off (Start)",
        )
    for prev, seg in pairwise(segments):
        if prev.label == D and seg.label != D and seg.start not in kickoffs.values():
            kind = rng.choice(
                ["Throw-in", "Free Kick", "Goal Kick", "Corner Kick"], p=[0.45, 0.2, 0.15, 0.2]
            )
            team_tag = "N/A" if kind == "Throw-in" else TEAM_TAGS[seg.label]
            add(
                seg.start - int(3.5 * fps),
                seg.start + int(1.5 * fps),
                "SET PIECES",
                Team=team_tag,
                Half=half_tag(seg.start),
                Type=str(kind),
            )
    for seg in segments:
        if seg.label == D or seg.stop - seg.start < 4 * fps or rng.random() > 0.85:
            continue
        team = seg.label
        if rng.random() < 0.05:
            team = 1 - team
        h = int(half_of[seg.start]) or 1
        x = float(np.mean(centre[seg.start : seg.stop, 0]))
        progress = x if _attack_dir(seg.label, h) > 0 else 1 - x
        code = (
            "BUILD UP"
            if progress < 0.42
            else "PROGRESSION"
            if progress < 0.6
            else "CREATING CHANCES"
        )
        start = seg.start + int(rng.normal(0.8, 1.0) * fps)
        stop = seg.stop + int(rng.normal(0.0, 1.5) * fps)
        add(start, stop, code, Team=TEAM_TAGS[team], Half=half_tag(seg.start))
        if rng.random() < 0.5:
            third = "DEFENSIVE" if progress > 0.6 else "MIDDLE" if progress > 0.42 else "ATTACKING"
            add(
                start,
                stop,
                f"DEFENDING IN {third} THIRD",
                Team=TEAM_TAGS[1 - seg.label],
                Half=half_tag(seg.start),
            )
    for _ in range(max(1, len(instances) // 25)):  # false positives
        half, first, last = spec.halves[int(rng.integers(len(spec.halves)))]
        start = int(rng.integers(first, last - 10 * fps))
        add(
            start,
            start + int(rng.uniform(3, 10) * fps),
            str(rng.choice(["LONG BALL", "PROGRESSION"])),
            Team=TEAM_TAGS[int(rng.integers(2))],
            Half=half_tag(start),
        )

    instances.sort(key=lambda item: item[0])
    body = []
    for index, (start_s, stop_s, code, tags) in enumerate(instances, start=1):
        labels = "".join(
            f"<label><group>{escape(g)}</group><text>{escape(t)}</text></label>"
            for g, t in tags.items()
        )
        body.append(
            f"    <instance><ID>{index}</ID><start>{start_s:.2f}</start><end>{stop_s:.2f}</end>"
            f"<code>{escape(code)}</code>{labels}</instance>"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<file>\n  <ALL_INSTANCES>\n'
        + "\n".join(body)
        + "\n  </ALL_INSTANCES>\n</file>\n"
    )


def _write_gt(path: Path, gt: npt.NDArray[np.int8]) -> None:
    letters = np.array(["H", "A", "D", "X"])[gt]
    rows = np.char.add(np.char.add(np.arange(len(gt)).astype(str), ","), letters)
    path.write_text(
        "frame,label\n" + "\n".join(rows.tolist()) + "\n", encoding="utf-8", newline="\n"
    )


# -- entry point ------------------------------------------------------------------------------


def generate(out_dir: Path | str, seed: int = 0, mini: bool = False) -> dict[str, Any]:
    out = Path(out_dir)
    spec = mini_spec() if mini else full_spec()
    rng = np.random.default_rng(seed)

    gt, segments, kickoffs = _timeline(rng, spec, spec.fps)
    positions, centre = _players(rng, spec, gt)
    ball_true, height = _ball(rng, spec, segments, positions)
    tracks, slot_team = _slots(rng, spec, positions)
    speed = _speed(tracks, spec)

    graded = spec.graded_mask()
    seen = _renewal_mask(rng, spec.n_frames, 3.2 * 25, 2.2 * 25)
    seen[gt == D] &= rng.random(int((gt == D).sum())) < 0.75
    seen[~graded] &= rng.random(int((~graded).sum())) < 0.05
    ball_obs = np.where(
        seen[:, None], ball_true + 0.002 * rng.standard_normal(ball_true.shape), np.nan
    )
    ball_obs = np.clip(ball_obs, 0.0, 1.0)

    tactical = out / "tactical"
    synthetic = out / "synthetic"
    tactical.mkdir(parents=True, exist_ok=True)
    synthetic.mkdir(parents=True, exist_ok=True)
    stem = tactical / "BAR-ATH_tactical"
    _write_atd(Path(f"{stem}_atd.txt"), tracks, speed, ball_obs, height)
    _write_metadata(Path(f"{stem}_atd.xml"), slot_team, spec)
    _write_homography(Path(f"{stem}_homography.csv"), centre, spec, rng)
    Path(f"{stem}_smart_tagging.xml").write_text(
        _tagging_xml(rng, spec, segments, kickoffs, centre), encoding="utf-8", newline="\n"
    )

    public = np.full_like(gt, X)
    first = spec.halves[0][1]
    public[first : first + 5 * 60 * spec.fps] = gt[first : first + 5 * 60 * spec.fps]
    _write_gt(synthetic / "gt_full.csv", gt)
    _write_gt(synthetic / "gt_public.csv", public)

    alive_runs = [s for s in segments if s.label in (H, A)]
    stats = {
        "n_possessions": len(alive_runs),
        "mean_possession_s": round(
            float(np.mean([s.stop - s.start for s in alive_runs])) / spec.fps, 2
        ),
        "ball_visible_graded": round(float(seen[graded].mean()), 4),
        "label_counts_graded": {
            k: int(np.sum(gt[graded] == c)) for k, c in (("H", H), ("A", A), ("D", D))
        },
    }
    meta = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "mini": mini,
        "spec": spec.to_dict(),
        "kickoffs": kickoffs,
        "stats": stats,
        "warning": "SYNTHETIC DATA, not the real match. Scores here are smoke tests only.",
    }
    with open(synthetic / "meta.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2)
    (out / "README.md").write_text(
        "# Synthetic bundle\n\nGenerated by `mtah synthetic` "
        f"(seed {seed}, generator v{GENERATOR_VERSION}). Same file formats as the real data, fake "
        "content. Never use scores on it to choose between solutions.\n",
        encoding="utf-8",
        newline="\n",
    )
    return meta
