"""Runs, spells and abstention filling -- delegating to the official definitions where they exist."""

from __future__ import annotations

import numpy as np

from mtah import official
from mtah.labels import A, Codes, H, X
from mtah.spec import MatchSpec


def half_runs(spec: MatchSpec) -> list[tuple[int, int]]:
    """Half-open [start, stop) graded spans, the `runs` argument of the official functions."""
    return [(first, last + 1) for _, first, last in spec.halves]


def segments(codes: Codes, spec: MatchSpec) -> list[tuple[int, int, int]]:
    """[(start, stop, code)] constant-label runs inside each graded half (official definition)."""
    return list(official.metrics().segments(np.asarray(codes), half_runs(spec)))


def spells(codes: Codes, spec: MatchSpec) -> list[tuple[int, int, int]]:
    """Possessions: maximal H or A runs inside each half (official definition)."""
    return list(official.metrics().possession_spells(np.asarray(codes), half_runs(spec)))


def fill_abstentions(codes: Codes, spec: MatchSpec) -> Codes:
    """What spellF1 sees: X forward-filled (back-filled at the head) inside each half."""
    return np.asarray(
        official.metrics().fill_abstentions(np.asarray(codes, dtype=np.int8), half_runs(spec)),
        dtype=np.int8,
    )


def fill_graded_x(codes: Codes, spec: MatchSpec) -> Codes:
    """Replace X on graded frames by what the scorer would fill anyway.

    Identical spellF1 by construction, and never worse on macroF1 (which charges every X as a miss).
    Frames outside the halves are left untouched.
    """
    filled = fill_abstentions(codes, spec)
    out = np.asarray(codes, dtype=np.int8).copy()
    graded = spec.graded_mask()
    out[graded] = filled[graded]
    return out


def ungraded_to_x(codes: Codes, spec: MatchSpec) -> Codes:
    """Write X outside the graded halves (cosmetic: those frames are never scored)."""
    out = np.asarray(codes, dtype=np.int8).copy()
    out[~spec.graded_mask()] = X
    return out


def merge_short_spells(codes: Codes, spec: MatchSpec, min_frames: int) -> Codes:
    """Absorb H/A spells shorter than `min_frames` into the preceding label of the same half.

    Applied once, left to right, on the abstention-filled codes. A crude flicker filter: use it as a
    reference point, a proper decoder (see smoothing.viterbi) usually does better.
    """
    out = fill_abstentions(codes, spec)
    if min_frames <= 1:
        return out
    for start, stop in half_runs(spec):
        window = out[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            if window[a] in (H, A) and b - a < min_frames:
                window[a:b] = (
                    window[a - 1] if a > 0 else window[b] if b < len(window) else window[a]
                )
    return out
