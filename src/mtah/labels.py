"""Label alphabet. Codes are identical to the official scorer's (possession_metrics.py)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
_LETTERS = np.array(LABELS)

Codes = npt.NDArray[np.int8]


def to_codes(labels: npt.ArrayLike) -> Codes:
    """'H'/'A'/'D'/'X' strings (any case, stripped) -> int8 codes."""
    values = np.char.upper(np.char.strip(np.asarray(labels, dtype=str)))
    codes = np.full(values.shape, -1, dtype=np.int8)
    for code, letter in enumerate(LABELS):
        codes[values == letter] = code
    if (codes < 0).any():
        bad = sorted(set(values[codes < 0].tolist()))[:5]
        raise ValueError(f"labels outside {LABELS}: {bad}")
    return codes


def to_letters(codes: npt.ArrayLike) -> npt.NDArray[np.str_]:
    codes = np.asarray(codes)
    check_codes(codes)
    return _LETTERS[codes.astype(np.intp)]


def check_codes(codes: npt.ArrayLike, n_frames: int | None = None) -> Codes:
    """Return codes as int8 or raise with a readable reason. Used on every solution output."""
    arr = np.asarray(codes)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D array of codes, got shape {arr.shape}")
    if n_frames is not None and len(arr) != n_frames:
        raise ValueError(f"expected {n_frames} codes (one per video frame), got {len(arr)}")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"codes must be integers (H=0, A=1, D=2, X=3), got dtype {arr.dtype}")
    if len(arr) and (arr.min() < 0 or arr.max() > X):
        raise ValueError("codes must be in 0..3 (H=0, A=1, D=2, X=3)")
    return arr.astype(np.int8, copy=False)


def counts(codes: npt.ArrayLike) -> dict[str, int]:
    arr = np.asarray(codes)
    return {letter: int(np.sum(arr == code)) for code, letter in enumerate(LABELS)}
