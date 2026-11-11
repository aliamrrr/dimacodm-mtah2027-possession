"""Ball trajectory helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def interpolate_gaps(xy: npt.NDArray[np.floating], max_gap: int) -> npt.NDArray[np.float64]:
    """Linearly fill NaN gaps of at most `max_gap` frames bounded by observations on both sides."""
    out = np.array(xy, dtype=np.float64, copy=True)
    missing = np.isnan(out[:, 0])
    if not missing.any() or max_gap <= 0:
        return out
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops, strict=True):
        if start == 0 or stop >= len(out) or stop - start > max_gap:
            continue
        before, after = out[start - 1], out[stop]
        t = (np.arange(start, stop) - (start - 1)) / (stop - start + 1)
        out[start:stop] = before + t[:, None] * (after - before)
    return out


def speed(xy_m: npt.NDArray[np.floating], fps: int) -> npt.NDArray[np.float64]:
    """Frame-to-frame speed in m/s (NaN where either frame is missing). Same length as input."""
    xy_m = np.asarray(xy_m, dtype=np.float64)
    out = np.full(len(xy_m), np.nan)
    out[1:] = np.linalg.norm(np.diff(xy_m, axis=0), axis=1) * fps
    return out
