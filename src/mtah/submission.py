"""The submission file: writing it byte-exactly, validating it with the official code, describing it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mtah import official
from mtah.common.segments import fill_abstentions, spells
from mtah.labels import Codes, X, check_codes, counts, to_letters
from mtah.spec import MatchSpec

HEADER = "frame,label\n"


def write_submission(path: Path, codes: Codes) -> Path:
    """Write `frame,label` with LF line endings: the same bytes as the official baseline writer."""
    codes = check_codes(codes)
    letters = to_letters(codes)
    frames = np.arange(len(codes)).astype(str)
    body = "\n".join(np.char.add(np.char.add(frames, ","), letters).tolist())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write((HEADER + body + "\n").encode("ascii"))
    return path


def read_submission(path: Path) -> Codes:
    pm = official.metrics()
    return np.asarray(pm.label_codes(pm.read_label_csv(path)), dtype=np.int8)


def validate_file(path: Path, n_expected: int) -> str | None:
    """None if valid, else the official one-line reason (same function the scorer calls)."""
    pm = official.metrics()
    try:
        table = pm.read_label_csv(path)
    except Exception as error:
        return f"could not read: {error}"
    reason = pm.validate_submission(table, n_expected)
    return None if reason is None else str(reason)


def describe(codes: Codes, spec: MatchSpec) -> dict[str, Any]:
    """What the file claims, as the scorer will see it (X filled inside the halves)."""
    codes = check_codes(codes, spec.n_frames)
    filled = fill_abstentions(codes, spec)
    possessions = spells(filled, spec)
    lengths = [b - a for a, b, _ in possessions]
    graded = spec.graded_mask()
    return {
        "rows": len(codes),
        "label_counts": counts(codes),
        "label_counts_graded": counts(codes[graded]),
        "x_on_graded_frames": int(np.sum(codes[graded] == X)),
        "n_possessions": len(possessions),
        "mean_possession_s": float(np.mean(lengths)) / spec.fps if lengths else 0.0,
        "short_possessions_lt_1s": int(sum(1 for n in lengths if n < spec.fps)),
    }
