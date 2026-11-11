"""Turn Smart Tagging phases into per-frame evidence (a prior, never an answer)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from mtah.data.tagging import ATTACKING_CODES, DEFENDING_CODES, SET_PIECE_CODE, TagEvent
from mtah.labels import A, H

# Tag values as they appear in the file (DATA.md): "Barcelona" and "Atletic Club" (sic).
TEAM_TAG_TO_SIDE = {"barcelona": H, "fc barcelona": H, "atletic club": A, "athletic club": A}


def side_of_team_tag(tag: str | None) -> int | None:
    if tag is None:
        return None
    return TEAM_TAG_TO_SIDE.get(tag.strip().lower())


def team_votes(events: Iterable[TagEvent], n_frames: int) -> npt.NDArray[np.float32]:
    """(n, 2) votes for H / A in possession: attacking phases vote for their team, defending phases
    for the opponent. Unknown or N/A teams are ignored."""
    votes = np.zeros((n_frames, 2), dtype=np.float32)
    for event in events:
        side = side_of_team_tag(event.team)
        if side is None:
            continue
        if event.code in ATTACKING_CODES:
            target = side
        elif event.code in DEFENDING_CODES:
            target = 1 - side
        else:
            continue
        votes[max(0, event.start_frame) : max(0, min(n_frames, event.end_frame)), target] += 1.0
    return votes


def set_piece_windows(events: Iterable[TagEvent]) -> list[tuple[int, int, str | None, int | None]]:
    """[(start_frame, end_frame, type, side)] of tagged set pieces (windows AROUND the restart)."""
    return [
        (e.start_frame, e.end_frame, e.type, side_of_team_tag(e.team))
        for e in events
        if e.code == SET_PIECE_CODE
    ]
