"""Smart Tagging (SportsCode XML) -> intervals in video frames.

Times in the file are seconds from the first frame of the tactical video, so
``video_frame = round(seconds * fps)`` (verified by the organisers against both kickoffs).
Intervals are half-open: ``end_frame`` is the first frame AFTER the phase.

Remember (DATA.md): it is automatic, not ground truth; ``Team`` is ``N/A`` on every throw-in; the
intervals are phases, not possessions; set pieces are tagged as a window AROUND the restart.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

ATTACKING_CODES = frozenset(
    {
        "BUILD UP",
        "PROGRESSION",
        "CREATING CHANCES",
        "ATTACKING TRANSITION",
        "LONG BALL",
        "BALL IN FINAL THIRD",
        "BALL IN THE BOX",
        "PLAYERS IN THE BOX",
    }
)
DEFENDING_CODES = frozenset(
    {
        "DEFENDING IN ATTACKING THIRD",
        "DEFENDING IN MIDDLE THIRD",
        "DEFENDING IN DEFENSIVE THIRD",
        "DEFENSIVE TRANSITION",
    }
)
SET_PIECE_CODE = "SET PIECES"


@dataclass(frozen=True)
class TagEvent:
    code: str
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def team(self) -> str | None:
        """Team tag, or None when missing / 'N/A'."""
        value = self.tags.get("Team")
        return None if value is None or value.strip().upper() == "N/A" else value

    @property
    def type(self) -> str | None:
        return self.tags.get("Type")


def load_tagging(path: Path, fps: int) -> list[TagEvent]:
    events = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start") or "nan")
        end_s = float(instance.findtext("end") or "nan")
        tags = {}
        for label in instance.findall("label"):
            group, text = label.findtext("group"), label.findtext("text")
            if group is not None and text is not None:
                tags[group] = text
        events.append(
            TagEvent(
                code=(instance.findtext("code") or "").strip().upper(),
                start_s=start_s,
                end_s=end_s,
                start_frame=round(start_s * fps),
                end_frame=round(end_s * fps),
                tags=tags,
            )
        )
    return events


def to_frame(events: Iterable[TagEvent]) -> pd.DataFrame:
    rows = [
        {
            "code": e.code,
            "team": e.team,
            "type": e.type,
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "start_s": e.start_s,
            "end_s": e.end_s,
            **e.tags,
        }
        for e in events
    ]
    return pd.DataFrame(rows)


def interval_mask(
    events: Iterable[TagEvent],
    n_frames: int,
    codes: Iterable[str] | None = None,
    team: str | None = None,
) -> npt.NDArray[np.bool_]:
    """True on frames covered by at least one matching event (case-insensitive code match)."""
    wanted = None if codes is None else {c.upper() for c in codes}
    mask = np.zeros(n_frames, dtype=bool)
    for event in events:
        if wanted is not None and event.code not in wanted:
            continue
        if team is not None and (event.team or "").lower() != team.lower():
            continue
        mask[max(0, event.start_frame) : max(0, min(n_frames, event.end_frame))] = True
    return mask
