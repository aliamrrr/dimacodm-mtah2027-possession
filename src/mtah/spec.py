"""MatchSpec: the frame geometry of a match (length, fps, graded halves, pitch size).

The real spec is read from the vendored official games.py, never re-typed. A synthetic bundle
carries its own spec in synthetic/meta.json so short test matches work with the same code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mtah import official

SYNTHETIC_META = Path("synthetic") / "meta.json"


@dataclass(frozen=True)
class MatchSpec:
    game: str
    n_frames: int
    fps: int
    pitch: tuple[float, float]  # (length, width) in metres
    halves: tuple[tuple[int, int, int], ...]  # (half, first_frame, last_frame), inclusive
    synthetic: bool = False

    @classmethod
    def official(cls, game: str = "BAR-ATH") -> MatchSpec:
        g = official.games()
        entry = g.GAMES[game]
        return cls(
            game=game,
            n_frames=int(entry["n_video_frames"]),
            fps=int(entry["fps"]),
            pitch=(float(entry["pitch"][0]), float(entry["pitch"][1])),
            halves=tuple((int(h), int(a), int(b)) for h, a, b in g.half_video_blocks(game)),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchSpec:
        return cls(
            game=str(data["game"]),
            n_frames=int(data["n_frames"]),
            fps=int(data["fps"]),
            pitch=(float(data["pitch"][0]), float(data["pitch"][1])),
            halves=tuple((int(h), int(a), int(b)) for h, a, b in data["halves"]),
            synthetic=bool(data.get("synthetic", False)),
        )

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> MatchSpec:
        """Synthetic spec if the bundle carries one, otherwise the official BAR-ATH spec."""
        meta = Path(data_dir) / SYNTHETIC_META
        if meta.is_file():
            with open(meta, encoding="utf-8") as handle:
                return cls.from_dict(json.load(handle)["spec"])
        return cls.official()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- frame masks ---------------------------------------------------------------------------
    def half_of_frame(self) -> npt.NDArray[np.int8]:
        """0 outside the graded halves, else 1 or 2."""
        out = np.zeros(self.n_frames, dtype=np.int8)
        for half, first, last in self.halves:
            out[first : last + 1] = half
        return out

    def graded_mask(self) -> npt.NDArray[np.bool_]:
        """True on the frames the official ground truth grades (both halves)."""
        return self.half_of_frame() > 0

    def half_slices(self) -> list[slice]:
        return [slice(first, last + 1) for _, first, last in self.halves]

    @property
    def n_graded(self) -> int:
        return int(sum(last - first + 1 for _, first, last in self.halves))
