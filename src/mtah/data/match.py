"""MatchData: the single object every solution receives.

Everything is loaded lazily, so a solution only pays for the assets it actually touches:

    match.spec                  MatchSpec (n_frames, fps, halves, pitch)
    match.tracking              Tracking, tactical ATD (always the reference timeline)
    match.tagging               list[TagEvent], Smart Tagging
    match.homography            Homography, tactical
    match.broadcast_tracking    Tracking | None, broadcast ATD (NOT aligned to tactical)
    match.broadcast_homography  Homography | None
    match.video_path(source)    Path to the mp4 (may not exist)
    match.public_gt()           int8 codes of the 5 public minutes (X elsewhere) -- for checking
                                that the pipeline is correct ONLY: the rules forbid training,
                                tuning or choosing between versions on it. Never the private one.

``match.used_assets`` records what was touched; it ends up in the run manifest, which is what the
technical document must list ("which assets you used and how").
"""

from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from mtah import official
from mtah.data.atd import Tracking, load_tracking
from mtah.data.homography import Homography, load_homography
from mtah.data.tagging import TagEvent, load_tagging
from mtah.labels import Codes
from mtah.paths import PUBLIC_GT, data_dir
from mtah.spec import MatchSpec

_PARTIAL_HASH_BYTES = 4 << 20


class MissingAssetError(FileNotFoundError):
    pass


class MatchData:
    def __init__(
        self,
        root: Path | str | None = None,
        spec: MatchSpec | None = None,
        use_cache: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None else data_dir()
        self.spec = spec or MatchSpec.for_data_dir(self.root)
        self.use_cache = use_cache
        self.used_assets: set[str] = set()

    def __repr__(self) -> str:
        kind = "synthetic" if self.is_synthetic else "real"
        return f"MatchData({self.root}, {kind}, {self.spec.n_frames} frames)"

    @property
    def is_synthetic(self) -> bool:
        return self.spec.synthetic

    # -- paths ---------------------------------------------------------------------------------
    def asset_path(self, asset: str, source: str = "tactical") -> Path:
        paths = official.games().asset_paths(self.spec.game, str(self.root), source)
        if asset not in paths:
            raise KeyError(f"source {source!r} has no asset {asset!r}")
        return Path(paths[asset])

    def _require(self, asset: str, source: str = "tactical") -> Path:
        path = self.asset_path(asset, source)
        if not path.is_file():
            raise MissingAssetError(
                f"{path} not found.\nThe data is gated: unpack the bundle into {self.root} "
                "(or set MTAH_DATA_DIR). For a fake bundle: `mtah synthetic`."
            )
        self.used_assets.add(f"{source}:{asset}")
        return path

    def has(self, asset: str, source: str = "tactical") -> bool:
        try:
            return self.asset_path(asset, source).is_file()
        except KeyError:
            return False

    def video_path(self, source: str = "tactical") -> Path:
        self.used_assets.add(f"{source}:video")
        return self.asset_path("video", source)

    # -- assets --------------------------------------------------------------------------------
    @cached_property
    def tracking(self) -> Tracking:
        return load_tracking(
            self._require("atd_meta"),
            self._require("atd_raw"),
            self.spec.n_frames,
            source="tactical",
            use_cache=self.use_cache,
        )

    @cached_property
    def tagging(self) -> list[TagEvent]:
        return load_tagging(self._require("smart_tagging"), self.spec.fps)

    @cached_property
    def homography(self) -> Homography:
        return load_homography(self._require("homography"), self.spec.n_frames)

    @cached_property
    def broadcast_tracking(self) -> Tracking | None:
        if not (self.has("atd_meta", "broadcast") and self.has("atd_raw", "broadcast")):
            return None
        n = int(official.games().source_frames("broadcast", self.spec.game))
        return load_tracking(
            self._require("atd_meta", "broadcast"),
            self._require("atd_raw", "broadcast"),
            n,
            source="broadcast",
            use_cache=self.use_cache,
        )

    @cached_property
    def broadcast_homography(self) -> Homography | None:
        if not self.has("homography", "broadcast"):
            return None
        return load_homography(self._require("homography", "broadcast"))

    def public_gt(self) -> Codes:
        """Codes of the PUBLIC ground truth (5 minutes, X elsewhere).

        Pipeline checks only (frames aligned, labels meaning what you think). RULES.md: it is not
        training data and not a tuning set, so a solution's predict() must never read it.
        """
        self.used_assets.add("public_gt")
        path = self.root / "synthetic" / "gt_public.csv" if self.is_synthetic else PUBLIC_GT
        pm = official.metrics()
        return np.asarray(pm.label_codes(pm.read_label_csv(path)), dtype=np.int8)

    # -- provenance ----------------------------------------------------------------------------
    def fingerprint(self) -> dict[str, Any]:
        """Size + sha256 of every present asset (videos: partial hash). Goes into the manifest."""
        out: dict[str, Any] = {"root": str(self.root), "synthetic": self.is_synthetic}
        for source in ("tactical", "broadcast"):
            for asset, path_str in (
                official.games().asset_paths(self.spec.game, str(self.root), source).items()
            ):
                path = Path(path_str)
                if path.is_file():
                    out[f"{source}:{asset}"] = _file_fingerprint(path, partial=asset == "video")
        return out


def _file_fingerprint(path: Path, partial: bool) -> dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        if partial and size > 2 * _PARTIAL_HASH_BYTES:
            digest.update(handle.read(_PARTIAL_HASH_BYTES))
            handle.seek(-_PARTIAL_HASH_BYTES, 2)
            digest.update(handle.read(_PARTIAL_HASH_BYTES))
        else:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "partial": partial and size > 2 * _PARTIAL_HASH_BYTES,
    }
