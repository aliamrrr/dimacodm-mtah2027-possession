"""mtah: frame-level ball possession for More than a Hack 2027, Phase 1.

The package is organised around one contract: a *solution* takes a `MatchData` and returns one
label code per video frame of the tactical video. Everything else (loading, validation, scoring,
manifests, reproducibility checks) is shared by every solution, so their outputs stay comparable.
"""

__version__ = "0.1.0"
