"""run -> validate -> score -> manifest. The only path from code to a submission file.

Every run writes

    submissions/<solution>/<run_id>/submission.csv
    submissions/<solution>/<run_id>/manifest.json   provenance: git, config, seed, data, env, sha256
    submissions/<solution>/latest/                  copy of the most recent run

`reproduce()` re-runs a manifest and compares bytes; `final()` is the submission-day procedure.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mtah import official
from mtah.config import RunConfig, load_config
from mtah.data.match import MatchData
from mtah.evaluation import resolve_gt, score
from mtah.labels import Codes, check_codes
from mtah.paths import FINAL_DIR, REPO_ROOT, SUBMISSIONS_DIR
from mtah.solutions import get
from mtah.spec import MatchSpec
from mtah.submission import describe, validate_file, write_submission
from mtah.utils import gitinfo
from mtah.utils.repro import environment, seed_everything, sha256_file

TEAM_PLACEHOLDER = "CHANGE_ME"


class PipelineError(RuntimeError):
    pass


@dataclass
class RunResult:
    solution: str
    run_dir: Path
    csv_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    codes: Codes

    @property
    def sha256(self) -> str:
        return str(self.manifest["output"]["sha256"])


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    short = gitinfo.info().get("short") or "nogit"
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{short}"


def predict(config: RunConfig, match: MatchData) -> Codes:
    seed_everything(config.seed)
    solution = get(config.solution)(**config.params)
    return check_codes(solution.predict(match), match.spec.n_frames)


def run(
    solution: str,
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    data_root: Path | None = None,
    out_root: Path = SUBMISSIONS_DIR,
    run_id: str | None = None,
    gts: tuple[str, ...] | None = None,
    use_cache: bool = True,
    update_latest: bool = True,
    official_validator: bool = True,
) -> RunResult:
    config = load_config(solution, config_path, overrides)
    match = MatchData(data_root, use_cache=use_cache)
    started = time.perf_counter()
    codes = predict(config, match)
    runtime = time.perf_counter() - started

    run_dir = Path(out_root) / solution / (run_id or default_run_id())
    csv_path = write_submission(run_dir / "submission.csv", codes)

    reason = validate_file(csv_path, match.spec.n_frames)
    if reason:
        raise PipelineError(f"{csv_path} failed the official validation: {reason}")
    validator_output = None
    if official_validator:
        ok, validator_output = official.run_validator(csv_path, rows=match.spec.n_frames)
        if not ok:
            raise PipelineError(
                f"official validate_submission.py rejected {csv_path}:\n{validator_output}"
            )

    if gts is None:
        gts = ("synthetic", "synthetic-public") if match.is_synthetic else ("public",)
    scores = {gt: score(codes, resolve_gt(gt, match.root)) for gt in gts}

    manifest = {
        "schema": 1,
        "solution": solution,
        "run_id": run_dir.name,
        "created_utc": _now(),
        "runtime_s": round(runtime, 2),
        "git": gitinfo.info(),
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "config_path": config.path and str(Path(config.path).resolve().relative_to(REPO_ROOT)),
        "data": {
            "kind": "synthetic" if match.is_synthetic else "real",
            "spec": match.spec.to_dict(),
            "used_assets": sorted(match.used_assets),
            "fingerprint": match.fingerprint(),
        },
        "environment": environment(),
        "output": {
            "path": str(csv_path.relative_to(REPO_ROOT))
            if csv_path.is_relative_to(REPO_ROOT)
            else str(csv_path),
            "sha256": sha256_file(csv_path),
            **describe(codes, match.spec),
        },
        "official_validator": "passed" if validator_output is not None else "skipped",
        "scores": scores,
        "reproduce": f"mtah reproduce {run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir}/manifest.json",
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, default=str)
        handle.write("\n")

    if update_latest:
        latest = Path(out_root) / solution / "latest"
        shutil.rmtree(latest, ignore_errors=True)
        shutil.copytree(run_dir, latest)
    return RunResult(solution, run_dir, csv_path, manifest_path, manifest, codes)


# -- reproduction ---------------------------------------------------------------------------


@dataclass
class ReproResult:
    ok: bool
    expected: str
    got: str
    notes: list[str]


def reproduce(
    manifest_path: Path, data_root: Path | None = None, use_cache: bool = False
) -> ReproResult:
    """Re-run a manifest (same solution, config, seed) from scratch and compare the bytes."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    notes = []
    current = gitinfo.info()
    recorded = manifest.get("git", {}).get("commit")
    if recorded and current.get("commit") and recorded != current["commit"]:
        notes.append(
            f"code changed since the run: manifest {recorded[:7]}, HEAD {current['short']}"
        )
    if current.get("dirty"):
        notes.append("working tree has uncommitted changes")

    cfg = manifest["config"]
    config = RunConfig(solution=cfg["solution"], seed=int(cfg["seed"]), params=dict(cfg["params"]))
    root = data_root if data_root is not None else Path(manifest["data"]["fingerprint"]["root"])
    match = MatchData(root, use_cache=use_cache)
    codes = predict(config, match)
    with tempfile.TemporaryDirectory() as tmp:
        got = sha256_file(write_submission(Path(tmp) / "submission.csv", codes))
    expected = str(manifest["output"]["sha256"])
    return ReproResult(got == expected, expected, got, notes)


# -- submission day -------------------------------------------------------------------------


def check_team_name(team: str) -> str:
    team = team.strip()
    if not team or team == TEAM_PLACEHOLDER:
        raise PipelineError("set the team name: --team NAME, or [tool.mtah] team in pyproject.toml")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}", team):
        raise PipelineError(
            f"team name {team!r} is not a safe file name (letters, digits, _ . - space)"
        )
    return team


def final(
    solution: str,
    team: str,
    *,
    data_root: Path | None = None,
    out_dir: Path = FINAL_DIR,
    allow_dirty: bool = False,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Produce <team>.csv the way the organisers will: clean tree, twice, official validator."""
    team = check_team_name(team)
    git = gitinfo.info()
    if git.get("dirty") and not allow_dirty:
        raise PipelineError(
            "uncommitted changes: commit first, the submitted file must be "
            f"regenerable from a commit ({git.get('dirty_files')})"
        )
    spec = MatchSpec.for_data_dir(MatchData(data_root).root)
    if spec.synthetic and not allow_synthetic:
        raise PipelineError("this is the SYNTHETIC bundle; point MTAH_DATA_DIR at the real data")

    with tempfile.TemporaryDirectory() as tmp:
        first = run(
            solution,
            data_root=data_root,
            out_root=Path(tmp) / "a",
            run_id="final",
            update_latest=False,
            use_cache=True,
        )
        second = run(
            solution,
            data_root=data_root,
            out_root=Path(tmp) / "b",
            run_id="final",
            update_latest=False,
            use_cache=False,
            official_validator=False,
        )
        if first.sha256 != second.sha256:
            raise PipelineError(
                "NOT DETERMINISTIC: two runs from the same commit differ "
                f"({first.sha256[:12]} vs {second.sha256[:12]}). Rule 3 would fail."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{team}.csv"
        shutil.copyfile(first.csv_path, target)

    ok, validator_output = official.run_validator(target, rows=spec.n_frames)
    if not ok:
        raise PipelineError(f"official validator rejected {target}:\n{validator_output}")
    record = {
        "team": team,
        "solution": solution,
        "file": target.name,
        "sha256": sha256_file(target),
        "rows": int(np.asarray(first.codes).size),
        "created_utc": _now(),
        "git": {k: git.get(k) for k in ("commit", "branch", "dirty", "remote")},
        "config": first.manifest["config"],
        "config_sha256": first.manifest["config_sha256"],
        "data_fingerprint": first.manifest["data"]["fingerprint"],
        "used_assets": first.manifest["data"]["used_assets"],
        "environment": first.manifest["environment"],
        "output": first.manifest["output"],
        "scores": first.manifest["scores"],
        "determinism": "two runs (cache on / cache off) produced identical bytes",
        "official_validator": validator_output.splitlines()[0] if validator_output else "",
        "reproduce": f'mtah final {solution} --team "{team}"   # then compare sha256',
    }
    with open(out_dir / "SUBMISSION.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, default=str)
        handle.write("\n")
    return record
