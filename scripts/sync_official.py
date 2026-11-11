#!/usr/bin/env python3
"""Keep vendor/official/ an exact, verifiable copy of the organisers' repository.

The scorer you run locally is the one that decides the competition, so we never edit it: we copy
it byte for byte from https://github.com/metrica-sports/more-than-a-hack-2027 and record the
sha256 of every copied file in vendor/official/MANIFEST.json.

Standard library only, so it runs before any environment exists.

    python scripts/sync_official.py --check                 # vendored files == MANIFEST (offline)
    python scripts/sync_official.py --diff-upstream          # vendored files == upstream main?
    python scripts/sync_official.py --update                 # re-copy from upstream, rewrite MANIFEST
    python scripts/sync_official.py --update --upstream /path/to/local/clone

Exit code 1 means "something differs" for --check / --diff-upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "official"
MANIFEST = VENDOR / "MANIFEST.json"
UPSTREAM_URL = "https://github.com/metrica-sports/more-than-a-hack-2027.git"
UPSTREAM_BRANCH = "main"

# vendored path (relative to vendor/official) -> upstream path (relative to the upstream root)
FILES = {
    "evaluation/possession_metrics.py": "evaluation/possession_metrics.py",
    "evaluation/run_evaluation.py": "evaluation/run_evaluation.py",
    "evaluation/validate_submission.py": "evaluation/validate_submission.py",
    "evaluation/gt/BAR-ATH_possession_gt_public.csv": "evaluation/gt/BAR-ATH_possession_gt_public.csv",
    "evaluation/gt/BAR-ATH_possession_gt_public_meta.json": "evaluation/gt/BAR-ATH_possession_gt_public_meta.json",
    "games.py": "games.py",
    "starter/baseline.py": "starter/baseline.py",
    "LICENSE": "LICENSE",
    "docs/README.md": "README.md",
    "docs/CHALLENGE.md": "CHALLENGE.md",
    "docs/DATA.md": "DATA.md",
    "docs/EVALUATION.md": "EVALUATION.md",
    "docs/FAQ.md": "FAQ.md",
    "docs/OVERVIEW.md": "OVERVIEW.md",
    "docs/RULES.md": "RULES.md",
    "docs/SUBMISSION.md": "SUBMISSION.md",
}

# Upstream outputs we do not vendor but whose hashes we keep: they let `mtah data check-loader`
# prove, once the real data is unpacked, that our loader reproduces the official baseline exactly.
REFERENCE_OUTPUTS = {
    "baseline_raw": "evaluation/results/baseline_raw.csv",
    "baseline_smooth25": "evaluation/results/baseline_smooth25.csv",
    "baseline_smooth50": "evaluation/results/baseline_smooth50.csv",
    "baseline_smooth125": "evaluation/results/baseline_smooth125.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        sys.exit(
            "vendor/official/MANIFEST.json is missing -- run: python scripts/sync_official.py --update"
        )
    with open(MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def clone_upstream(target: Path) -> Path:
    # core.autocrlf=false: on Windows the default (true) would check the files out with CRLF, so
    # every sha256 would differ from upstream and --update would vendor altered bytes.
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "clone",
            "--config",
            "core.autocrlf=false",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            UPSTREAM_BRANCH,
            UPSTREAM_URL,
            str(target),
        ],
        check=True,
    )
    return target


def upstream_commit(clone: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def check_local() -> int:
    """Vendored bytes still match what the manifest recorded. Offline."""
    manifest = load_manifest()
    problems = []
    for rel, info in manifest["files"].items():
        path = VENDOR / rel
        if not path.is_file():
            problems.append(f"missing   vendor/official/{rel}")
        elif sha256(path) != info["sha256"]:
            problems.append(f"modified  vendor/official/{rel}")
    extra = sorted(set(FILES) - set(manifest["files"]))
    problems += [f"unlisted  vendor/official/{rel}" for rel in extra]
    if problems:
        print(
            "vendor/official/ differs from MANIFEST.json (the official scorer must stay untouched):"
        )
        print("\n".join("  " + p for p in problems))
        print("If upstream changed, run: python scripts/sync_official.py --update")
        return 1
    print(
        "OK  vendor/official/ matches MANIFEST.json (upstream {})".format(
            manifest["upstream_commit"][:10]
        )
    )
    return 0


def diff_upstream(upstream: Path | None) -> int:
    """Compare the manifest against upstream main. Used by the scheduled drift workflow."""
    manifest = load_manifest()
    with tempfile.TemporaryDirectory() as tmp:
        clone = upstream or clone_upstream(Path(tmp) / "upstream")
        changed = []
        for rel, upstream_rel in FILES.items():
            source = clone / upstream_rel
            recorded = manifest["files"].get(rel, {}).get("sha256")
            if not source.is_file():
                changed.append(("removed upstream", rel))
            elif sha256(source) != recorded:
                changed.append(("changed upstream", rel))
        for name, upstream_rel in REFERENCE_OUTPUTS.items():
            source = clone / upstream_rel
            recorded = manifest.get("reference_outputs", {}).get(name, {}).get("sha256")
            if source.is_file() and sha256(source) != recorded:
                changed.append(("changed upstream", upstream_rel))
        commit = upstream_commit(clone)
    report = {
        "upstream_commit": commit,
        "vendored_commit": manifest["upstream_commit"],
        "changes": [{"kind": kind, "path": path} for kind, path in changed],
    }
    print(json.dumps(report, indent=2))
    return 1 if changed else 0


def update(upstream: Path | None) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        clone = upstream or clone_upstream(Path(tmp) / "upstream")
        files = {}
        for rel, upstream_rel in FILES.items():
            source = clone / upstream_rel
            if not source.is_file():
                print(f"WARNING  {upstream_rel} not found upstream, keeping the old copy")
                continue
            target = VENDOR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            before = sha256(target) if target.is_file() else None
            shutil.copyfile(source, target)
            after = sha256(target)
            files[rel] = {"upstream_path": upstream_rel, "sha256": after}
            if before != after:
                print(f"updated  vendor/official/{rel}")
        references = {}
        for name, upstream_rel in REFERENCE_OUTPUTS.items():
            source = clone / upstream_rel
            if source.is_file():
                references[name] = {"upstream_path": upstream_rel, "sha256": sha256(source)}
        manifest = {
            "upstream_url": UPSTREAM_URL.removesuffix(".git"),
            "upstream_branch": UPSTREAM_BRANCH,
            "upstream_commit": upstream_commit(clone),
            "synced_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "note": "Exact copies. Never edit these files: the integrity test fails if a byte changes.",
            "files": files,
            "reference_outputs": references,
        }
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="vendored files match MANIFEST.json")
    mode.add_argument(
        "--diff-upstream", action="store_true", help="MANIFEST.json matches upstream main"
    )
    mode.add_argument(
        "--update", action="store_true", help="copy from upstream and rewrite MANIFEST.json"
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help=f"use a local clone instead of cloning {UPSTREAM_URL}",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.check:
        return check_local()
    if args.diff_upstream:
        return diff_upstream(args.upstream)
    return update(args.upstream)


if __name__ == "__main__":
    sys.exit(main())
