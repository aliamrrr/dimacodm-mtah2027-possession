"""`mtah`: the command line of the project. `mtah --help` lists every command; README.md gives
the order to run them in."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mtah import official
from mtah.paths import OUTPUTS_DIR, REPO_ROOT, SUBMISSIONS_DIR, SYNTHETIC_DIR, data_dir

app = typer.Typer(
    help="More than a Hack 2027, Phase 1: frame-level ball possession.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
data_app = typer.Typer(help="Check and prepare the data bundle.", no_args_is_help=True)
viz_app = typer.Typer(
    help="Visualise submissions (written under outputs/, never committed).",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(viz_app, name="viz")
console = Console()

DataDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--data-dir",
        "-d",
        help="data bundle (default: $MTAH_DATA_DIR or ./data; "
        "`--data-dir data_synthetic` for the synthetic one)",
    ),
]


def _team_from_pyproject() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return str(tomllib.load(handle).get("tool", {}).get("mtah", {}).get("team", "CHANGE_ME"))


def _fail(message: str) -> None:
    console.print(f"[bold red]ERROR[/] {escape(message)}")
    raise typer.Exit(1)


def _scores_table(title: str, rows: list[tuple[str, dict[str, Any]]]) -> Table:
    table = Table(title=title, title_justify="left")
    for col in ("", "spellF1", "macroF1", "CI90 spell", "poss/truth", "duration s/truth", "X"):
        table.add_column(col, justify="right" if col else "left")
    for name, s in rows:
        ci = "--" if not s.get("spell_ci") else f"{s['spell_ci'][0]:.3f}-{s['spell_ci'][1]:.3f}"
        table.add_row(
            name,
            f"{s['spell_f1']:.4f}",
            f"{s['macro_f1']:.4f}",
            ci,
            f"{s['n_pred_spells']}/{s['n_gt_spells']}",
            f"{s['mean_spell_s']:.1f}/{s['gt_mean_spell_s']:.1f}",
            str(s["x_on_graded"]),
        )
    return table


# -- solutions ------------------------------------------------------------------------------


@app.command("list")
def list_solutions(
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
) -> None:
    """List the registered solutions (and those that fail to import)."""
    from mtah.solutions import broken, discover

    solutions = discover()
    failed = broken()
    if as_json:
        typer.echo(json.dumps({"solutions": sorted(solutions), "broken": sorted(failed)}))
        raise typer.Exit(1 if failed else 0)
    table = Table("solution", "authors", "summary")
    for name, cls in solutions.items():
        table.add_row(name, ", ".join(cls.authors), cls.summary)
    console.print(table)
    for name, error in failed.items():
        console.print(f"[red]{name} fails to import[/]:\n{escape(error)}")
    if failed:
        raise typer.Exit(1)


@app.command()
def run(
    solution: Annotated[str, typer.Argument(help="solution name (see `mtah list`)")],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="default: configs/<solution>.yaml")
    ] = None,
    param: Annotated[
        list[str] | None, typer.Option("--param", "-p", help="override key=value (repeatable)")
    ] = None,
    data: DataDirOpt = None,
    gt: Annotated[
        list[str] | None,
        typer.Option("--gt", help="public | synthetic | synthetic-public | path"),
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="default: date-time-commit")] = None,
    out: Annotated[Path, typer.Option(help="root folder of the runs")] = SUBMISSIONS_DIR,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="parse the tracking again")] = False,
) -> None:
    """Produce a submission: predict, validate (official validator), score, write the manifest."""
    from mtah.config import parse_overrides
    from mtah.pipeline import PipelineError
    from mtah.pipeline import run as run_pipeline

    try:
        result = run_pipeline(
            solution,
            config_path=config,
            overrides=parse_overrides(param or []),
            data_root=data,
            out_root=out,
            run_id=run_id,
            gts=tuple(gt) if gt else None,
            use_cache=not no_cache,
        )
    except PipelineError as error:
        _fail(str(error))
        return
    m = result.manifest
    o = m["output"]
    shown = (
        result.csv_path.relative_to(REPO_ROOT)
        if result.csv_path.is_relative_to(REPO_ROOT)
        else result.csv_path
    )
    console.print(
        f"[green]OK[/] {shown}  sha256 {o['sha256'][:12]}  "
        f"({m['runtime_s']} s, {m['data']['kind']} data)"
    )
    console.print(
        f"   {o['n_possessions']} possessions of {o['mean_possession_s']:.1f} s on average, "
        f"{o['short_possessions_lt_1s']} under 1 s, X on graded frames: {o['x_on_graded_frames']}"
    )
    console.print(_scores_table("scores", list(m["scores"].items())))
    if "public" in m["scores"]:
        console.print(
            "[yellow]Reminder[/]: the public 5 minutes hold 17 possessions. They check that the "
            "pipeline is correct; they are not for training, tuning or choosing between two "
            "versions (RULES.md)."
        )


# -- checking -------------------------------------------------------------------------------


@app.command()
def validate(
    csv: Annotated[Path, typer.Argument(help="a frame,label file")],
    rows: Annotated[int | None, typer.Option(help="default: 199319")] = None,
) -> None:
    """Validate a CSV with the unmodified OFFICIAL validator (before any submission)."""
    ok, output = official.run_validator(csv, rows)
    console.print(output)
    raise typer.Exit(0 if ok else 1)


@app.command("eval")
def evaluate_cmd(
    csv: Annotated[list[Path], typer.Argument(help="one or more CSV files")],
    gt: Annotated[
        str, typer.Option(help="public | synthetic | synthetic-public | path")
    ] = "public",
    bootstrap: Annotated[
        int, typer.Option(help="bootstrap draws (1000 before trusting a gap)")
    ] = 0,
    data: DataDirOpt = None,
) -> None:
    """Score CSV files with the official metric."""
    from mtah.evaluation import resolve_gt, score
    from mtah.submission import read_submission

    gt_path = resolve_gt(gt, data or (SYNTHETIC_DIR if gt.startswith("synthetic") else None))
    rows = [
        (
            p.parent.name if p.name == "submission.csv" else p.stem,
            score(read_submission(p), gt_path, bootstrap),
        )
        for p in csv
    ]
    console.print(_scores_table(f"ground truth: {gt}", rows))


def _latest_submissions(names: list[str] | None) -> dict[str, Path]:
    found = {}
    for path in sorted(SUBMISSIONS_DIR.glob("*/latest/submission.csv")):
        name = path.parent.parent.name
        if names is None or name in names:
            found[name] = path
    return found


@app.command("official-board")
def official_board(
    gt: Annotated[str, typer.Option(help="public or a path")] = "public",
    bootstrap: Annotated[int, typer.Option()] = 0,
) -> None:
    """The board of the OFFICIAL run_evaluation.py script on the latest run of each solution."""
    latest = _latest_submissions(None)
    if not latest:
        _fail("no run yet: `mtah run <solution>` first")
    out = OUTPUTS_DIR / "official_board"
    results = out / "results"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    for name, path in latest.items():
        shutil.copyfile(path, results / f"{name}.csv")
    raise typer.Exit(official.run_scoreboard(results, gt, out, bootstrap))


@app.command()
def reproduce(
    manifest: Annotated[Path, typer.Argument(help="submissions/<solution>/<run>/manifest.json")],
    data: DataDirOpt = None,
) -> None:
    """Replay a run from its manifest and check that the file is byte-identical (rule 3)."""
    from mtah.pipeline import reproduce as repro

    result = repro(manifest, data)
    for note in result.notes:
        console.print(f"[yellow]note[/] {note}")
    if result.ok:
        console.print(f"[green]REPRODUCED[/] sha256 {result.got[:16]} identical")
    else:
        _fail(f"DIFFERENT: expected {result.expected[:16]}, got {result.got[:16]}")


@app.command()
def final(
    solution: Annotated[str, typer.Argument()],
    team: Annotated[str | None, typer.Option(help="default: [tool.mtah] team")] = None,
    data: DataDirOpt = None,
    allow_dirty: bool = False,
    allow_synthetic: Annotated[bool, typer.Option(help="dry run on the synthetic match")] = False,
) -> None:
    """Submission file: clean git tree, two identical runs, official validator -> final/<team>.csv."""
    from mtah.pipeline import PipelineError
    from mtah.pipeline import final as make_final

    try:
        record = make_final(
            solution,
            team or _team_from_pyproject(),
            data_root=data,
            allow_dirty=allow_dirty,
            allow_synthetic=allow_synthetic,
        )
    except PipelineError as error:
        _fail(str(error))
        return
    console.print(f"[bold green]READY[/] final/{record['file']}  sha256 {record['sha256']}")
    console.print("final/SUBMISSION.json written: commit it with the submission.")


# -- data -----------------------------------------------------------------------------------

EXPECTED_REAL = {  # DATA.md, for the tactical recording
    "atd_records": 199_319,
    "slots": 46,
    "homography_rows": 199_319,
    "tagging_instances": 638,
    "ball_visible_all": 0.569,
    "ball_visible_graded": 0.593,
}


@data_app.command("check")
def data_check(data: DataDirOpt = None) -> None:
    """Check the bundle: files present, alignment, figures published in DATA.md."""
    from mtah.data import MatchData

    match = MatchData(data)
    console.print(f"bundle: {match.root}  ({'SYNTHETIC' if match.is_synthetic else 'real'})")
    missing = False
    for source in ("tactical", "broadcast"):
        for asset in ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"):
            try:
                path = match.asset_path(asset, source)
            except KeyError:
                continue
            present = path.is_file()
            optional = source == "broadcast" or asset == "video"
            missing |= not present and not optional
            mark = (
                "[green]ok[/]"
                if present
                else ("[yellow]missing (optional)[/]" if optional else "[red]MISSING[/]")
            )
            console.print(f"  {source:<9} {asset:<13} {mark}  {path.name}")
    if missing:
        _fail("required files are missing (see data/README.md)")

    from mtah.data.atd import count_records

    records, first_id, last_id, contiguous = count_records(match.asset_path("atd_raw"))
    console.print(
        f"  ATD frame_id {first_id}..{last_id}, contiguous: {contiguous} "
        "(video_frame = frame_id - 1)"
    )
    tr = match.tracking
    graded = match.spec.graded_mask()
    found = {
        "atd_records": records,
        "slots": tr.n_slots,
        "homography_rows": match.homography.n_frames,
        "tagging_instances": len(match.tagging),
        "ball_visible_all": round(float(tr.ball_visible.mean()), 3),
        "ball_visible_graded": round(float(tr.ball_visible[graded].mean()), 3),
    }
    table = Table("check", "found", "expected (DATA.md)")
    worst = False
    for key, value in found.items():
        expected = EXPECTED_REAL[key]
        same = abs(value - expected) < 0.002 if isinstance(expected, float) else value == expected
        worst |= not same and not match.is_synthetic
        table.add_row(
            key,
            str(value),
            str(expected) if not match.is_synthetic else "(synthetic)",
            style=None if same or match.is_synthetic else "red",
        )
    console.print(table)
    console.print(
        f"homography valid on {match.homography.valid.mean():.1%} of the frames, "
        f"teams {tr.team_names}"
    )
    if worst:
        console.print(
            "[yellow]Some figures differ from DATA.md: a different version of the bundle? "
            "Ask the organisers before building on it.[/]"
        )


@data_app.command("cache")
def data_cache(data: DataDirOpt = None) -> None:
    """Parse the tracking once (cached under .cache/, memory-mapped afterwards)."""
    from mtah.data import MatchData

    match = MatchData(data)
    tr = match.tracking
    console.print(
        f"[green]ok[/] {tr.n_frames} frames, {tr.n_slots} slots, "
        f"ball seen {tr.ball_visible.mean():.1%}"
    )
    if match.broadcast_tracking is not None:
        console.print(f"[green]ok[/] broadcast {match.broadcast_tracking.n_frames} frames")


@data_app.command("check-loader")
def check_loader(
    data: DataDirOpt = None,
    official_repo: Annotated[
        Path | None, typer.Option(help="clone of the official repository, for a per-frame diff")
    ] = None,
) -> None:
    """Proof that OUR loader equals the organisers': `dummy` rebuilds their four baselines, byte for byte."""
    from mtah.data import MatchData
    from mtah.solutions import get
    from mtah.submission import read_submission, write_submission
    from mtah.utils.repro import sha256_file

    match = MatchData(data)
    if match.is_synthetic:
        _fail("check-loader compares with the official files: it needs the REAL data")
    references = official.manifest()["reference_outputs"]
    windows = {
        "baseline_raw": 0,
        "baseline_smooth25": 25,
        "baseline_smooth50": 50,
        "baseline_smooth125": 125,
    }
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, window in windows.items():
            path = write_submission(
                Path(tmp) / f"{name}.csv", get("dummy")(window=window).predict(match)
            )
            ok = sha256_file(path) == references[name]["sha256"]
            all_ok &= ok
            line = f"  {name:<20} {'[green]identical[/]' if ok else '[red]DIFFERENT[/]'}"
            if not ok and official_repo:
                theirs = read_submission(official_repo / references[name]["upstream_path"])
                line += f"  ({int(np.sum(theirs != read_submission(path)))} frames differ)"
            console.print(line)
    if not all_ok:
        _fail("the loader does not reproduce the official baseline: do not build on it")
    console.print("[bold green]Loader validated[/]: same bytes as the four official baselines.")


# -- synthetic / doctor ---------------------------------------------------------------------


@app.command()
def synthetic(
    out: Annotated[Path, typer.Option(help="output folder")] = SYNTHETIC_DIR,
    seed: Annotated[int, typer.Option()] = 0,
    mini: Annotated[bool, typer.Option(help="short match (24 min) for the tests")] = False,
) -> None:
    """Generate a fake match in the exact official formats (to try the pipeline without the data)."""
    from mtah.synthetic import generate

    meta = generate(out, seed=seed, mini=mini)
    console.print(f"[green]ok[/] {out}  {meta['spec']['n_frames']} frames, stats {meta['stats']}")
    console.print(f"   mtah run dummy --data-dir {out}")


@app.command()
def doctor() -> None:
    """Diagnose the environment (run it when in doubt)."""
    import platform

    from mtah.solutions import broken, discover
    from mtah.utils import gitinfo

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        ("python 3.11", platform.python_version().startswith("3.11"), platform.python_version())
    )
    checks.append(("uv.lock present", (REPO_ROOT / "uv.lock").is_file(), "uv lock"))
    integrity = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sync_official.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    checks.append(
        (
            "official scorer untouched",
            integrity.returncode == 0,
            integrity.stdout.strip().splitlines()[-1] if integrity.stdout else "",
        )
    )
    git = gitinfo.info()
    checks.append(("git repository", git["commit"] is not None, git.get("branch") or "git init"))
    root = data_dir()
    checks.append(("real data", (root / "tactical").is_dir(), str(root)))
    checks.append(
        (
            "synthetic bundle",
            (SYNTHETIC_DIR / "synthetic" / "meta.json").is_file(),
            "mtah synthetic",
        )
    )
    team = _team_from_pyproject()
    checks.append(("team name set", team != "CHANGE_ME", "pyproject.toml [tool.mtah] team"))
    solutions = discover()
    checks.append((f"solutions ({len(solutions)})", not broken(), ", ".join(solutions)))
    for label, ok, hint in checks:
        console.print(
            f"  {'[green]ok[/]  ' if ok else '[yellow]todo[/]'} {label:<28} {escape(hint)}"
        )


# -- viz ------------------------------------------------------------------------------------


@viz_app.command("timeline")
def viz_timeline(
    csv: Annotated[list[Path], typer.Argument(help="submissions to draw")],
    start: Annotated[int, typer.Option(help="first frame")] = 11185,
    end: Annotated[int, typer.Option(help="last frame")] = 18684,
    gt: Annotated[str | None, typer.Option(help="public | synthetic | path")] = "public",
    data: DataDirOpt = None,
) -> None:
    """H/A/D timeline (PNG): ground truth and submissions over a range of frames."""
    from mtah.viz.timeline import plot_timeline

    path = plot_timeline(csv, start, end, gt, data)
    console.print(f"[green]ok[/] {path}")


@viz_app.command("clip")
def viz_clip(
    frame: Annotated[int, typer.Option(help="central frame (for example a missed transition)")],
    csv: Annotated[list[Path] | None, typer.Argument(help="submissions to overlay")] = None,
    seconds: Annotated[float, typer.Option(help="total duration")] = 8.0,
    gt: Annotated[str | None, typer.Option(help="public | synthetic | path")] = "public",
    data: DataDirOpt = None,
) -> None:
    """Annotated video clip: projected tracking and labels (local output, NEVER committed)."""
    from mtah.viz.clip import make_clip

    path = make_clip(frame, csv or [], seconds, gt, data)
    console.print(f"[green]ok[/] {path}  (derived from the data: do not share)")


if __name__ == "__main__":  # pragma: no cover
    app()
