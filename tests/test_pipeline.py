import json

import pytest

from mtah.pipeline import PipelineError, check_team_name, final, reproduce, run


@pytest.fixture()
def dummy_run(mini_bundle, tmp_path):
    return run("dummy", data_root=mini_bundle, out_root=tmp_path / "subs", run_id="t1")


def test_run_writes_manifest_and_latest(dummy_run, tmp_path):
    m = dummy_run.manifest
    assert m["solution"] == "dummy" and m["config"]["params"]["window"] == 50
    assert m["data"]["kind"] == "synthetic"
    assert "tactical:atd_raw" in m["data"]["used_assets"]
    assert set(m["scores"]) == {"synthetic", "synthetic-public"}
    assert m["official_validator"] == "passed"
    assert (tmp_path / "subs" / "dummy" / "latest" / "submission.csv").is_file()
    on_disk = json.loads(dummy_run.manifest_path.read_text())
    assert on_disk["output"]["sha256"] == dummy_run.sha256


def test_reproduce_detects_identity_and_tampering(dummy_run, mini_bundle):
    assert reproduce(dummy_run.manifest_path, mini_bundle).ok
    manifest = json.loads(dummy_run.manifest_path.read_text())
    manifest["config"]["params"]["window"] = 25
    dummy_run.manifest_path.write_text(json.dumps(manifest))
    assert not reproduce(dummy_run.manifest_path, mini_bundle).ok


def test_overrides_change_the_output(mini_bundle, tmp_path):
    a = run("dummy", data_root=mini_bundle, out_root=tmp_path, run_id="a", official_validator=False)
    b = run(
        "dummy",
        data_root=mini_bundle,
        out_root=tmp_path,
        run_id="b",
        overrides={"window": 10},
        official_validator=False,
    )
    assert a.sha256 != b.sha256 and b.manifest["config"]["params"]["window"] == 10


def test_final_dry_run(mini_bundle, tmp_path):
    record = final(
        "dummy",
        "Team-Test",
        data_root=mini_bundle,
        out_dir=tmp_path / "final",
        allow_dirty=True,
        allow_synthetic=True,
    )
    assert (tmp_path / "final" / "Team-Test.csv").is_file()
    assert (
        json.loads((tmp_path / "final" / "SUBMISSION.json").read_text())["sha256"]
        == record["sha256"]
    )


def test_final_refuses_synthetic_and_placeholder(mini_bundle, tmp_path):
    with pytest.raises(PipelineError, match="SYNTHETIC"):
        final("dummy", "Team", data_root=mini_bundle, out_dir=tmp_path, allow_dirty=True)
    with pytest.raises(PipelineError, match="team name"):
        check_team_name("CHANGE_ME")
    with pytest.raises(PipelineError, match="safe file name"):
        check_team_name("../evil")


def test_unknown_param_is_rejected(mini_bundle, tmp_path):
    with pytest.raises(ValueError, match="unknown params"):
        run("dummy", data_root=mini_bundle, out_root=tmp_path, overrides={"windw": 3})


def test_git_dirty_files_are_reported_with_full_paths(tmp_path, monkeypatch):
    import subprocess

    from mtah.utils import gitinfo

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    (repo / "src" / "a.py").write_text("x = 2\n")
    monkeypatch.setattr(gitinfo, "REPO_ROOT", repo)
    info = gitinfo.info()
    assert info["dirty"] is True and info["dirty_files"] == ["src/a.py"]


def test_untracked_code_makes_the_tree_dirty(tmp_path, monkeypatch):
    """A solution nobody committed would run in `mtah final` without being in the commit."""
    import subprocess

    from mtah.utils import gitinfo

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    (repo / "notes.txt").write_text("outside the code paths: ignored\n")
    monkeypatch.setattr(gitinfo, "REPO_ROOT", repo)
    assert gitinfo.info()["dirty"] is False
    (repo / "src" / "new_solution.py").write_text("y = 2\n")
    info = gitinfo.info()
    assert info["dirty"] is True and info["dirty_files"] == ["src/new_solution.py"]
