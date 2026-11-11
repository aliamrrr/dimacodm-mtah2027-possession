import json

from typer.testing import CliRunner

from mtah import cli

runner = CliRunner()


def test_list_json():
    result = runner.invoke(cli.app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    assert "dummy" in json.loads(result.output)["solutions"]


def test_run_validate_eval(mini_bundle, tmp_path, monkeypatch):
    out = tmp_path / "subs"
    monkeypatch.setattr(cli, "SUBMISSIONS_DIR", out)
    result = runner.invoke(cli.app, ["run", "dummy", "-d", str(mini_bundle), "--out", str(out)])
    assert result.exit_code == 0, result.output
    csv = out / "dummy" / "latest" / "submission.csv"
    assert runner.invoke(cli.app, ["validate", str(csv), "--rows", "36000"]).exit_code == 0
    assert (
        runner.invoke(cli.app, ["validate", str(csv)]).exit_code == 1
    )  # wrong length for the real match
    result = runner.invoke(cli.app, ["eval", str(csv), "--gt", "synthetic", "-d", str(mini_bundle)])
    assert result.exit_code == 0 and "spellF1" in result.output


def test_param_override_and_bad_solution(mini_bundle, tmp_path):
    ok = runner.invoke(
        cli.app, ["run", "dummy", "-d", str(mini_bundle), "--out", str(tmp_path), "-p", "window=5"]
    )
    assert ok.exit_code == 0, ok.output
    bad = runner.invoke(cli.app, ["run", "nope", "-d", str(mini_bundle), "--out", str(tmp_path)])
    assert bad.exit_code != 0


def test_doctor_runs():
    assert runner.invoke(cli.app, ["doctor"]).exit_code == 0
