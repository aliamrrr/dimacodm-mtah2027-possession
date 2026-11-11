import os

from mtah import paths


def test_dotenv_fills_missing_settings_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nMTAH_DATA_DIR=./bundle\nMTAH_CACHE_DIR=/elsewhere\nNOT_MTAH_KEY=1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    monkeypatch.delenv("NOT_MTAH_KEY", raising=False)
    monkeypatch.setenv("MTAH_CACHE_DIR", "/from/the/shell")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    paths._load_dotenv(env)
    assert os.environ["MTAH_DATA_DIR"] == str((tmp_path / "bundle").resolve())
    assert paths.data_dir() == (tmp_path / "bundle").resolve()
    assert os.environ["MTAH_CACHE_DIR"] == "/from/the/shell"  # the shell wins
    assert "NOT_MTAH_KEY" not in os.environ


def test_missing_dotenv_is_fine(tmp_path):
    paths._load_dotenv(tmp_path / ".env")
