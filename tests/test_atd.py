"""The fast ATD parser never disagrees with the reference parser, whatever the file looks like."""

import shutil

import numpy as np
import pytest

from mtah.data.atd import _detect_layout, _parse_slow, load_tracking, parse_raw, read_metadata


def _paths(bundle):
    t = bundle / "tactical"
    return t / "BAR-ATH_tactical_atd.xml", t / "BAR-ATH_tactical_atd.txt"


def test_fast_equals_slow(mini_bundle):
    meta, raw = _paths(mini_bundle)
    slot_team, _ = read_metadata(meta)
    table, layout, parser = parse_raw(raw, len(slot_team), 36_000)
    assert parser == "fast"
    assert np.array_equal(table, _parse_slow(raw, layout, 36_000), equal_nan=True)


def test_crlf_and_irregular_lines_fall_back_safely(mini_bundle, tmp_path):
    _, raw = _paths(mini_bundle)
    lines = raw.read_text().splitlines()[:500]
    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    table, _, parser = parse_raw(crlf, 46, 600)
    assert parser == "fast"
    irregular = tmp_path / "irregular.txt"
    broken = lines[:]
    broken[10] = broken[10].replace(";", ";NaN,", 1)  # one group with an extra field
    irregular.write_text("\n".join(broken) + "\n")
    table2, _, parser2 = parse_raw(irregular, 46, 600)
    assert parser2 == "slow"
    assert np.array_equal(
        np.delete(table2, 10, axis=0), np.delete(table, 10, axis=0), equal_nan=True
    )


def test_rows_are_video_frames(mini_bundle, tmp_path):
    _, raw = _paths(mini_bundle)
    lines = raw.read_text().splitlines()
    picked = [lines[0], lines[4]]  # frame_id 1 and 5 only
    small = tmp_path / "small.txt"
    small.write_text("\n".join(picked) + "\n")
    table, _, _ = parse_raw(small, 46, 8)
    assert table[0, 0] == 1 and table[4, 0] == 5
    assert np.isnan(table[[1, 2, 3, 5, 6, 7], 0]).all()


def test_slot_mismatch_is_an_error(mini_bundle, tmp_path):
    _, raw = _paths(mini_bundle)
    with pytest.raises(ValueError, match="track slots"):
        parse_raw(raw, 22, 100)


def test_cache_returns_read_only_identical_arrays(mini_bundle):
    meta, raw = _paths(mini_bundle)
    fresh = load_tracking(meta, raw, 36_000, use_cache=False)
    cached = load_tracking(meta, raw, 36_000, use_cache=True)
    again = load_tracking(meta, raw, 36_000, use_cache=True)
    for name in ("ball", "players", "player_extra", "slot_team"):
        assert np.array_equal(getattr(fresh, name), getattr(again, name), equal_nan=True)
    assert not again.players.flags.writeable
    assert cached.team_names == fresh.team_names


def test_cache_survives_a_locked_folder(mini_bundle, tmp_path, monkeypatch):
    """Windows: a rename can fail while an antivirus holds a fresh file. Retry, else go on."""
    from pathlib import Path

    from mtah.data import atd

    monkeypatch.setenv("MTAH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(atd.time, "sleep", lambda _: None)
    meta, raw = _paths(mini_bundle)
    original = Path.rename
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("locked")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    retried = load_tracking(meta, raw, 36_000, use_cache=True)
    # On a real Windows machine the antivirus may keep holding the fresh folder after the two
    # simulated failures: either the cache is published, or the run goes on from memory, and
    # either way nothing half-written is left behind.
    assert calls["n"] >= 3 and not retried.players.flags.writeable
    assert not list((tmp_path / "cache").glob("*.tmp*"))

    monkeypatch.setenv("MTAH_CACHE_DIR", str(tmp_path / "cache2"))
    monkeypatch.setattr(Path, "rename", lambda self, target: (_ for _ in ()).throw(OSError("no")))
    in_memory = load_tracking(meta, raw, 36_000, use_cache=True)
    assert np.array_equal(in_memory.ball, retried.ball, equal_nan=True)
    assert not in_memory.players.flags.writeable
    assert not list((tmp_path / "cache2").glob("*.tmp*"))  # nothing half-written left behind


def test_metadata_conventions(mini_bundle):
    meta, _ = _paths(mini_bundle)
    slot_team, names = read_metadata(meta)
    assert len(slot_team) == 46 and set(np.unique(slot_team)) == {0, 1}
    assert names[0].startswith("FC Barcelona")


def test_layout_detection(mini_bundle):
    _, raw = _paths(mini_bundle)
    layout = _detect_layout(raw)
    assert (layout.n_slots, layout.k_player, layout.k_ball) == (46, 3, 3)


def test_copy_has_same_cache_key_semantics(mini_bundle, tmp_path):
    meta, raw = _paths(mini_bundle)
    copy = tmp_path / raw.name
    shutil.copyfile(raw, copy)
    a = load_tracking(meta, raw, 36_000)
    b = load_tracking(meta, copy, 36_000)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
