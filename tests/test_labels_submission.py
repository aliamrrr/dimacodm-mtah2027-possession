import numpy as np
import pytest

from mtah import official
from mtah.labels import A, D, H, X, check_codes, to_codes, to_letters
from mtah.spec import MatchSpec
from mtah.submission import describe, read_submission, validate_file, write_submission


def test_codes_roundtrip():
    letters = np.array(["H", "a", " D ", "X"])
    codes = to_codes(letters)
    assert codes.tolist() == [H, A, D, X]
    assert to_letters(codes).tolist() == ["H", "A", "D", "X"]
    with pytest.raises(ValueError):
        to_codes(["H", "Z"])


def test_check_codes_rejects_bad_outputs():
    with pytest.raises(ValueError, match="one per video frame"):
        check_codes(np.zeros(5, dtype=np.int8), n_frames=6)
    with pytest.raises(ValueError, match="integers"):
        check_codes(np.zeros(3, dtype=float))
    with pytest.raises(ValueError, match=r"0\.\.3"):
        check_codes(np.array([0, 4]))


def test_written_file_is_byte_exact(tmp_path):
    path = write_submission(tmp_path / "s.csv", np.array([H, A, D, X], dtype=np.int8))
    assert path.read_bytes() == b"frame,label\n0,H\n1,A\n2,D\n3,X\n"
    assert read_submission(path).tolist() == [H, A, D, X]


def test_official_validator_accepts_full_length_file(tmp_path):
    spec = MatchSpec.official()
    codes = np.full(spec.n_frames, D, dtype=np.int8)
    path = write_submission(tmp_path / "team.csv", codes)
    ok, output = official.run_validator(path)
    assert ok, output
    assert validate_file(path, spec.n_frames) is None


def test_validator_rejects_wrong_length_and_blank(tmp_path):
    path = write_submission(tmp_path / "short.csv", np.zeros(10, dtype=np.int8))
    assert "row count" in validate_file(path, 11)
    blank = tmp_path / "blank.csv"
    blank.write_text("frame,label\n0,H\n1,\n")
    assert "empty label" in validate_file(blank, 2)


def test_describe_counts_x_on_graded_frames():
    spec = MatchSpec("T", 100, 25, (105.0, 68.0), ((1, 10, 89),))
    codes = np.full(100, H, dtype=np.int8)
    codes[40:60] = A
    codes[50:55] = X
    info = describe(codes, spec)
    assert info["x_on_graded_frames"] == 5
    assert info["n_possessions"] == 3  # H, A (X filled), H
