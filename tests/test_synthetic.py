import hashlib

import numpy as np

from mtah.data import MatchData
from mtah.submission import read_submission, validate_file
from mtah.synthetic import generate


def _digest(root):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.*"))
        if p.suffix != ".md"
    }


def test_same_seed_same_bytes(tmp_path, mini_bundle):
    generate(tmp_path / "again", seed=0, mini=True)
    assert _digest(tmp_path / "again") == _digest(mini_bundle)


def test_bundle_is_realistic(mini_match, mini_bundle):
    tr = mini_match.tracking
    graded = mini_match.spec.graded_mask()
    assert 0.45 < tr.ball_visible[graded].mean() < 0.7
    assert 18 < np.nanmean((~np.isnan(tr.players[graded, :, 0])).sum(axis=1)) <= 22
    gt = read_submission(mini_bundle / "synthetic" / "gt_full.csv")
    assert validate_file(mini_bundle / "synthetic" / "gt_full.csv", 36_000) is None
    assert (gt[~graded] == 3).all() and (gt[graded] != 3).all()
    assert len(mini_match.tagging) > 20 and mini_match.homography.valid.all()


def test_public_gt_is_five_minutes(mini_match):
    public = mini_match.public_gt()
    assert int((public != 3).sum()) == 5 * 60 * 25


def test_different_seed_differs(tmp_path):
    a = generate(tmp_path / "a", seed=1, mini=True)
    b = generate(tmp_path / "b", seed=2, mini=True)
    assert a["stats"] != b["stats"]
    assert MatchData(tmp_path / "a").is_synthetic
