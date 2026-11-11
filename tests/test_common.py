import importlib.util
import sys

import numpy as np

from mtah import official
from mtah.common.ball import interpolate_gaps, speed
from mtah.common.segments import fill_abstentions, fill_graded_x, merge_short_spells, spells
from mtah.common.smoothing import majority_vote, viterbi
from mtah.common.tagging_priors import side_of_team_tag, team_votes
from mtah.data.tagging import TagEvent
from mtah.labels import A, D, H, X, to_codes
from mtah.paths import VENDOR_DIR
from mtah.spec import MatchSpec

SPEC = MatchSpec("T", 200, 25, (105.0, 68.0), ((1, 10, 99), (2, 120, 189)))


def test_majority_vote_matches_official_smooth():
    sys.path.insert(0, str(VENDOR_DIR))
    spec = importlib.util.spec_from_file_location("bl", VENDOR_DIR / "starter" / "baseline.py")
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    rng = np.random.default_rng(1)
    team = rng.choice([-1, 0, 1], size=5000, p=[0.4, 0.3, 0.3]).astype(np.int8)
    for window in (0, 3, 25, 50):
        assert np.array_equal(majority_vote(team, window), to_codes(bl.smooth(team, window)))


def test_viterbi_extremes():
    rng = np.random.default_rng(0)
    em = rng.normal(size=(300, 3))
    assert np.array_equal(viterbi(em, 0.0), em.argmax(axis=1))
    assert len(np.unique(viterbi(em, 1e9))) == 1
    em[:, :] = 0.0
    em[100:200, 1] = 5.0
    path = viterbi(em, 10.0)
    assert (path[100:200] == 1).all() and (path[:100] == path[0]).all()


def test_fill_matches_official_and_graded_x():
    codes = np.full(200, H, dtype=np.int8)
    codes[10:15] = X  # head of half 1 -> back-filled
    codes[50:60] = X
    codes[55] = A
    filled = fill_abstentions(codes, SPEC)
    expected = official.metrics().fill_abstentions(codes, [(10, 100), (120, 190)])
    assert np.array_equal(filled, expected)
    graded_only = fill_graded_x(codes, SPEC)
    assert (graded_only[SPEC.graded_mask()] != X).all()


def test_merge_short_spells():
    codes = np.full(200, H, dtype=np.int8)
    codes[40:43] = A  # 3-frame flicker
    codes[60:90] = A
    out = merge_short_spells(codes, SPEC, min_frames=5)
    assert (out[40:43] == H).all() and (out[60:90] == A).all()
    assert len(spells(out, SPEC)) == len(spells(codes, SPEC)) - 2


def test_ball_helpers():
    xy = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [3.0, 3.0], [np.nan, np.nan]])
    filled = interpolate_gaps(xy, max_gap=2)
    assert np.allclose(filled[1], [1.0, 1.0]) and np.isnan(filled[4]).all()
    assert np.isnan(interpolate_gaps(xy, max_gap=1)[1]).all()
    assert np.isclose(speed(np.array([[0.0, 0.0], [1.0, 0.0]]), 25)[1], 25.0)


def test_tagging_votes():
    events = [
        TagEvent("BUILD UP", 0, 1, 0, 10, {"Team": "Barcelona"}),
        TagEvent("DEFENDING IN MIDDLE THIRD", 0, 1, 5, 15, {"Team": "Barcelona"}),
        TagEvent("SET PIECES", 0, 1, 0, 20, {"Team": "N/A"}),
    ]
    votes = team_votes(events, 20)
    assert (
        votes[2].tolist() == [1, 0] and votes[7].tolist() == [1, 1] and votes[12].tolist() == [0, 1]
    )
    assert side_of_team_tag("Atletic Club") == A and side_of_team_tag("N/A") is None
    assert D == 2
