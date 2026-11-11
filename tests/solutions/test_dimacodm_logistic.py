"""Unit tests of the `dimacodm_logistic` solution: labels of the other games, features, rules, decoder.

The generic contract (length, determinism, official validator) is tested for every solution
in test_solutions_contract.py. The `mini_match` fixture (a short synthetic match) comes from
tests/conftest.py.
"""

import numpy as np

from mtah.common.smoothing import viterbi
from mtah.labels import A, D, H, X
from mtah.solutions import get
from mtah.solutions.dimacodm import bench
from mtah.solutions.dimacodm import features as F
from mtah.solutions.dimacodm.model import (
    S_A,
    S_DA,
    S_DH,
    S_H,
    Weights,
    controlled_possession,
    decode,
    fit_logistic,
    transition_costs,
    viterbi_varying,
)


def test_ali_runs_on_mini_match(mini_match):
    codes = get("dimacodm_logistic")().predict(mini_match)
    assert codes.shape == (mini_match.spec.n_frames,)
    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    graded = mini_match.spec.graded_mask()
    assert (codes[~graded] == X).all() and (codes[graded] != X).all()


def test_weights_are_from_other_games_only():
    weights = Weights.load()
    assert weights.trained_on
    assert all(name.startswith(("metrica_", "dfl_")) for name in weights.trained_on)
    assert len(weights.dead) == len(F.DEAD_FEATURES) + 1
    assert len(weights.team) == len(F.TEAM_FEATURES) + 1 and weights.team[-1] == 0.0


# -- benchmark ground truth ----------------------------------------------------------------


def _event(team, kind, start, end=None, subtypes=(), period=1):
    return bench.Event(team, kind, tuple(subtypes), period, start, start if end is None else end)


def _game(events, n=200):
    return bench.ExternalGame(
        "toy",
        1,
        ((1, 1, n),),
        np.zeros((n, 2)),
        np.zeros((n, 2, 2)),
        np.array([H, A], dtype=np.int8),
        tuple(events),
    )


def test_labels_follow_possession_events_and_stoppages():
    events = [
        _event(H, "SET PIECE", 11, subtypes=("KICK OFF",)),
        _event(H, "PASS", 11, 20),
        _event(H, "BALL LOST", 40, 50, ("INTERCEPTION",)),
        _event(A, "RECOVERY", 50),
        _event(A, "CHALLENGE", 60, subtypes=("GROUND", "LOST")),  # a duel changes nothing
        _event(A, "BALL OUT", 70, 80),
        _event(H, "SET PIECE", 120, subtypes=("THROW IN",)),
        _event(H, "PASS", 120, 130),
        _event(A, "CHALLENGE", 150, subtypes=("TACKLE", "FAULT", "WON")),
        _event(H, "SET PIECE", 170, subtypes=("FREE KICK",)),
    ]
    codes = bench.derive_labels(_game(events))
    rows = lambda a, b: codes[a - 1 : b - 1]  # frame ids [a, b) -> rows  # noqa: E731
    assert (rows(1, 11) == D).all()  # before the kick-off
    assert (rows(11, 50) == H).all()  # H until the opponent recovers
    assert (rows(50, 80) == A).all()  # A, through the duel, until the ball crosses the line
    assert (rows(80, 120) == D).all()  # ball out -> throw-in
    assert (rows(120, 150) == H).all()
    assert (rows(150, 170) == D).all()  # foul -> free kick
    assert (rows(170, 201) == H).all()


def test_goal_kick_is_a_restart_and_advantage_is_not_a_stoppage():
    events = [
        _event(H, "PASS", 1, 10),
        _event(A, "CHALLENGE", 20, subtypes=("GROUND", "ADVANTAGE", "LOST")),
        _event(H, "SHOT", 30, 40, ("OFF TARGET", "OUT")),
        _event(A, "PASS", 90, 100, ("GOAL KICK",)),
    ]
    codes = bench.derive_labels(_game(events))
    assert (codes[0:39] == H).all() and (codes[39:89] == D).all() and (codes[89:] == A).all()


def test_subtypes_from_csv_and_json():
    assert bench._subtypes("HEAD-ON TARGET-GOAL") == ("HEAD", "ON TARGET", "GOAL")
    assert bench._subtypes({"name": "Throw In"}) == ("THROW IN",)
    assert bench._subtypes([{"name": "Aerial"}, {"name": "Won"}]) == ("AERIAL", "WON")
    assert bench._subtypes(None) == () and bench._subtypes(float("nan")) == ()


def test_what_stops_play():
    assert bench.stoppage_frame(_event(H, "BALL OUT", 10, 20)) == 20
    assert bench.stoppage_frame(_event(H, "SHOT", 10, 30, ("ON TARGET", "GOAL"))) == 30
    assert bench.stoppage_frame(_event(H, "SHOT", 10, 30, ("ON TARGET", "SAVED"))) is None
    foul = _event(A, "CHALLENGE", 40, 41, ("TACKLE", "FAULT", "WON"))
    assert bench.stoppage_frame(foul) == 40
    advantage = _event(A, "CHALLENGE", 40, 41, ("GROUND", "FAULT", "ADVANTAGE", "WON"))
    assert bench.stoppage_frame(advantage) is None
    assert bench.stoppage_frame(_event(H, "BALL LOST", 50, 60, ("OFFSIDE",))) == 60
    assert bench.stoppage_frame(_event(H, "PASS", 50, 60)) is None
    assert bench.is_restart(_event(A, "BALL LOST", 70, 80, ("GOAL KICK", "INTERCEPTION")))


def _skillcorner_files(folder):
    """A tiny fake SkillCorner match: 2 players, 2 periods of 30 frames at 10 fps."""
    import json

    match = {
        "pitch_length": 100,
        "pitch_width": 50,
        "home_team": {"id": 1},
        "away_team": {"id": 2},
        "players": [{"id": 10, "team_id": 1}, {"id": 20, "team_id": 2}],
    }
    (folder / "match.json").write_text(json.dumps(match))
    rows = []
    for period, first in ((1, 0), (2, 100)):
        for f in range(first, first + 30):
            rows.append(
                {
                    "frame": f,
                    "period": period,
                    "ball_data": {"x": float(f - first), "y": 0.0, "is_detected": f % 7 != 0},
                    "player_data": [
                        {"player_id": 10, "x": -10.0, "y": 0.0},
                        {"player_id": 20, "x": 10.0, "y": float(f - first)},
                    ],
                }
            )
    rows.insert(0, {"frame": -1, "period": None})  # pre-match frame, skipped
    (folder / "tracking.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    # period 1: home 0-9, a 3-frame gap (< 0.5 s: same team), home 13-19, a long gap (dead),
    # away 26-29; period 2: away 100-129
    phases = "period,frame_start,frame_end,team_in_possession_id\n"
    phases += "1,0,9,1\n1,13,19,1\n1,26,29,2\n2,100,129,2\n"
    (folder / "phases.csv").write_text(phases)
    return folder / "match.json", folder / "phases.csv", folder / "tracking.jsonl"


def test_skillcorner_parser_resamples_and_labels(tmp_path):
    ball, players, team, labels, spans = bench._parse_skillcorner(*_skillcorner_files(tmp_path))
    assert team.tolist() == [H, A]
    (a1, b1), (a2, b2) = spans
    assert b1 - a1 + 1 == int(29 * 2.5) + 1 and a2 - b1 - 1 == bench.HALF_GAP
    first = labels[a1 : b1 + 1]
    at = lambda f: first[round(f * 2.5)]  # noqa: E731 (10 fps frame -> 25 fps row)
    assert at(5) == H and at(11) == H  # the short gap keeps the team
    assert at(22) == D and at(27) == A
    assert (labels[a2 : b2 + 1] == A).all() and (labels[b1 + 1 : a2] == X).all()
    # metres from the centre -> 0-1; positions interpolated at 25 fps; undetected ball = NaN
    assert np.allclose(players[a1, 0], [0.4, 0.5]) and np.allclose(players[a1 + 5, 1, 1], 0.54)
    assert np.isnan(ball[a1, 0]) and np.isclose(ball[a1 + 5, 0], 0.52)


def test_skillcorner_dead_frames_are_never_targets():
    from mtah.solutions.dimacodm import deep_data

    n = 3000
    players = np.full((n, 22, 2), 0.5, dtype=np.float32)
    players += np.linspace(0, 0.2, 22, dtype=np.float32)[None, :, None]
    labels = np.full(n, H, dtype=np.int8)
    labels[1000:1600] = D
    for dead_targets in (True, False):
        game = deep_data.CleanGame(
            "sc_1" if not dead_targets else "dfl_1",
            2,
            players,
            np.array([H] * 11 + [A] * 11, dtype=np.int8),
            players[:, 0].copy(),
            labels,
            ((0, n),),
            dead_targets=dead_targets,
        )
        rnd = deep_data.Randomization(swap_p=0.0)
        state = deep_data.make_sample(game, 800, 1800, np.random.default_rng(0), rnd)["state"]
        assert ((state == D).sum() > 0) == dead_targets
        assert (state == H).sum() > 0


def test_bundle_spec_leaves_a_gap_between_adjacent_periods():
    game = bench.ExternalGame(
        "toy",
        1,
        ((1, 1, 100), (2, 101, 200)),
        np.zeros((200, 2)),
        np.zeros((200, 1, 2)),
        np.array([H], dtype=np.int8),
        (),
    )
    spec = bench.bundle_spec(game)
    assert spec.halves == ((1, 0, 98), (2, 100, 199))
    assert not spec.graded_mask()[99]


def test_degradation_is_deterministic_and_anonymous():
    n = 2000
    rng = np.random.default_rng(0)
    players = np.cumsum(rng.normal(0, 0.001, (n, 4, 2)), axis=0) + 0.5
    ball = players[:, 0].copy()
    game = bench.ExternalGame(
        "toy", 1, ((1, 1, n),), ball, players, np.array([H, H, A, A], dtype=np.int8), ()
    )
    codes = np.full(n, H, dtype=np.int8)
    codes[500:800] = D
    a = bench.degrade(game, codes, seed=3)
    b = bench.degrade(game, codes, seed=3)
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left, right, equal_nan=True)
    slots, ball_out, slot_team = a
    assert slots.shape == (n, bench.N_SLOTS, 2) and (np.bincount(slot_team) == 23).all()
    assert 0.2 < np.isnan(ball_out[:, 0]).mean() < 0.7


# -- features ------------------------------------------------------------------------------


def test_halves_separation_marks_the_kickoff_formation():
    slot_team = np.array([H, H, A, A], dtype=np.int8)
    kickoff = np.array([[[10, 30], [40, 20], [60, 30], [90, 40]]], dtype=float)
    mixed = np.array([[[10, 30], [70, 20], [60, 30], [20, 40]]], dtype=float)
    assert F.halves_separation(kickoff, slot_team, 100.0)[0] == 1.0
    assert F.halves_separation(mixed, slot_team, 100.0)[0] == 0.25


def test_rolling_helpers():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    assert np.allclose(F.rolling_mean(x, 1), [1.0, 2.0, 4.0, 4.0])
    assert F.rolling_max(np.array([0.0, 1.0, 0.0, 0.0]), 1).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_past_and_next_means_exclude_the_frame_itself():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    past, following = F.past_mean(x, 2), F.next_mean(x, 2)
    assert np.isnan(past[0]) and past[2] == 1.5 and past[4] == 3.5
    assert following[1] == 3.5 and np.isnan(following[3]) and np.isnan(following[4])


def test_team_features_are_antisymmetric():
    rng = np.random.default_rng(2)
    n = 50
    players = rng.uniform(0, 1, (n, 4, 2))
    ball = rng.uniform(0, 1, (n, 2))
    graded = np.ones(n, dtype=bool)
    teams = np.array([H, H, A, A], dtype=np.int8)
    team = F.frame_features(ball, players, teams, (105.0, 68.0), 25, graded).team
    swapped = F.frame_features(ball, players, 1 - teams, (105.0, 68.0), 25, graded).team
    assert np.allclose(team, -swapped)


# -- model ---------------------------------------------------------------------------------


def test_fit_logistic_recovers_a_known_model():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20_000, 2))
    y = rng.random(20_000) < 1 / (1 + np.exp(-(1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.5)))
    coef = fit_logistic(x, y, l2=0.0)
    assert np.allclose(coef, [1.5, -2.0, 0.5], atol=0.1)
    no_bias = fit_logistic(x, y, intercept=False)
    assert no_bias[-1] == 0.0


def test_varying_viterbi_is_the_constant_one_without_bonus():
    rng = np.random.default_rng(4)
    emission = rng.normal(size=(500, 4))
    costs = transition_costs(3.0, 2.0, 2.0, 1.0)
    zero = np.zeros(500)
    assert np.array_equal(viterbi_varying(emission, costs, zero, zero), viterbi(emission, costs))
    # A bonus where the players slow down moves the stoppage there.
    emission = np.zeros((200, 4))
    emission[:, 0] = 0.1
    emission[100:, 2] = emission[100:, 3] = 0.3
    stop = np.zeros(200)
    stop[60] = 50.0
    path = viterbi_varying(emission, transition_costs(5.0, 20.0, 5.0, 0.0), stop, zero)
    assert path[59] == S_H and path[60] == S_DH


def test_switch_bonus_opens_a_team_change_where_a_boundary_is_detected():
    emission = np.zeros((200, 4))
    emission[:, S_H], emission[:, S_A] = 0.02, -0.02
    emission[120:, S_H], emission[120:, S_A] = -0.02, 0.02
    costs = transition_costs(5.0, 50.0, 50.0, 0.0)
    zero = np.zeros(200)
    assert (viterbi_varying(emission, costs, zero, zero) == S_H).all()  # too weak to pay the cost
    switch = np.zeros(200)
    switch[120] = 5.0
    path = viterbi_varying(emission, costs, zero, zero, switch)
    assert (path[:120] == S_H).all() and (path[120:] == S_A).all()


def test_restart_prefers_the_other_team_when_nothing_says_otherwise():
    costs = transition_costs(5.0, 3.0, 3.0, 1.0)
    assert costs[S_DH, S_A] < costs[S_DH, S_H]
    assert costs[S_DH, S_DA] >= 1e9 and costs[S_H, S_DA] >= 1e9
    n = 300
    team = np.zeros(n)
    team[:100] = 4.0  # H clearly in control
    dead = np.full(n, -2.0)
    dead[100:200] = 4.0  # a stoppage, then no team evidence at all
    codes = decode(team, dead, [(0, n)], n, 1.0, 1.0, 0.0, costs)
    assert (codes[:100] == H).all() and (codes[100:200] == D).all() and (codes[200:] == A).all()


def test_nan_quantiles_is_bit_exact_with_numpy():
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 1.5, (3000, 46))
    x[rng.random(x.shape) < 0.55] = np.nan
    x[0] = np.nan  # no value
    x[1, 1:] = np.nan  # one value
    x[2, 2:] = np.nan  # two values
    qs = (0.5, 0.75, 0.9)
    ours = F.nan_quantiles(x, qs)
    has = ~np.isnan(x).all(axis=1)
    reference = np.full_like(ours, np.nan)
    reference[has] = np.nanquantile(x[has], qs, axis=1).T
    assert np.array_equal(ours, reference, equal_nan=True)


def _codes(text: str) -> np.ndarray:
    return np.array([{"H": H, "A": A, "D": D, "X": X}[c] for c in text], dtype=np.int8)


def _controlled(text: str, min_frames: int) -> str:
    out = controlled_possession(_codes(text), [(0, len(text))], min_frames)
    return "".join("HADX"[c] for c in out)


def test_a_touch_without_control_does_not_change_possession():
    # between two spells of the other team, and before a stoppage after the other team
    assert _controlled("HHHHAAHHHH", 3) == "HHHHHHHHHH"
    assert _controlled("HHHHAADDDD", 3) == "HHHHHHDDDD"
    # long enough: control was established
    assert _controlled("HHHHAAAHHH", 3) == "HHHHAAAHHH"
    # a restart has a clear possessor, however short its spell; so does the start of a half
    assert _controlled("DDDAAHHHHH", 3) == "DDDAAHHHHH"
    assert _controlled("AAHHHHHHHH", 3) == "AAHHHHHHHH"
    # the end of a half is neither the other team nor a stoppage
    assert _controlled("HHHHHHHHAA", 3) == "HHHHHHHHAA"


def test_controlled_possession_takes_the_shortest_touch_first_and_is_idempotent():
    # H then two touches (A 1 frame, H 2 frames) then A: the 1-frame touch goes first, which
    # leaves one long H spell and a real change of possession at the end
    assert _controlled("HHHHAHHAAAAA", 3) == "HHHHHHHAAAAA"
    text = "HHHHAHHADDDAAHAAAAHHDDHAAAAA"
    once = _controlled(text, 3)
    assert _controlled(once, 3) == once
    assert _controlled(text, 0) == text  # off: the labels are untouched


def test_controlled_possession_never_crosses_a_half_and_keeps_stoppages():
    codes = _codes("HHHHAAHHHHXXHHHHAADDAAAA")
    out = controlled_possession(codes, [(0, 10), (12, 24)], 3)
    assert "".join("HADX"[c] for c in out) == "HHHHHHHHHHXXHHHHHHDDAAAA"
    assert ((codes == D) == (out == D)).all() and ((codes == X) == (out == X)).all()


def test_min_spell_s_is_off_by_default_and_applies_the_rule_when_set():
    from mtah.solutions.dimacodm.deep_solution import DimacodmSolution
    from mtah.solutions.dimacodm.solution import DimacodmLogisticSolution, decode_logits

    assert "min_spell_s" not in DimacodmLogisticSolution.defaults
    assert (
        DimacodmSolution.defaults["min_spell_s"] == 0.0
    )  # the public reference keeps brief spells
    rng = np.random.default_rng(0)
    n = 4000
    team = np.cumsum(rng.normal(0, 0.6, n))
    team = 6 * np.sin(team / 4)
    dead = np.full(n, -6.0)
    trend = np.zeros(n)
    params = {**DimacodmLogisticSolution.defaults, "switch_cost": 1.0}
    raw = decode_logits(team, dead, trend, params, [(0, n)])
    ruled = decode_logits(team, dead, trend, {**params, "min_spell_s": 2.0}, [(0, n)], fps=25)
    assert (ruled == controlled_possession(raw, [(0, n)], 50)).all()
    assert (raw != ruled).any()  # the synthetic signal does contain short touches
