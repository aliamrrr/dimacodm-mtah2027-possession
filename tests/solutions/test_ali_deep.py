"""The deep model: numpy inference == torch training forward, features, degradation, symmetry."""

import numpy as np
import pytest

from mtah.labels import A, D, H
from mtah.solutions.ali import deep_data
from mtah.solutions.ali.deep_features import (
    COUNT_COLUMNS,
    N_GLOBAL,
    N_TOKEN,
    TEAM_COLUMN,
    inputs,
    swap_teams,
)
from mtah.solutions.ali.deep_net import DeepConfig, NumpyNet, ensemble_logits

SMALL = DeepConfig(d_token=16, heads=2, set_layers=1, d_frame=24, dilations=(1, 2, 4))


def _toy_game(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    players = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22, 2)), axis=0), 0.01, 0.99)
    ball = players[:, 3].copy()
    ball[rng.random(n) < 0.1] = np.nan
    labels = np.full(n, H, dtype=np.int8)
    labels[800:1200] = D
    labels[1200:] = A
    labels[:50] = 3
    return deep_data.CleanGame(
        "toy",
        0,
        players.astype(np.float32),
        np.array([H] * 11 + [A] * 11, dtype=np.int8),
        ball.astype(np.float32),
        labels,
        ((50, n),),
    )


def test_inputs_shapes_and_mask():
    game = _toy_game()
    raw = deep_data.degrade_window(
        game, 0, 600, np.random.default_rng(1), deep_data.Randomization()
    )
    tokens, mask, globals_ = inputs(raw["slots"], raw["slot_team"], raw["ball"], (105.0, 68.0), 25)
    assert tokens.shape == (600, deep_data.N_SLOTS + 1, N_TOKEN)
    assert globals_.shape == (600, N_GLOBAL) and mask[:, -1].all()
    assert np.isfinite(tokens).all() and np.isfinite(globals_).all()
    assert (tokens[:, :-1][~mask[:, :-1]] == 0).all()


def test_swap_teams_is_an_involution():
    rng = np.random.default_rng(2)
    tokens = rng.normal(size=(10, 5, N_TOKEN)).astype(np.float32)
    globals_ = rng.normal(size=(10, N_GLOBAL)).astype(np.float32)
    t1, g1 = swap_teams(tokens, globals_)
    assert np.allclose(t1[..., TEAM_COLUMN], -tokens[..., TEAM_COLUMN])
    assert np.allclose(g1[:, COUNT_COLUMNS[0]], globals_[:, COUNT_COLUMNS[1]])
    t2, g2 = swap_teams(t1, g1)
    assert np.array_equal(t2, tokens) and np.array_equal(g2, globals_)


def test_degradation_targets():
    game = _toy_game()
    rng = np.random.default_rng(3)
    sample = deep_data.make_sample(game, 700, 1400, rng, deep_data.Randomization())
    state = sample["state"]
    assert set(np.unique(state)) <= {H, A, D, deep_data.IGNORE}
    possessor = sample["possessor"]
    alive = np.isin(state, (H, A))
    assert (possessor[~alive] == deep_data.IGNORE).all()  # no possessor while the ball is dead
    targeted = possessor[alive][possessor[alive] != deep_data.IGNORE]  # clean ball known
    assert len(targeted) and (targeted >= 0).all() and (targeted <= deep_data.N_SLOTS).all()
    # windows reaching outside the game are padded and ignored
    edge = deep_data.make_sample(game, -100, 400, rng, deep_data.Randomization())
    assert (edge["state"][:150] == deep_data.IGNORE).all()


def test_numpy_forward_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from mtah.solutions.ali.deep_model import SetTCN, export_numpy

    torch.manual_seed(0)
    model = SetTCN(SMALL).eval()
    path = export_numpy(model, tmp_path / "fold.npz")
    net = NumpyNet.load(path)
    rng = np.random.default_rng(4)
    t, k = 300, 9
    tokens = rng.normal(size=(t, k, N_TOKEN)).astype(np.float32)
    tokens[..., TEAM_COLUMN] = np.sign(rng.normal(size=(t, k)))
    mask = rng.random((t, k)) < 0.8
    mask[:, -1] = True
    tokens[~mask] = 0
    globals_ = rng.normal(size=(t, N_GLOBAL)).astype(np.float32)
    ours = net.state_logits(tokens, mask, globals_, (0, 1), chunk=128)
    with torch.no_grad():
        state, *_ = model(
            torch.from_numpy(tokens)[None],
            torch.from_numpy(mask)[None],
            torch.from_numpy(globals_)[None],
            torch.tensor([1]),
        )
    theirs = model.state_logits_half(tokens, mask, globals_, 0, "cpu", chunk=128)
    assert np.allclose(ours[1], state[0].numpy(), atol=1e-4)
    assert np.allclose(ours[0], theirs, atol=1e-4)
    # the ensemble is exactly symmetric in the two teams
    team, dead = ensemble_logits([net], tokens, mask, globals_)
    st, sg = swap_teams(tokens, globals_)
    team2, dead2 = ensemble_logits([net], st, mask, sg)
    assert np.allclose(team, -team2, atol=1e-5) and np.allclose(dead, dead2, atol=1e-5)


def test_boundary_targets_and_run_lengths():
    labels = np.array([H] * 20 + [A] * 10 + [D] * 5 + [A] * 5, dtype=np.int8)
    soft = deep_data.boundary_targets(labels)
    assert soft[20] == 1.0 and soft[19] > soft[10] and soft[33] < 1e-3  # H -> A only, not restarts
    assert soft[35] < 1e-3
    assert deep_data.run_lengths(labels).tolist() == [20] * 20 + [10] * 10 + [5] * 5 + [5] * 5


def test_spell_alpha_weighs_short_runs_more():
    game = _toy_game()
    rng = np.random.default_rng(5)
    rnd = deep_data.Randomization(swap_p=0.0)
    plain = deep_data.make_sample(game, 700, 1400, rng, rnd)
    assert (plain["state_weight"] == 1.0).all()
    weighted = deep_data.make_sample(game, 700, 1400, rng, rnd, spell_alpha=1.0)
    assert weighted["state_weight"].min() > 0 and weighted["state_weight"].max() <= 2.0


def test_boundary_head_numpy_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from mtah.solutions.ali.deep_model import SetTCN, export_numpy

    torch.manual_seed(2)
    model = SetTCN(replace(SMALL, boundary_head=True)).eval()
    net = NumpyNet.load(export_numpy(model, tmp_path / "fold.npz"))
    rng = np.random.default_rng(6)
    t, k = 200, 9
    tokens = rng.normal(size=(t, k, N_TOKEN)).astype(np.float32)
    tokens[..., TEAM_COLUMN] = np.sign(rng.normal(size=(t, k)))
    mask = rng.random((t, k)) < 0.8
    mask[:, -1] = True
    tokens[~mask] = 0
    globals_ = rng.normal(size=(t, N_GLOBAL)).astype(np.float32)
    ((_, ours),) = net.heads(tokens, mask, globals_, (1,), chunk=64)
    _, theirs = model.heads_half(tokens, mask, globals_, 1, "cpu", chunk=64)
    assert np.allclose(ours, theirs, atol=1e-4)


def test_published_weights_are_pinned_by_sha256(tmp_path, monkeypatch):
    from mtah.solutions.ali import deep_weights

    models, other = tmp_path / "models", tmp_path / "other"
    models.mkdir()
    other.mkdir()
    (models / "fold0.npz").write_bytes(b"weights")
    (other / "fold0.npz").write_bytes(b"anything")
    monkeypatch.setattr(deep_weights, "MANIFEST", tmp_path / "deep_weights.json")
    monkeypatch.setattr(deep_weights, "MODELS", models)
    assert deep_weights.verified(models) == [models / "fold0.npz"]  # no manifest yet
    deep_weights.write(folder=models)
    assert deep_weights.verified(models) == [models / "fold0.npz"]
    assert deep_weights.verified(other) == [other / "fold0.npz"]  # experiments: unchecked
    (models / "fold0.npz").write_bytes(b"retrained")
    with pytest.raises(ValueError, match="sha256"):
        deep_weights.verified(models)
    (models / "fold0.npz").unlink()
    with pytest.raises(FileNotFoundError):
        deep_weights.verified(models)


def test_ali_deep_solution_runs_end_to_end(mini_match, tmp_path):
    """Numpy inference of a (random) exported model inside the real solution, twice: same bytes."""
    torch = pytest.importorskip("torch")
    from mtah.solutions.ali.deep_model import SetTCN, export_numpy
    from mtah.solutions.ali.deep_solution import AliDeepSolution

    torch.manual_seed(1)
    export_numpy(SetTCN(SMALL).eval(), tmp_path / "fold0.npz")
    solution = AliDeepSolution(models=str(tmp_path), blend=0.5)
    first = solution.predict(mini_match)
    second = solution.predict(mini_match)
    graded = mini_match.spec.graded_mask()
    assert first.dtype == np.int8 and first.shape == (mini_match.spec.n_frames,)
    assert (first[~graded] == 3).all() and set(np.unique(first[graded])) <= {0, 1, 2}
    assert np.array_equal(first, second)


def _jitter_m(xy_norm):
    xy = xy_norm.astype(np.float64) * np.array([105.0, 68.0])
    second = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1)
    return float(np.nanmedian(second))


def test_smooth_windows_have_no_jitter_and_noisy_ones_do():
    game = _toy_game(n=3000)
    smooth = deep_data.Randomization(smooth_p=1.0, blackout_share=(0.0, 0.0), flip_p=0.0)
    noisy = deep_data.Randomization(
        smooth_p=0.0, position_noise_m=(0.2, 0.2), ball_noise_m=(0.3, 0.3), flip_p=0.0
    )
    a = deep_data.degrade_window(game, 100, 2900, np.random.default_rng(4), smooth)
    b = deep_data.degrade_window(game, 100, 2900, np.random.default_rng(4), noisy)
    clean = _jitter_m(game.players[100:2900])  # the toy walk itself moves ~0.2 m a frame
    assert _jitter_m(a["slots"]) < clean + 0.08 < _jitter_m(b["slots"])
    assert _jitter_m(a["ball"]) < _jitter_m(b["ball"])


def test_long_ball_gaps_are_drawn():
    game = _toy_game(n=6000)
    rnd = deep_data.Randomization(ball_gap_s=(20.0, 20.0), blackout_share=(0.0, 0.0))
    raw = deep_data.degrade_window(game, 100, 5900, np.random.default_rng(5), rnd)
    missing = np.isnan(raw["ball"][:, 0]).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert lengths.max() > 10 * 25  # a run of more than 10 s without the ball


def test_blackouts_remove_players_and_their_possessor_targets():
    game = _toy_game(n=3000)
    rng = np.random.default_rng(6)
    slots = np.full((3000, deep_data.N_SLOTS, 2), 0.5)
    before = slots.copy()
    deep_data.blackouts(slots, np.full(3000, 0.5), rng, 0.0, (2.0, 14.0))
    assert np.array_equal(slots, before)  # share 0: nothing happens
    deep_data.blackouts(slots, np.full(3000, 0.5), rng, 0.5, (2.0, 14.0))
    share = np.isnan(slots[:, :, 0]).mean()
    assert 0.05 < share < 0.6
    rnd = deep_data.Randomization(blackout_share=(0.5, 0.5), flip_p=0.0)
    raw = deep_data.degrade_window(game, 100, 2900, np.random.default_rng(7), rnd)
    named = raw["possessor"][(raw["possessor"] >= 0) & (raw["possessor"] < deep_data.N_SLOTS)]
    rows = np.flatnonzero((raw["possessor"] >= 0) & (raw["possessor"] < deep_data.N_SLOTS))
    assert len(named) and not np.isnan(raw["slots"][rows, named, 0]).any()


def test_real_like_benchmark_profile_is_smooth_with_long_ball_runs():
    from mtah.solutions.ali import bench

    real = bench.PROFILES["real-like"]
    assert real.position_noise_m < 0.01 and real.ball_mean_gap_s >= 10 and real.blackout_share > 0
    assert bench.PROFILES["simulated"] == bench.Degradation()  # the first benchmark is unchanged


def test_dead_ball_motion_moves_the_ball_only_inside_stoppages():
    game = _toy_game(n=4000)
    players = game.players.astype(np.float64)
    ball = players[:, 3].copy()
    dead = np.zeros(4000, dtype=bool)
    for a, b in ((500, 1100), (1500, 1530), (2000, 2900), (3990, 4000)):
        dead[a:b] = True
    scale = np.array([105.0, 68.0])
    for seed in range(25):
        moved = ball.copy()
        changed = deep_data.dead_ball_motion(moved, players, dead, np.random.default_rng(seed), 1.0)
        assert np.isfinite(moved).all() and not changed[~dead].any()
        assert np.array_equal(moved[~changed], ball[~changed])
        step = np.linalg.norm(np.diff(moved * scale, axis=0), axis=1) * 25
        inside = changed[1:] & changed[:-1]
        assert step[inside].max() < 30  # never faster than a struck ball
    untouched = ball.copy()
    assert not deep_data.dead_ball_motion(
        untouched, players, dead, np.random.default_rng(0), 0.0
    ).any()
    assert np.array_equal(untouched, ball)


def test_rewritten_dead_ball_frames_have_no_position_target():
    game = _toy_game(n=3000)
    rnd = deep_data.Randomization(
        dead_ball_motion=(1.0, 1.0), blackout_share=(0.0, 0.0), flip_p=0.0
    )
    sample = deep_data.make_sample(game, 600, 1500, np.random.default_rng(8), rnd)
    dead = sample["state"] == D
    assert dead.any() and (sample["ball_weight"][dead] == 0).mean() > 0.9
    assert (sample["ball_weight"][~dead] > 0).any()
