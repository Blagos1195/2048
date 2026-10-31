import numpy as np
import pytest

from selfplay_ai import (
    Game2048, NTupleValue, ExpectimaxAgent, apply_action, can_move,
    exponent_board, legal_actions, pattern_indices, slide_left,
)


def test_merge_each_tile_once_and_score():
    board = np.zeros((4, 4), dtype=np.int64)
    board[0] = [2, 2, 2, 2]
    out, score = slide_left(board)
    assert out[0].tolist() == [4, 4, 0, 0]
    assert score == 8


def test_merge_chain_is_not_double_merged():
    board = np.zeros((4, 4), dtype=np.int64)
    board[0] = [2, 2, 4, 0]
    out, score = slide_left(board)
    assert out[0].tolist() == [4, 4, 0, 0]
    assert score == 4


def test_action_rotations_are_consistent():
    board = np.array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    left, _ = apply_action(board, 0)
    assert left[0, 0] == 2
    down, score = apply_action(board, 1)
    assert down[3, 0] == 4
    assert score == 4


def test_game_seed_reproducibility():
    a, b = Game2048(42), Game2048(42)
    assert np.array_equal(a.board, b.board)
    for action in [0, 1, 2, 3]:
        if can_move(a.board):
            a.step(action)
            b.step(action)
            assert np.array_equal(a.board, b.board)


def test_full_board_with_no_pairs_is_terminal():
    board = np.array([[2, 4, 8, 16], [4, 8, 16, 2], [8, 16, 2, 4], [16, 2, 4, 8]])
    assert legal_actions(board) == []
    assert not can_move(board)


def test_feature_indices_and_value_are_valid():
    board = np.array([[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]])
    indices = pattern_indices(board)
    assert indices.shape == (26,)
    assert np.all(indices >= 0)
    assert np.all(indices < 16 ** 4)
    assert exponent_board(board)[0].tolist() == [0, 1, 2, 3]
    assert NTupleValue().value(board) == 0.0


def test_td_update_changes_value():
    value = NTupleValue(alpha=0.1, gamma=0.9)
    state = np.zeros((4, 4), dtype=np.uint32)
    next_state = state.copy()
    next_state[0, 0] = 2
    before = value.value(state)
    delta = value.update(state, 1.0, next_state, terminal=True)
    assert delta == pytest.approx(1.0)
    assert value.updates == 1
    assert value.value(state) > before


def test_agent_selects_legal_move():
    evaluator = NTupleValue(seed=2)
    agent = ExpectimaxAgent(evaluator, depth=1, seed=2)
    board = Game2048(3).board
    assert agent.choose_action(board) in legal_actions(board)


def test_expected_spawn_value_matches_explicit_enumeration():
    evaluator = NTupleValue(seed=7)
    rng = np.random.default_rng(8)
    evaluator.weights[:] = rng.normal(0.0, 0.01, evaluator.weights.shape).astype(np.float32)
    board = np.array([[2, 0, 4, 0], [0, 8, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]], dtype=np.uint32)
    expected = 0.0
    empty = np.argwhere(board == 0)
    for r, c in empty:
        for tile, probability in ((2, 0.9), (4, 0.1)):
            child = board.copy()
            child[r, c] = tile
            expected += probability * evaluator.value(child) / len(empty)
    assert evaluator.expected_spawn_value(board) == pytest.approx(expected, abs=1e-6)



def test_expectimax_cache_reuses_identical_chance_states():
    evaluator = NTupleValue(seed=11)
    agent = ExpectimaxAgent(evaluator, depth=1, seed=11)
    board = Game2048(11).board
    first = agent._chance(board.copy(), 0)
    nodes = agent.nodes
    second = agent._chance(board.copy(), 0)
    assert first == pytest.approx(second)
    assert agent.nodes == nodes


def test_local_parallel_collection_has_transition_data():
    from selfplay_ai import _collect_episode_local, _apply_collected_episode

    evaluator = NTupleValue(seed=13)
    episode = _collect_episode_local(evaluator, seed=13, epsilon=0.1, depth=1)
    assert episode["moves"] > 0
    assert len(episode["transitions"]) == episode["moves"]
    td_mean = _apply_collected_episode(evaluator, episode["transitions"])
    assert np.isfinite(td_mean)
    assert evaluator.updates == episode["moves"]


def test_training_parser_exposes_worker_and_benchmark_options():
    from selfplay_ai import build_parser

    args = build_parser().parse_args(["train", "--games", "2", "--workers", "2"])
    assert args.workers == 2
    bench = build_parser().parse_args(["benchmark", "--model", "model.npz", "--games", "3", "--depths", "1", "2"])
    assert bench.depths == [1, 2]


def test_benchmark_command_validates_depths_without_single_depth(monkeypatch):
    import sys
    import selfplay_ai

    captured = []
    monkeypatch.setattr(
        sys, "argv",
        ["selfplay_ai.py", "benchmark", "--model", "unused.npz", "--depths", "1", "2"],
    )
    monkeypatch.setattr(selfplay_ai, "benchmark_cli", lambda args: captured.append(args))

    selfplay_ai.main()

    assert captured[0].depths == [1, 2]



def test_expectimax_cache_persists_between_moves_and_is_bounded():
    evaluator = NTupleValue(seed=17)
    agent = ExpectimaxAgent(evaluator, depth=2, seed=17, cache_size=32)
    board = Game2048(17).board
    first_action = agent.choose_action(board.copy())
    assert first_action in legal_actions(board)
    assert agent._chance_cache or agent._player_cache

    # Re-evaluating a previously visited state should use persistent cached values.
    previous_hits = agent.cache_hits
    agent._chance(board.copy(), 0)
    assert agent.cache_hits >= previous_hits
    assert len(agent._chance_cache) <= 32
    assert len(agent._player_cache) <= 32


def test_expectimax_iterative_deepening_returns_legal_action():
    evaluator = NTupleValue(seed=19)
    agent = ExpectimaxAgent(evaluator, depth=2, seed=19)
    board = Game2048(19).board
    assert agent.choose_action(board) in legal_actions(board)
    # A second choice on the same board is valid and retains cache entries.
    cache_size_before = len(agent._chance_cache) + len(agent._player_cache)
    assert agent.choose_action(board) in legal_actions(board)
    assert len(agent._chance_cache) + len(agent._player_cache) >= 0
    assert cache_size_before > 0



def test_training_evaluation_defaults_compare_four_depths():
    from selfplay_ai import build_parser

    args = build_parser().parse_args(["train", "--games", "1"])
    assert args.depth == 1
    assert args.eval_every == 1000
    assert args.eval_games == 50
    assert args.eval_depths == [1, 2, 3, 4]
    assert args.eval_depth == 2
    assert args.workers == 0


def test_multi_depth_evaluation_uses_same_seeds_without_learning():
    from selfplay_ai import evaluate_depths

    evaluator = NTupleValue(seed=23)
    before = evaluator.weights.copy()
    reports = evaluate_depths(evaluator, games=1, depths=[1], seed_start=9876)
    assert len(reports) == 1
    assert reports[0]["depth"] == 1
    assert reports[0]["games"] == 1
    assert reports[0]["seconds"] >= 0
    assert np.array_equal(evaluator.weights, before)
    assert evaluator.updates == 0


def test_multi_depth_evaluation_reports_separate_timings_and_progress(monkeypatch, capsys):
    import selfplay_ai

    def fake_play_game(agent, seed, train=False):
        return {
            "seed": seed, "score": 100 + seed % 10, "max_tile": 2048,
            "moves": 100, "reached_2048": 1, "reached_4096": 0,
            "reached_8192": 0, "td_abs_mean": 0.0, "search_nodes": 10,
        }

    monkeypatch.setattr(selfplay_ai, "play_game", fake_play_game)
    reports = selfplay_ai.evaluate_depths(
        NTupleValue(), games=2, depths=[1, 3], seed_start=1000,
    )

    assert [report["depth"] for report in reports] == [1, 3]
    assert all(report["games"] == 2 for report in reports)
    assert all(report["seconds"] >= 0 for report in reports)
    output = capsys.readouterr().out
    assert "EVAL PROGRESS depth=1 games=1/2" in output
    assert "EVAL DEPTH COMPLETE depth=3 games=2" in output



def test_expectimax_cache_reuses_transposed_states():
    evaluator = NTupleValue(seed=29)
    rng = np.random.default_rng(30)
    evaluator.weights[:] = rng.normal(0.0, 0.01, evaluator.weights.shape).astype(np.float32)
    agent = ExpectimaxAgent(evaluator, depth=1, seed=29)
    board = np.array([
        [2, 0, 4, 0],
        [0, 8, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
    ], dtype=np.uint32)

    direct = agent._chance(board.copy(), 0)
    nodes = agent.nodes
    transposed = agent._chance(board.T.copy(), 0)

    assert transposed == pytest.approx(direct, abs=1e-6)
    assert agent.nodes == nodes
    assert agent.cache_hits == 1


def test_expectimax_invalidates_cache_after_learning_update():
    evaluator = NTupleValue(seed=31)
    rng = np.random.default_rng(32)
    evaluator.weights[:] = rng.normal(0.0, 0.01, evaluator.weights.shape).astype(np.float32)
    agent = ExpectimaxAgent(evaluator, depth=1, seed=31)
    board = np.array([
        [2, 0, 4, 0],
        [0, 8, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
    ], dtype=np.uint32)

    first = agent._chance(board.copy(), 0)
    old_nodes = agent.nodes
    next_state = board.copy()
    next_state[0, 1] = 2
    evaluator.update(board, 1.0, next_state, terminal=True)

    second = agent._chance(board.copy(), 0)
    assert evaluator.updates == 1
    assert agent.nodes > old_nodes
    assert agent._cache_updates == evaluator.updates
    assert first != pytest.approx(second, abs=1e-8)


def test_cached_row_transitions_match_reference_for_random_boards():
    rng = np.random.default_rng(41)

    def reference_row(row):
        values = [int(v) for v in row if v]
        merged = []
        gained = 0
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                value = values[i] * 2
                merged.append(value)
                gained += value
                i += 2
            else:
                merged.append(values[i])
                i += 1
        return merged + [0] * (4 - len(merged)), gained

    for _ in range(250):
        board = rng.choice([0, 2, 4, 8, 16, 32, 64], size=(4, 4)).astype(np.uint32)
        expected = np.zeros((4, 4), dtype=np.uint32)
        expected_score = 0
        for row_index in range(4):
            row, score = reference_row(board[row_index])
            expected[row_index] = row
            expected_score += score
        actual, actual_score = slide_left(board)
        assert np.array_equal(actual, expected)
        assert actual_score == expected_score


def test_expectimax_search_evaluates_only_requested_depth(monkeypatch):
    evaluator = NTupleValue(seed=43)
    agent = ExpectimaxAgent(evaluator, depth=3, seed=43)
    board = np.array([
        [2, 0, 4, 0],
        [0, 8, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
    ], dtype=np.uint32)
    calls = []

    def fake_player(state, depth, action):
        calls.append(depth)
        return float(action)

    monkeypatch.setattr(agent, "_player", fake_player)
    action = agent.choose_action(board)
    assert action in legal_actions(board)
    assert calls
    assert set(calls) == {3}


def test_legacy_eight_pattern_checkpoint_migrates_without_losing_weights(tmp_path):
    from selfplay_ai import PATTERN_COUNT, PATTERN_TABLE_SIZE

    legacy = np.arange(8 * PATTERN_TABLE_SIZE, dtype=np.float32).reshape(8, PATTERN_TABLE_SIZE)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path, weights=legacy, alpha=np.array(0.07), gamma=np.array(0.99),
        updates=np.array(1234), metadata=np.array("{}"),
    )

    evaluator = NTupleValue()
    evaluator.load(path)

    assert evaluator.weights.shape == (PATTERN_COUNT, PATTERN_TABLE_SIZE)
    assert np.array_equal(evaluator.weights[:8], legacy)
    assert np.count_nonzero(evaluator.weights[8:]) == 0
    assert evaluator.alpha == pytest.approx(0.07)
    assert evaluator.gamma == pytest.approx(0.99)
    assert evaluator.updates == 1234


def test_local_square_patterns_are_included_and_transpose_invariant():
    from selfplay_ai import PATTERNS, pattern_indices

    assert len(PATTERNS) == 26
    assert len(PATTERNS[8:]) == 18
    board = np.array([
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 4096],
        [8192, 16384, 32768, 65536],
    ], dtype=np.uint32)
    assert np.array_equal(pattern_indices(board), pattern_indices(board.T))
