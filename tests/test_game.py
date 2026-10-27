import numpy as np
from game import Game2048


def test_seeded_game_is_reproducible():
    a = Game2048(seed=123)
    b = Game2048(seed=123)
    assert np.array_equal(a.board, b.board)


def test_merge_pairs_once_each():
    row, score = Game2048.merge(np.array([2, 2, 2, 2]))
    assert row.tolist() == [4, 0, 4, 0]
    assert score == 8


def test_merge_does_not_chain_new_tile():
    row, score = Game2048.merge(np.array([2, 2, 4, 0]))
    assert Game2048.compress(row).tolist() == [4, 4, 0, 0]
    assert score == 4


def test_valid_actions_only_include_board_changes():
    game = Game2048(seed=5)
    for action in game.valid_actions():
        after, _ = game.board_after_move(action)
        assert not np.array_equal(after, game.board)


def test_full_locked_board_is_game_over():
    game = Game2048(seed=1)
    game.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2],
                           [2, 4, 2, 4], [4, 2, 4, 2]])
    assert not game.can_move()
    assert game.valid_actions() == []


def test_direction_indices_match_action_names():
    game = Game2048(seed=10)
    game.board = np.zeros((4, 4), dtype=int)
    game.board[0, 1] = 2
    down, _ = game.board_after_move(1)
    up, _ = game.board_after_move(3)
    assert down[3, 1] == 2
    assert up[0, 1] == 2
    right, _ = game.board_after_move(2)
    left, _ = game.board_after_move(0)
    assert right[0, 3] == 2
    assert left[0, 0] == 2


def test_agent_legal_mask_matches_engine_on_random_boards():
    from ai import valid_action_mask
    rng = np.random.default_rng(45)
    game = Game2048(seed=2)
    for _ in range(500):
        exponents = rng.integers(0, 6, size=(4, 4))
        board = np.where(exponents == 0, 0, 2 ** exponents)
        game.board = board.astype(np.int64)
        expected = np.zeros(4, dtype=bool)
        expected[game.valid_actions()] = True
        assert np.array_equal(valid_action_mask(board), expected), board
