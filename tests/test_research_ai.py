import numpy as np
import pytest

import selfplay_ai as ai
import research_ai as research


def test_stage_boundaries():
    assert research.MultiStageNTupleValue.stage_for(np.array([[2]], dtype=np.uint32)) == 0
    assert research.MultiStageNTupleValue.stage_for(np.array([[2048]], dtype=np.uint32)) == 0
    assert research.MultiStageNTupleValue.stage_for(np.array([[8192]], dtype=np.uint32)) == 1
    assert research.MultiStageNTupleValue.stage_for(np.array([[16384]], dtype=np.uint32)) == 2
    assert research.MultiStageNTupleValue.stage_for(np.array([[32768]], dtype=np.uint32)) == 3
    assert research.MultiStageNTupleValue.stage_for(np.array([[131072]], dtype=np.uint32)) == 5


def test_stage_specific_values_are_selected():
    model = research.MultiStageNTupleValue()
    model.weights[1].fill(1.0)
    model.weights[2].fill(2.0)
    board = np.zeros((4, 4), dtype=np.uint32)
    board[0, 0] = 8192
    assert model.value(board) == pytest.approx(ai.PATTERN_COUNT)
    board[0, 0] = 16384
    assert model.value(board) == pytest.approx(ai.PATTERN_COUNT * 2)


def test_temporal_coherence_updates_the_selected_stage():
    model = research.MultiStageNTupleValue(alpha=0.1, learning_mode="tc")
    state = np.zeros((4, 4), dtype=np.uint32)
    delta = model.update_target(state, 100.0)
    assert delta == pytest.approx(100.0)
    assert np.all(model.coherence_sum[0, :, 0] > 0)
    assert np.all(model.coherence_abs[0, :, 0] > 0)
    assert np.all(model.weights[0, :, 0] > 0)
    assert not np.any(model.weights[1:])


def test_temporal_coherence_reduces_rate_after_conflicting_updates():
    model = research.MultiStageNTupleValue(alpha=0.1, learning_mode="tc")
    state = np.zeros((4, 4), dtype=np.uint32)
    model.update_target(state, 100.0)
    first_weight = float(model.weights[0, 0, 0])
    model.update_target(state, -50.0)
    assert model.coherence_abs[0, 0, 0] > abs(model.coherence_sum[0, 0, 0])
    assert model.weights[0, 0, 0] != first_weight


def test_expected_spawn_value_matches_explicit_enumeration():
    rng = np.random.default_rng(17)
    model = research.MultiStageNTupleValue()
    model.weights[0] = rng.normal(0.0, 0.01, model.weights[0].shape).astype(np.float32)
    board = np.array(
        [[2, 0, 4, 0], [0, 8, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
        dtype=np.uint32,
    )
    expected = 0.0
    empty = np.argwhere(board == 0)
    for row, column in empty:
        for tile, probability in ((2, 0.9), (4, 0.1)):
            child = board.copy()
            child[row, column] = tile
            expected += probability * model.value(child) / len(empty)
    assert model.expected_spawn_value(board) == pytest.approx(expected, abs=1e-6)


def test_optimistic_initialisation_preserves_base_preferences(tmp_path):
    base = ai.NTupleValue()
    base.weights[:] = np.random.default_rng(4).normal(
        0.0, 0.01, base.weights.shape
    ).astype(np.float32)
    path = tmp_path / "base.npz"
    base.save(path)
    model = research.MultiStageNTupleValue.from_checkpoint(path, optimism=260.0)
    board = np.array(
        [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint32,
    )
    assert model.value(board) - base.value(board) == pytest.approx(260.0, abs=1e-4)
    assert model.value(board) - model.value(board[::-1, ::-1]) == pytest.approx(
        base.value(board) - base.value(board[::-1, ::-1]), abs=1e-4
    )


def test_research_checkpoint_round_trip(tmp_path):
    model = research.MultiStageNTupleValue()
    model.weights[3] = np.random.default_rng(5).normal(
        0.0, 0.01, model.weights[3].shape
    ).astype(np.float32)
    model.coherence_sum[3, 0, 0] = 2.5
    model.coherence_abs[3, 0, 0] = 3.0
    model.updates = 123
    path = tmp_path / "research.npz"
    model.save(path, {"episode": 5})
    loaded = research.MultiStageNTupleValue()
    loaded.load(path)
    assert np.array_equal(model.weights, loaded.weights)
    assert np.array_equal(model.coherence_sum, loaded.coherence_sum)
    assert np.array_equal(model.coherence_abs, loaded.coherence_abs)
    assert loaded.updates == 123


def test_restart_episode_uses_supplied_board():
    model = research.MultiStageNTupleValue()
    ai._WORKER_EVALUATOR = model
    ai._WORKER_DEPTH = 1
    board = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 0],
        ],
        dtype=np.uint32,
    )
    result = research.collect_episode(
        (123, 0.0, 0.0, 1024, 1, board.tobytes())
    )
    assert result["start_stage"] == 3
    assert result["moves"] > 0


def test_optimistic_td_uses_fixed_learning_rate_before_tc():
    model = research.MultiStageNTupleValue(alpha=0.1, learning_mode="td")
    state = np.zeros((4, 4), dtype=np.uint32)
    model.update_target(state, 100.0)
    assert np.allclose(model.weights[0, :, 0], 0.1 * 100.0 / ai.PATTERN_COUNT)
    assert not np.any(model.coherence_sum)
    model.learning_mode = "tc"
    model.coherence_sum.fill(0.0)
    model.coherence_abs.fill(0.0)
    model.update_target(state, 50.0)
    assert np.all(model.coherence_sum[0, :, 0] > 0)


def test_multistage_thresholds_focus_on_late_game_milestones():
    assert research.STAGE_THRESHOLDS.tolist() == [0, 8192, 16384, 32768, 65536, 131072]


def test_tile_downgrade_compresses_high_tile_gaps_without_merging_ranks():
    board = np.zeros((4, 4), dtype=np.uint32)
    board[0, :3] = [2, 4, 16]
    downgraded = research.tile_downgrade(board)
    assert downgraded[0, :3].tolist() == [2, 4, 8]
    assert board[0, :3].tolist() == [2, 4, 16]


def test_tile_downgraded_evaluator_keeps_original_learning_stage():
    model = research.MultiStageNTupleValue()
    model.weights[3].fill(3.0)
    model.weights[4].fill(4.0)
    board = np.zeros((4, 4), dtype=np.uint32)
    board[0, :4] = [2, 4, 8, 65536]
    wrapped = research.TileDowngradedEvaluator(model)
    assert model.stage_for(board) == 4
    assert wrapped.value(board) == pytest.approx(ai.PATTERN_COUNT * 4.0)


def test_tile_downgrade_uses_the_highest_missing_tile_levels():
    board = np.zeros((4, 4), dtype=np.uint32)
    board[0, :4] = [2, 4, 16, 256]
    assert research.tile_downgrade(board, shifts=1)[0, :4].tolist() == [2, 4, 16, 128]
    assert research.tile_downgrade(board, shifts=2)[0, :4].tolist() == [2, 4, 16, 64]


def test_tile_downgrade_age_advances_with_real_moves():
    model = research.MultiStageNTupleValue()
    wrapped = research.TileDowngradedEvaluator(model)
    agent = ai.ExpectimaxAgent(wrapped, depth=1, seed=44)
    board = np.zeros((4, 4), dtype=np.uint32)
    board[0, :2] = [32768, 16384]
    agent.choose_action(board)
    assert wrapped.steps == 1
    agent.choose_action(board)
    assert wrapped.steps == 2
