import numpy as np
from ai import NeuralNetwork, encode_state, PrioritizedReplayBuffer


def test_log_encoding_is_position_neutral():
    a = np.zeros((4, 4), dtype=int)
    b = np.zeros((4, 4), dtype=int)
    a[0, 0], b[3, 3] = 128, 128
    ea, eb = encode_state(a), encode_state(b)
    assert np.sum(ea) == np.sum(eb)
    assert np.count_nonzero(ea) == np.count_nonzero(eb) == 1
    assert ea[0] == eb[15]


def test_tile_planes_encoding_is_position_neutral_and_one_hot():
    a = np.zeros((4, 4), dtype=int)
    b = np.zeros((4, 4), dtype=int)
    a[0, 0], b[3, 3] = 128, 128
    ea, eb = encode_state(a, "tile_planes"), encode_state(b, "tile_planes")
    assert ea.shape == (272,)
    assert np.array_equal(ea.reshape(16, 17).sum(axis=1), np.ones(16))
    assert np.array_equal(np.sort(ea), np.sort(eb))


def test_forward_returns_four_finite_q_values_for_both_encodings():
    state = np.array([[2, 0, 4, 0]] * 4)
    for encoding in ("log", "tile_planes"):
        ai = NeuralNetwork(seed=4, state_encoding=encoding)
        q = ai.forward(state)
        assert q.shape == (4,)
        assert np.all(np.isfinite(q))


def test_prioritized_replay_samples_and_updates_priorities():
    buf = PrioritizedReplayBuffer(32, 16)
    buf.rng = np.random.default_rng(1)
    for i in range(20):
        s = np.full(16, i, dtype=np.float32)
        buf.add(s, i % 4, float(i), s + 1, False, [True, True, True, True], 3)
    idx, weights = buf.sample(8)
    assert len(set(idx.tolist())) == 8
    assert weights.shape == (8,)
    buf.update_priorities(idx, np.arange(1, 9))
    assert np.all(buf.priorities[idx] > 0)


def test_n_step_returns_and_checkpoint_resume(tmp_path):
    ai = NeuralNetwork(seed=9, n_step=3, replay_capacity=512)
    s0 = np.zeros((4, 4), dtype=int)
    s1 = s0.copy(); s1[0, 0] = 2
    s2 = s1.copy(); s2[0, 1] = 2
    s3 = s2.copy(); s3[0, 2] = 2
    ai.store_experience(s0, 0, 1.0, s1, False, [0, 1, 2, 3])
    ai.store_experience(s1, 0, 2.0, s2, False, [0, 1, 2, 3])
    ai.store_experience(s2, 0, 3.0, s3, False, [0, 1, 2, 3])
    assert len(ai.memory_buffer) == 1
    expected = 1 + ai.gamma * 2 + ai.gamma**2 * 3
    assert np.isclose(ai.memory_buffer.rewards[0], expected)
    for _ in range(260):
        ai.store_experience(s0, 1, 0.25, s1, False, [0, 1, 2, 3])
    ai.learn(batch_size=32)
    path = tmp_path / "checkpoint.npz"
    ai.save(path, include_training_state=True)
    restored = NeuralNetwork(seed=1)
    restored.load(path, load_training_state=True)
    assert restored.layer_sizes == ai.layer_sizes
    assert restored.train_steps == ai.train_steps
    assert len(restored.memory_buffer) == len(ai.memory_buffer)
    assert restored.memory_buffer.sample_count == ai.memory_buffer.sample_count
    for a, b in zip(ai.weights, restored.weights):
        assert np.array_equal(a, b)


def test_legal_action_mask_has_no_moves_on_full_stuck_board():
    from ai import valid_action_mask
    board = np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]])
    assert not valid_action_mask(board).any()


def test_symmetry_augmentation_keeps_legal_action_mask_consistent():
    from ai import valid_action_mask

    board = np.array([[2, 0, 4, 0], [0, 2, 0, 0],
                      [4, 0, 0, 0], [0, 0, 0, 0]])
    ai = NeuralNetwork(seed=123, state_encoding="tile_planes")
    encoded = ai.encode(board)
    states = encoded.reshape(1, -1).copy()
    next_states = encoded.reshape(1, -1).copy()
    actions = np.array([0], dtype=np.int64)
    masks = valid_action_mask(board).reshape(1, 4).copy()

    ai._augment_transitions(states, next_states, actions, masks)

    transformed_categories = states[0].reshape(16, 17).argmax(axis=1)
    transformed_board = np.where(
        transformed_categories.reshape(4, 4) == 0,
        0,
        2 ** transformed_categories.reshape(4, 4),
    )
    assert masks[0].tolist() == valid_action_mask(transformed_board).tolist()
    assert actions[0] in np.flatnonzero(masks[0])
    assert np.all(states[0].reshape(16, 17).sum(axis=1) == 1)


def test_unknown_state_encoding_fails_with_clear_error():
    import pytest
    with pytest.raises(ValueError, match="Choose 'log' or 'tile_planes'"):
        NeuralNetwork(state_encoding="tile_planez")


def test_tile_planes_checkpoint_preserves_encoding_and_training_state(tmp_path):
    ai = NeuralNetwork(seed=17, state_encoding="tile_planes", replay_capacity=32)
    board = np.zeros((4, 4), dtype=np.int64)
    board[0, 0] = 2
    ai.store_experience(board, 0, 0.5, board, False, [0, 1, 2, 3])
    path = tmp_path / "planes.npz"
    ai.save(path, include_training_state=True)
    restored = NeuralNetwork(seed=99, state_encoding="tile_planes", replay_capacity=32)
    restored.load(path, load_training_state=True)
    assert restored.state_encoding == "tile_planes"
    assert restored.state_dim == 272
    assert len(restored.memory_buffer) == len(ai.memory_buffer)
    assert np.array_equal(restored.weights[0], ai.weights[0])
