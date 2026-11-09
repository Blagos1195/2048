from __future__ import annotations

import os
import random
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

import selfplay_ai as ai


STAGE_THRESHOLDS = np.asarray(
    [0, 8192, 16384, 32768, 65536, 131072],
    dtype=np.uint32,
)
MILESTONES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


class MultiStageNTupleValue:
    def __init__(self, weights=None, alpha=0.08, gamma=0.995, optimism=320000.0, learning_mode="td"): 
        shape = (len(STAGE_THRESHOLDS), ai.PATTERN_COUNT, ai.PATTERN_TABLE_SIZE)
        self.weights = (
            np.zeros(shape, dtype=np.float32)
            if weights is None else np.asarray(weights, dtype=np.float32)
        )
        if self.weights.shape != shape:
            raise ValueError(f"Incompatible multistage weights: {self.weights.shape}")
        self.coherence_sum = np.zeros(shape, dtype=np.float32)
        self.coherence_abs = np.zeros(shape, dtype=np.float32)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.optimism = float(optimism)
        self.learning_mode = str(learning_mode)
        if self.learning_mode not in ("td", "tc"):
            raise ValueError("learning_mode must be 'td' or 'tc'")
        self.updates = 0

    @classmethod
    def from_checkpoint(cls, path, optimism=320000.0):
        base = ai.NTupleValue()
        base.load(path)
        weights = np.broadcast_to(
            base.weights,
            (len(STAGE_THRESHOLDS),) + base.weights.shape,
        ).copy()
        weights += np.float32(optimism / ai.PATTERN_COUNT)
        model = cls(weights, base.alpha, base.gamma, optimism)
        model.updates = base.updates
        return model

    @staticmethod
    def stage_for(board):
        maximum = int(np.max(board))
        return int(np.searchsorted(STAGE_THRESHOLDS, maximum, side="right") - 1)

    def value(self, board):
        stage = self.stage_for(board)
        return float(ai._value_kernel(
            np.asarray(board).reshape(-1),
            ai._TILE_EXPONENT_LOOKUP,
            self.weights[stage].reshape(-1),
            ai._PATTERN_CELLS,
            ai._PATTERN_PLACE_VALUES,
            ai._PATTERN_WEIGHT_OFFSETS,
        ))

    def expected_spawn_value(self, board):
        flat = np.asarray(board).reshape(-1)
        empty_mask = 0
        for cell, value in enumerate(flat):
            if value == 0:
                empty_mask |= 1 << cell
        if empty_mask == 0:
            return self.value(board)
        stage = self.stage_for(board)
        patterns, positions = ai._empty_pattern_memberships(empty_mask)
        return float(ai._expected_spawn_value_kernel(
            flat,
            ai._TILE_EXPONENT_LOOKUP,
            self.weights[stage].reshape(-1),
            ai._PATTERN_CELLS,
            ai._PATTERN_PLACE_VALUES,
            ai._PATTERN_WEIGHT_OFFSETS,
            patterns,
            positions,
            ai._PATTERN_INDEX_INCREMENTS,
            empty_mask.bit_count(),
        ))

    def update_target(self, state, target):
        stage = self.stage_for(state)
        indices = ai.pattern_indices(state)
        pattern_ids = np.arange(ai.PATTERN_COUNT)
        contribution = np.float32((float(target) - float(
            self.weights[stage, pattern_ids, indices].sum(dtype=np.float64)
        )) / ai.PATTERN_COUNT)
        if self.learning_mode == "tc":
            signed = self.coherence_sum[stage, pattern_ids, indices] + contribution
            absolute = self.coherence_abs[stage, pattern_ids, indices] + abs(contribution)
            self.coherence_sum[stage, pattern_ids, indices] = signed
            self.coherence_abs[stage, pattern_ids, indices] = absolute
            rates = self.alpha * np.abs(signed) / np.maximum(absolute, 1e-12)
            self.weights[stage, pattern_ids, indices] += rates * contribution
        else:
            self.weights[stage, pattern_ids, indices] += self.alpha * contribution
        self.updates += 1
        return float(contribution) * ai.PATTERN_COUNT

    def update(self, state, reward, next_state, terminal):
        target = float(reward)
        if not terminal:
            target += self.gamma * self.value(next_state)
        return self.update_target(state, target)

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                weights=self.weights,
                coherence_sum=self.coherence_sum,
                coherence_abs=self.coherence_abs,
                thresholds=STAGE_THRESHOLDS,
                alpha=self.alpha,
                gamma=self.gamma,
                optimism=self.optimism,
                learning_mode=np.array(self.learning_mode),
                updates=self.updates,
                metadata=np.array(str(metadata or {})),
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            weights = data["weights"]
            if weights.shape != self.weights.shape:
                raise ValueError(f"Incompatible checkpoint shape: {weights.shape}")
            if not np.array_equal(data["thresholds"], STAGE_THRESHOLDS):
                raise ValueError("Checkpoint stage thresholds do not match this build")
            self.weights[:] = weights
            self.coherence_sum[:] = data["coherence_sum"]
            self.coherence_abs[:] = data["coherence_abs"]
            self.alpha = float(data["alpha"])
            self.gamma = float(data["gamma"])
            self.optimism = float(data["optimism"])
            self.learning_mode = str(data["learning_mode"].item()) if "learning_mode" in data else "tc"
            self.updates = int(data["updates"])


def init_worker(shm_name, shape, alpha, gamma, depth):
    memory = shared_memory.SharedMemory(name=shm_name)
    weights = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
    weights.flags.writeable = False
    evaluator = MultiStageNTupleValue.__new__(MultiStageNTupleValue)
    evaluator.weights = weights
    evaluator.coherence_sum = None
    evaluator.coherence_abs = None
    evaluator.alpha = float(alpha)
    evaluator.gamma = float(gamma)
    evaluator.optimism = 0.0
    evaluator.learning_mode = "td"
    evaluator.updates = 0
    ai._WORKER_SHM = memory
    ai._WORKER_WEIGHTS = weights
    ai._WORKER_EVALUATOR = evaluator
    ai._WORKER_DEPTH = int(depth)


def collect_episode(task):
    seed, epsilon, late_epsilon, late_threshold, depth, start_state = task
    game = ai.Game2048(seed)
    if start_state is not None:
        game.board = np.frombuffer(start_state, dtype=np.uint32).reshape(4, 4).copy()
        game.score = 0
        game.moves = 0
    agent = ai.ExpectimaxAgent(
        ai._WORKER_EVALUATOR,
        depth=depth,
        seed=seed,
        epsilon=epsilon,
        late_epsilon=late_epsilon,
        late_tile_threshold=late_threshold,
    )
    transitions = []
    restart_states = {}
    while ai.can_move(game.board) and game.moves < 10000:
        state = game.board.copy()
        if game.moves % 24 == 0:
            stage = MultiStageNTupleValue.stage_for(state)
            if stage > 0:
                restart_states.setdefault(stage, []).append(state.tobytes())
        action = agent.choose_action(state)
        moved, gained, terminal = game.step(action)
        if not moved:
            if ai.can_move(game.board):
                raise RuntimeError(f"Agent chose illegal action {ai.ACTION_NAMES[action]}")
            terminal = True
        reward = gained / 128.0 - (2.0 if terminal else 0.0)
        transitions.append((state.tobytes(), reward, game.board.tobytes(), terminal))
    return {
        "seed": seed,
        "score": game.score,
        "max_tile": game.max_tile,
        "moves": game.moves,
        "transitions": transitions,
        "restart_states": restart_states,
        "start_stage": MultiStageNTupleValue.stage_for(
            np.frombuffer(start_state, dtype=np.uint32).reshape(4, 4)
        ) if start_state is not None else 0,
    }


def evaluate_depths(evaluator, games, depths, seed_start, pool=None):
    reports = []
    for depth in depths:
        started = time.perf_counter()
        tasks = [(seed_start + i, depth) for i in range(games)]
        if pool is None:
            rows = [
                ai.play_game(ai.ExpectimaxAgent(evaluator, depth=depth, seed=seed), seed)
                for seed, _ in tasks
            ]
        else:
            rows = list(pool.map(ai._evaluate_game_worker, tasks, chunksize=1))
        report = ai._summarise_evaluation(rows, depth, time.perf_counter() - started)
        tiles = np.asarray([row["max_tile"] for row in rows], dtype=np.uint32)
        for milestone in MILESTONES:
            report[f"rate_{milestone}"] = float(np.mean(tiles >= milestone))
        reports.append(report)
        print(
            f"RESEARCH EVAL depth={depth} games={games} "
            + " ".join(f"{key}={value}" for key, value in report.items()),
            flush=True,
        )
    return reports


def composite_quality(reports):
    weights = {
        2048: 500,
        4096: 1000,
        8192: 2000,
        16384: 4000,
        32768: 8000,
        65536: 16000,
        131072: 32000,
    }
    return float(np.mean([
        report["avg_score"] + sum(
            weight * report[f"rate_{tile}"]
            for tile, weight in weights.items()
        )
        for report in reports
    ]))


def add_restart_states(reservoirs, collected, limit, rng):
    for stage, states in collected.items():
        reservoir = reservoirs.setdefault(int(stage), [])
        for state in states:
            if len(reservoir) < limit:
                reservoir.append(state)
            else:
                index = rng.randrange(len(reservoir) + 1)
                if index < limit:
                    reservoir[index] = state


def choose_restart_state(reservoirs, rng):
    stages = [stage for stage, states in reservoirs.items() if states]
    if not stages:
        return None
    stage = rng.choices(stages, weights=[2 ** stage for stage in stages], k=1)[0]
    return rng.choice(reservoirs[stage]), stage
