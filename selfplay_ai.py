"""Self-learning 2048 agent: n-tuple TD learning with learned-value Expectimax."""
from __future__ import annotations

import argparse
import ast
import csv
import random
import time
import os
from collections import OrderedDict
try:
    from concurrent.futures import ProcessPoolExecutor
except ImportError:  # Browser runtimes do not provide process pools.
    ProcessPoolExecutor = None
try:
    from multiprocessing import shared_memory
except ImportError:  # Shared memory is only needed by desktop parallel training.
    shared_memory = None
from pathlib import Path

import numpy as np

ACTIONS = (0, 1, 2, 3)  # left, down, right, up
ACTION_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")
TILE_EXPONENT_CAP = 15
BASE = TILE_EXPONENT_CAP + 1
PATTERNS = tuple(
    [tuple(r * 4 + c for c in range(4)) for r in range(4)]
    + [tuple(r * 4 + c for r in range(4)) for c in range(4)]
)
PATTERN_COUNT = len(PATTERNS)
PATTERN_TABLE_SIZE = BASE ** 4
PATTERN_MEMBERSHIP = {cell: [] for cell in range(16)}
for _p, _cells in enumerate(PATTERNS):
    for _pos, _cell in enumerate(_cells):
        PATTERN_MEMBERSHIP[_cell].append((_p, _pos))


def exponent_board(board: np.ndarray) -> np.ndarray:
    """Convert tile values into exponents; zero remains the empty-cell code."""
    board = np.asarray(board, dtype=np.int64)
    out = np.zeros(board.shape, dtype=np.uint8)
    mask = board > 0
    out[mask] = np.minimum(np.rint(np.log2(board[mask])).astype(np.int64), TILE_EXPONENT_CAP)
    return out


def slide_left(board: np.ndarray) -> tuple[np.ndarray, int]:
    """Pure 2048 left slide; a newly merged tile cannot merge a second time."""
    out = np.zeros((4, 4), dtype=np.uint32)
    gained = 0
    for r in range(4):
        values = [int(v) for v in board[r] if v]
        merged: list[int] = []
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
        out[r, :len(merged)] = merged
    return out, gained


def apply_action(board: np.ndarray, action: int) -> tuple[np.ndarray, int]:
    """Return the resulting board and merge score for 0=left, 1=down, 2=right, 3=up."""
    if action not in ACTIONS:
        raise ValueError("action must be 0=left, 1=down, 2=right, or 3=up")
    rotated = np.rot90(np.asarray(board), -action)
    moved, score = slide_left(rotated)
    return np.rot90(moved, action).copy(), score


def legal_actions(board: np.ndarray) -> list[int]:
    return [a for a in ACTIONS if not np.array_equal(apply_action(board, a)[0], board)]


def can_move(board: np.ndarray) -> bool:
    if np.any(board == 0):
        return True
    return bool(np.any(board[:, :-1] == board[:, 1:]) or np.any(board[:-1, :] == board[1:, :]))


class Game2048:
    """Fast headless simulator with reproducible per-game randomness."""
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.board = np.zeros((4, 4), dtype=np.uint32)
        self.score = 0
        self.moves = 0
        self._spawn()
        self._spawn()

    def _spawn(self) -> None:
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return
        r, c = empty[self.rng.randrange(len(empty))]
        self.board[int(r), int(c)] = 4 if self.rng.random() < 0.1 else 2

    def step(self, action: int) -> tuple[bool, int, bool]:
        new_board, reward = apply_action(self.board, action)
        if np.array_equal(new_board, self.board):
            return False, 0, not can_move(self.board)
        self.board = new_board
        self.score += reward
        self.moves += 1
        self._spawn()
        return True, reward, not can_move(self.board)

    @property
    def max_tile(self) -> int:
        return int(self.board.max())


def pattern_indices(board: np.ndarray) -> np.ndarray:
    exponents = exponent_board(board).reshape(-1)
    result = np.empty(PATTERN_COUNT, dtype=np.int32)
    for p, cells in enumerate(PATTERNS):
        idx = 0
        for cell in cells:
            idx = idx * BASE + int(exponents[cell])
        result[p] = idx
    return result


class NTupleValue:
    """Learned board evaluator; weights start at zero and change via real TD updates."""
    def __init__(self, alpha: float = 0.08, gamma: float = 0.995, seed: int = 1):
        self.weights = np.zeros((PATTERN_COUNT, PATTERN_TABLE_SIZE), dtype=np.float32)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def value(self, board: np.ndarray) -> float:
        indices = pattern_indices(board)
        return float(self.weights[np.arange(PATTERN_COUNT), indices].sum(dtype=np.float64))

    def expected_spawn_value(self, board: np.ndarray) -> float:
        """Exact expected evaluator after one random spawn, without allocating child boards."""
        flat = np.asarray(board).reshape(-1)
        empty = np.flatnonzero(flat == 0)
        if len(empty) == 0:
            return self.value(board)
        indices = pattern_indices(board)
        base_value = float(self.weights[np.arange(PATTERN_COUNT), indices].sum(dtype=np.float64))
        total = 0.0
        for cell_value in empty:
            cell = int(cell_value)
            for exponent, probability in ((1, 0.9), (2, 0.1)):
                value = base_value
                for pattern, position in PATTERN_MEMBERSHIP[cell]:
                    new_index = int(indices[pattern]) + exponent * (BASE ** (3 - position))
                    value += float(self.weights[pattern, new_index] - self.weights[pattern, indices[pattern]])
                total += probability * value
        return total / len(empty)

    def update(self, state: np.ndarray, reward: float, next_state: np.ndarray, terminal: bool) -> float:
        """Semi-gradient TD(0): fit value to reward plus discounted next-state value."""
        indices = pattern_indices(state)
        current = float(self.weights[np.arange(PATTERN_COUNT), indices].sum(dtype=np.float64))
        target = float(reward) + (0.0 if terminal else self.gamma * self.value(next_state))
        delta = target - current
        self.weights[np.arange(PATTERN_COUNT), indices] += np.float32(self.alpha * delta / PATTERN_COUNT)
        self.updates += 1
        return delta

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, alpha=self.alpha, gamma=self.gamma,
                            updates=self.updates, metadata=np.array(str(metadata or {})))

    def load(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            weights = data["weights"]
            if weights.shape != self.weights.shape:
                raise ValueError(f"Incompatible checkpoint shape: {weights.shape}")
            self.weights[:] = weights
            self.alpha = float(data["alpha"])
            self.gamma = float(data["gamma"])
            self.updates = int(data["updates"])


class ExpectimaxAgent:
    """Expectimax with bounded cross-turn caching and iterative deepening."""
    def __init__(
        self,
        evaluator: NTupleValue,
        depth: int = 1,
        seed: int = 1,
        epsilon: float = 0.0,
        cache_size: int = 50000,
    ):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.evaluator = evaluator
        self.depth = int(depth)
        self.rng = random.Random(seed)
        self.epsilon = float(epsilon)
        self.nodes = 0
        self.cache_size = int(cache_size)
        # These caches intentionally survive choose_action() calls, allowing LARP
        # to reuse work from earlier turns. OrderedDict gives a bounded LRU cache.
        self._chance_cache: OrderedDict[tuple[bytes, int], float] = OrderedDict()
        self._player_cache: OrderedDict[tuple[bytes, int], float] = OrderedDict()
        self.cache_hits = 0

    def _cache_get(self, cache: OrderedDict, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
            self.cache_hits += 1
        return value

    def _cache_set(self, cache: OrderedDict, key, value: float) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def choose_action(self, board: np.ndarray) -> int:
        actions = legal_actions(board)
        if not actions:
            raise ValueError("No legal actions available")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        self.nodes = 0
        self.cache_hits = 0

        # Iterative deepening leaves a usable shallower result if a future time
        # budget is introduced, and its earlier passes warm the deeper search.
        scored = []
        for current_depth in range(1, self.depth + 1):
            scored = [(self._player(board, current_depth, a), a) for a in actions]
        best = max(v for v, _ in scored)
        return self.rng.choice([a for v, a in scored if abs(v - best) < 1e-9])

    def _player(self, board: np.ndarray, depth: int, action: int) -> float:
        moved, merge_score = apply_action(board, action)
        if np.array_equal(moved, board):
            return -1e30
        return merge_score / 128.0 + self._chance(moved, depth - 1)

    def _chance(self, board: np.ndarray, depth: int) -> float:
        key = (board.tobytes(), depth)
        cached = self._cache_get(self._chance_cache, key)
        if cached is not None:
            return cached
        self.nodes += 1
        if depth <= 0:
            result = self.evaluator.expected_spawn_value(board)
            self._cache_set(self._chance_cache, key, result)
            return result
        empty = np.argwhere(board == 0)
        if len(empty) == 0:
            result = self.evaluator.value(board) if can_move(board) else -2.0
            self._cache_set(self._chance_cache, key, result)
            return result
        total = 0.0
        for r, c in empty:
            r, c = int(r), int(c)
            board[r, c] = 2
            total += 0.9 * self._best_player(board, depth)
            board[r, c] = 4
            total += 0.1 * self._best_player(board, depth)
            board[r, c] = 0
        result = total / len(empty)
        self._cache_set(self._chance_cache, key, result)
        return result

    def _best_player(self, board: np.ndarray, depth: int) -> float:
        key = (board.tobytes(), depth)
        cached = self._cache_get(self._player_cache, key)
        if cached is not None:
            return cached
        actions = legal_actions(board)
        if not actions:
            return -2.0
        if depth <= 0:
            result = self.evaluator.value(board)
        else:
            result = max(self._player(board, depth, a) for a in actions)
        self._cache_set(self._player_cache, key, result)
        return result


def play_game(agent: ExpectimaxAgent, seed: int, train: bool = False, max_moves: int = 10000) -> dict:
    game = Game2048(seed)
    td_abs = 0.0
    td_count = 0
    search_nodes = 0
    while can_move(game.board) and game.moves < max_moves:
        state = game.board.copy()
        action = agent.choose_action(state)
        search_nodes += agent.nodes
        moved, gained, terminal = game.step(action)
        if not moved:
            if can_move(game.board):
                raise RuntimeError(f"Agent chose illegal action {ACTION_NAMES[action]}")
            terminal = True
        if train:
            reward = gained / 128.0
            if terminal:
                reward -= 2.0
            delta = agent.evaluator.update(state, reward, game.board.copy(), terminal)
            td_abs += abs(delta)
            td_count += 1
    return {
        "seed": seed, "score": game.score, "max_tile": game.max_tile, "moves": game.moves,
        "reached_2048": int(game.max_tile >= 2048), "reached_4096": int(game.max_tile >= 4096),
        "reached_8192": int(game.max_tile >= 8192), "td_abs_mean": td_abs / max(td_count, 1),
        "search_nodes": search_nodes,
    }


def _summarise_evaluation(rows: list[dict], depth: int, seconds: float) -> dict:
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    tiles = np.asarray([r["max_tile"] for r in rows], dtype=np.float64)
    games = len(rows)
    return {
        "depth": depth, "games": games, "avg_score": float(scores.mean()),
        "median_score": float(np.median(scores)), "avg_max_tile": float(tiles.mean()),
        "best_max_tile": int(tiles.max()), "rate_2048": float(np.mean(tiles >= 2048)),
        "rate_4096": float(np.mean(tiles >= 4096)), "rate_8192": float(np.mean(tiles >= 8192)),
        "avg_moves": float(np.mean([r["moves"] for r in rows])),
        "seconds": float(seconds), "games_per_second": games / max(seconds, 1e-9),
    }


def evaluate(evaluator: NTupleValue, games: int = 50, depth: int = 1, seed_start: int = 900_000) -> dict:
    started = time.perf_counter()
    rows = []
    for i in range(games):
        agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed_start + i)
        rows.append(play_game(agent, seed_start + i, train=False))
    return _summarise_evaluation(rows, depth, time.perf_counter() - started)


def _evaluate_game_worker(task: tuple[int, int]) -> dict:
    """Evaluate one game using the worker's read-only shared model snapshot."""
    seed, depth = task
    agent = ExpectimaxAgent(_WORKER_EVALUATOR, depth=depth, seed=seed)
    return play_game(agent, seed, train=False)


def evaluate_depths(
    evaluator: NTupleValue, games: int, depths: list[int], seed_start: int,
    pool=None,
) -> list[dict]:
    """Evaluate every depth on identical seeds, sharing one process pool across all depths."""
    started = time.perf_counter()
    tasks = [(seed_start + i, depth) for depth in depths for i in range(games)]
    if pool is not None:
        rows = list(pool.map(_evaluate_game_worker, tasks))
    else:
        rows = []
        for seed, depth in tasks:
            agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed)
            rows.append(play_game(agent, seed, train=False))
    elapsed = time.perf_counter() - started
    reports = []
    offset = 0
    for depth in depths:
        depth_rows = rows[offset:offset + games]
        offset += games
        reports.append(_summarise_evaluation(depth_rows, depth, elapsed))
    return reports


def evaluate_depths_parallel(
    evaluator: NTupleValue, games: int, depths: list[int], seed_start: int, workers: int,
) -> list[dict]:
    """Standalone all-core evaluation, used when no training pool exists yet."""
    if ProcessPoolExecutor is None or shared_memory is None or workers <= 1:
        return evaluate_depths(evaluator, games, depths, seed_start)
    shm = shared_memory.SharedMemory(create=True, size=evaluator.weights.nbytes)
    try:
        shared_weights = np.ndarray(evaluator.weights.shape, dtype=np.float32, buffer=shm.buf)
        shared_weights[:] = evaluator.weights
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_parallel_worker,
            initargs=(shm.name, evaluator.weights.shape, evaluator.alpha, evaluator.gamma, depths[0]),
        ) as pool:
            return evaluate_depths(evaluator, games, depths, seed_start, pool=pool)
    finally:
        shm.close()
        shm.unlink()


_WORKER_SHM = None
_WORKER_WEIGHTS = None
_WORKER_EVALUATOR = None
_WORKER_DEPTH = 1


def _init_parallel_worker(shm_name: str, shape: tuple[int, ...], alpha: float, gamma: float, depth: int):
    """Attach each worker to one read-only snapshot of the shared learned weights."""
    global _WORKER_SHM, _WORKER_WEIGHTS, _WORKER_EVALUATOR, _WORKER_DEPTH
    _WORKER_SHM = shared_memory.SharedMemory(name=shm_name)
    _WORKER_WEIGHTS = np.ndarray(shape, dtype=np.float32, buffer=_WORKER_SHM.buf)
    _WORKER_WEIGHTS.flags.writeable = False
    evaluator = NTupleValue.__new__(NTupleValue)
    evaluator.weights = _WORKER_WEIGHTS
    evaluator.alpha = float(alpha)
    evaluator.gamma = float(gamma)
    evaluator.rng = np.random.default_rng(1)
    evaluator.updates = 0
    _WORKER_EVALUATOR = evaluator
    _WORKER_DEPTH = int(depth)


def _collect_episode(task: tuple[int, float]) -> dict:
    """Generate one game against the current shared snapshot; learner updates happen in the parent."""
    seed, epsilon = task
    game = Game2048(seed)
    agent = ExpectimaxAgent(_WORKER_EVALUATOR, depth=_WORKER_DEPTH, seed=seed, epsilon=epsilon)
    transitions = []
    while can_move(game.board) and game.moves < 10000:
        state = game.board.copy()
        action = agent.choose_action(state)
        moved, gained, terminal = game.step(action)
        if not moved:
            if can_move(game.board):
                raise RuntimeError(f"Agent chose illegal action {ACTION_NAMES[action]}")
            terminal = True
        reward = gained / 128.0
        if terminal:
            reward -= 2.0
        transitions.append((state.tobytes(), reward, game.board.tobytes(), terminal))
    return {
        "seed": seed, "score": game.score, "max_tile": game.max_tile, "moves": game.moves,
        "reached_2048": int(game.max_tile >= 2048), "reached_4096": int(game.max_tile >= 4096),
        "reached_8192": int(game.max_tile >= 8192), "transitions": transitions,
    }


def _apply_collected_episode(evaluator: NTupleValue, transitions: list) -> float:
    td_abs = 0.0
    count = 0
    for state_bytes, reward, next_bytes, terminal in transitions:
        state = np.frombuffer(state_bytes, dtype=np.uint32).reshape(4, 4)
        next_state = np.frombuffer(next_bytes, dtype=np.uint32).reshape(4, 4)
        delta = evaluator.update(state, reward, next_state, terminal)
        td_abs += abs(delta)
        count += 1
    return td_abs / max(count, 1)


def train(args: argparse.Namespace) -> None:
    evaluator = NTupleValue(alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    episode_base = 0
    if args.resume:
        evaluator.load(args.resume)
        try:
            with np.load(args.resume, allow_pickle=False) as checkpoint:
                metadata = ast.literal_eval(str(checkpoint["metadata"].item()))
                episode_base = int(metadata.get("episode", metadata.get("episodes", 0)))
        except (OSError, ValueError, SyntaxError, AttributeError):
            episode_base = 0
        print(f"Loaded {args.resume}; learned updates: {evaluator.updates}; prior episodes: {episode_base}", flush=True)
    agent = ExpectimaxAgent(evaluator, depth=args.depth, seed=args.seed, epsilon=args.epsilon)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "training.csv"
    mode = "a" if args.resume_metrics else "w"
    write_header = mode == "w" or not metrics_path.exists()
    workers = args.workers
    if workers < 1:
        workers = max(1, os.cpu_count() or 1)
    workers = min(workers, args.games)
    print(f"Training with {workers} worker(s); updates are applied centrally in episode order.", flush=True)

    with metrics_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "score", "max_tile", "moves", "epsilon", "td_abs_mean", "updates"])
        if write_header:
            writer.writeheader()
        best_eval = -float("inf")
        evaluation_path = out / "evaluation.csv"
        evaluation_fields = ["episode", "depth", "games", "avg_score", "median_score", "avg_max_tile",
                             "best_max_tile", "rate_2048", "rate_4096", "rate_8192", "avg_moves",
                             "seconds", "games_per_second"]
        evaluation_exists = evaluation_path.exists() and evaluation_path.stat().st_size > 0
        evaluation_file = evaluation_path.open("a" if args.resume and evaluation_exists else "w",
                                               newline="", encoding="utf-8")
        evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=evaluation_fields)
        if not (args.resume and evaluation_exists):
            evaluation_writer.writeheader()
        existing_best = out / "best.npz"
        if args.resume and existing_best.exists():
            try:
                champion = NTupleValue(alpha=args.alpha, gamma=args.gamma, seed=args.seed)
                champion.load(existing_best)
                prior_reports = evaluate_depths_parallel(
                    champion, games=args.eval_games, depths=[args.eval_depth],
                    seed_start=args.eval_seed, workers=workers,
                )
                prior = prior_reports[0]
                best_eval = prior["avg_score"] + 500.0 * prior["rate_2048"] + 1000.0 * prior["rate_4096"]
                print(f"Existing champion validation quality: {best_eval:.2f}", flush=True)
            except (OSError, ValueError, KeyError) as exc:
                print(f"Could not validate existing best checkpoint ({exc}); selecting a new one.", flush=True)

        started = time.time()
        episode = 0
        pool = None
        shm = None
        try:
            if workers > 1:
                shm = shared_memory.SharedMemory(create=True, size=evaluator.weights.nbytes)
                shared_weights = np.ndarray(evaluator.weights.shape, dtype=np.float32, buffer=shm.buf)
                shared_weights[:] = evaluator.weights
                evaluator.weights = shared_weights
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_parallel_worker,
                    initargs=(shm.name, evaluator.weights.shape, args.alpha, args.gamma, args.depth),
                )

            while episode < args.games:
                batch_size = 1 if workers == 1 else workers * 2
                completed_absolute = episode_base + episode
                next_boundaries = []
                if args.eval_every:
                    next_boundaries.append(((completed_absolute // args.eval_every) + 1) * args.eval_every)
                if args.extended_eval_every:
                    next_boundaries.append(((completed_absolute // args.extended_eval_every) + 1) * args.extended_eval_every)
                if next_boundaries:
                    batch_size = min(batch_size, min(next_boundaries) - completed_absolute)
                batch_size = min(batch_size, args.games - episode)
                tasks = []
                for offset in range(1, batch_size + 1):
                    absolute_episode = episode_base + episode + offset
                    epsilon = max(args.min_epsilon, args.epsilon * (args.epsilon_decay ** (absolute_episode - 1)))
                    tasks.append((args.seed + absolute_episode, epsilon))

                if pool is None:
                    collected = _collect_episode_local(evaluator, tasks[0][0], tasks[0][1], args.depth)
                    result = {k: v for k, v in collected.items() if k != "transitions"}
                    result["td_abs_mean"] = _apply_collected_episode(evaluator, collected["transitions"])
                    batch = [result]
                else:
                    collected_batch = list(pool.map(_collect_episode, tasks))
                    batch = []
                    for collected in collected_batch:
                        td_mean = _apply_collected_episode(evaluator, collected["transitions"])
                        batch.append({k: v for k, v in collected.items() if k != "transitions"} | {"td_abs_mean": td_mean})

                for result in batch:
                    episode += 1
                    absolute_episode = episode_base + episode
                    epsilon = max(args.min_epsilon, args.epsilon * (args.epsilon_decay ** (absolute_episode - 1)))
                    writer.writerow({"episode": absolute_episode, "score": result["score"], "max_tile": result["max_tile"],
                                     "moves": result["moves"], "epsilon": epsilon,
                                     "td_abs_mean": result["td_abs_mean"], "updates": evaluator.updates})
                    f.flush()
                    if episode % args.log_every == 0 or episode == 1:
                        elapsed = time.time() - started
                        print(f"episode {episode}/{args.games} | score {result['score']} | max {result['max_tile']} | "
                              f"moves {result['moves']} | epsilon {epsilon:.3f} | workers {workers} | "
                              f"TD updates {evaluator.updates:,} | {elapsed:.1f}s", flush=True)

                if args.eval_every and (episode_base + episode) % args.eval_every == 0:
                    reports = evaluate_depths(evaluator, games=args.eval_games, depths=args.eval_depths,
                                              seed_start=args.eval_seed, pool=pool)
                    is_extended_checkpoint = (
                        args.extended_eval_every
                        and (episode_base + episode) % args.extended_eval_every == 0
                    )
                    if is_extended_checkpoint:
                        extra_depths = [d for d in args.extended_eval_depths if d not in args.eval_depths]
                        if extra_depths:
                            reports.extend(evaluate_depths(
                                evaluator, games=args.extended_eval_games, depths=extra_depths,
                                seed_start=args.eval_seed, pool=pool,
                            ))
                    for report in reports:
                        print("EVAL " + " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                    for k, v in report.items()), flush=True)
                        evaluation_writer.writerow({"episode": episode_base + episode, **report})
                    evaluation_file.flush()
                    validation = next((r for r in reports if r["depth"] == args.eval_depth), None)
                    if validation is None:
                        validation = evaluate(evaluator, games=args.eval_games, depth=args.eval_depth,
                                              seed_start=args.eval_seed)
                    metadata = {"episode": episode_base + episode, "evaluations": reports,
                                "validation_depth": args.eval_depth, "workers": workers}
                    evaluator.save(out / "latest.npz", metadata)
                    quality = validation["avg_score"] + 500.0 * validation["rate_2048"] + 1000.0 * validation["rate_4096"]
                    if quality > best_eval:
                        best_eval = quality
                        evaluator.save(out / "best.npz", metadata)
                        print(f"New champion saved using depth-{args.eval_depth} validation: {out / 'best.npz'}", flush=True)
            evaluator.save(out / "latest.npz", {"episodes": episode_base + args.games, "workers": workers})
            print(f"Training finished. Latest model: {out / 'latest.npz'}")
            print(f"Best evaluated model: {out / 'best.npz'}")
        finally:
            evaluation_file.close()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            if shm is not None:
                shm.close()
                shm.unlink()


def _collect_episode_local(evaluator: NTupleValue, seed: int, epsilon: float, depth: int) -> dict:
    """Single-process version of the same transition collection path."""
    game = Game2048(seed)
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed, epsilon=epsilon)
    transitions = []
    while can_move(game.board) and game.moves < 10000:
        state = game.board.copy()
        action = agent.choose_action(state)
        moved, gained, terminal = game.step(action)
        if not moved:
            if can_move(game.board):
                raise RuntimeError(f"Agent chose illegal action {ACTION_NAMES[action]}")
            terminal = True
        reward = gained / 128.0 - (2.0 if terminal else 0.0)
        transitions.append((state.tobytes(), reward, game.board.tobytes(), terminal))
    return {"seed": seed, "score": game.score, "max_tile": game.max_tile, "moves": game.moves,
            "reached_2048": int(game.max_tile >= 2048), "reached_4096": int(game.max_tile >= 4096),
            "reached_8192": int(game.max_tile >= 8192), "transitions": transitions}


def benchmark(evaluator: NTupleValue, games: int, depths: list[int], seed_start: int) -> None:
    """Compare search depths on identical deterministic game seeds."""
    reports = []
    for depth in depths:
        started = time.perf_counter()
        rows = [play_game(ExpectimaxAgent(evaluator, depth=depth, seed=seed_start), seed_start + i)
                for i in range(games)]
        elapsed = time.perf_counter() - started
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        tiles = np.asarray([row["max_tile"] for row in rows], dtype=np.float64)
        report = {
            "depth": depth,
            "games": games,
            "avg_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "rate_2048": float(np.mean(tiles >= 2048)),
            "rate_4096": float(np.mean(tiles >= 4096)),
            "avg_max_tile": float(tiles.mean()),
            "seconds": elapsed,
            "games_per_second": games / max(elapsed, 1e-9),
            "avg_search_nodes_per_move": float(np.mean([
                row["search_nodes"] / max(row["moves"], 1) for row in rows
            ])),
        }
        reports.append(report)
        print("BENCHMARK " + " | ".join(
            f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in report.items()
        ), flush=True)
    if len(reports) > 1:
        base = reports[0]
        for report in reports[1:]:
            score_change = 100.0 * (report["avg_score"] - base["avg_score"]) / max(abs(base["avg_score"]), 1e-9)
            speed_change = 100.0 * (report["games_per_second"] - base["games_per_second"]) / max(base["games_per_second"], 1e-9)
            print(f"DEPTH {report['depth']} VS DEPTH {base['depth']}: "
                  f"average score {score_change:+.1f}% | throughput {speed_change:+.1f}%", flush=True)

def evaluate_cli(args: argparse.Namespace) -> None:
    evaluator = NTupleValue(seed=args.seed)
    evaluator.load(args.model)
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    report = evaluate_depths_parallel(
        evaluator, games=args.games, depths=[args.depth], seed_start=args.seed, workers=workers,
    )[0]
    for key, value in report.items():
        print(f"{key}: {value}")


def benchmark_cli(args: argparse.Namespace) -> None:
    evaluator = NTupleValue(seed=args.seed)
    evaluator.load(args.model)
    benchmark(evaluator, games=args.games, depths=args.depths, seed_start=args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-learning 2048 AI. Train a model, test a checkpoint, or compare search depths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Examples:
  python main.py
  python main.py train --games 100000 --depth 1
  python main.py train --resume models/selfplay/latest.npz --games 50000
  python main.py evaluate --model models/selfplay/best.npz --games 100 --depth 2
  python main.py benchmark --model models/selfplay/best.npz --depths 1 2

Tip: start with the default training settings. Search depth 1 is much faster;
depth 2 can play better but takes considerably longer.""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser(
        "train",
        help="teach the AI by playing games",
        description="Train a new model from scratch, or continue from a saved checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    t.add_argument("--games", type=int, default=100_000,
                   help="number of training games to play")
    t.add_argument("--depth", type=int, default=1,
                   help="search depth used while training; 1 is fast, higher values are slower")
    t.add_argument("--seed", type=int, default=20261028,
                   help="random seed; reuse it to make runs more reproducible")
    t.add_argument("--alpha", type=float, default=0.08,
                   help="learning rate; larger values change learned weights more per update")
    t.add_argument("--gamma", type=float, default=0.995,
                   help="discount factor for future value; closer to 1 values future states more")
    t.add_argument("--epsilon", type=float, default=0.20,
                   help="starting chance of choosing a random legal move to explore")
    t.add_argument("--min-epsilon", type=float, default=0.02,
                   help="lowest exploration chance after epsilon decays")
    t.add_argument("--epsilon-decay", type=float, default=0.99995,
                   help="multiplier applied to exploration after each game; closer to 1 decays slower")
    t.add_argument("--eval-every", type=int, default=1000,
                   help="run evaluation every this many training games; 0 disables periodic evaluation")
    t.add_argument("--eval-games", type=int, default=50,
                   help="number of games per search depth at each evaluation checkpoint")
    t.add_argument("--eval-depths", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="search depths compared at each regular checkpoint")
    t.add_argument("--extended-eval-every", type=int, default=0,
                   help="run additional search-depth evaluations every this many games; 0 disables them")
    t.add_argument("--extended-eval-games", type=int, default=30,
                   help="games per additional search depth at extended checkpoints")
    t.add_argument("--extended-eval-depths", type=int, nargs="+", default=[1, 3],
                   help="additional search depths tested at extended checkpoints")
    t.add_argument("--eval-depth", type=int, default=2,
                   help="fixed validation depth used to decide whether a checkpoint becomes best.npz")
    t.add_argument("--eval-seed", type=int, default=1_000_000,
                   help="starting random seed for check-up games")
    t.add_argument("--log-every", type=int, default=25,
                   help="print training progress every this many games")
    t.add_argument("--out", default="models/selfplay",
                   help="folder for checkpoints and the training.csv metrics file")
    t.add_argument("--resume", default=None,
                   help="checkpoint .npz to continue training from instead of starting fresh")
    t.add_argument("--resume-metrics", action="store_true",
                   help="append to the existing training.csv instead of replacing it")
    t.add_argument("--workers", type=int, default=0,
                   help="parallel game workers; 0 automatically uses all available CPU cores")
    t.set_defaults(func=train)

    e = sub.add_parser(
        "evaluate",
        help="measure how well a saved model plays",
        description="Run games with a saved model and report scores, tile sizes, and win milestones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    e.add_argument("--model", required=True,
                   help="checkpoint .npz file to test")
    e.add_argument("--games", type=int, default=100,
                   help="number of test games; more games give a more reliable estimate")
    e.add_argument("--depth", type=int, default=2,
                   help="search depth; higher can play better but is much slower")
    e.add_argument("--seed", type=int, default=2_000_000,
                   help="starting random seed for repeatable test games")
    e.add_argument("--workers", type=int, default=0,
                   help="parallel evaluation workers; 0 automatically uses all available CPU cores")
    e.set_defaults(func=evaluate_cli)

    b = sub.add_parser(
        "benchmark",
        help="compare how search depths affect strength and speed",
        description="Test each depth on the same game seeds and report performance and speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    b.add_argument("--model", required=True,
                   help="checkpoint .npz file to benchmark")
    b.add_argument("--games", type=int, default=10,
                   help="number of games per search depth")
    b.add_argument("--depths", type=int, nargs="+", default=[1, 2],
                   help="one or more search depths to compare, for example: --depths 1 2 3")
    b.add_argument("--seed", type=int, default=4_000_000,
                   help="starting random seed; each depth uses the same seeds")
    b.set_defaults(func=benchmark_cli)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "depth") and args.depth < 1:
        raise SystemExit("--depth must be >= 1")
    if args.command == "train" and args.games < 1:
        raise SystemExit("--games must be >= 1")
    if args.command == "train" and (args.eval_depth < 1 or not args.eval_depths or min(args.eval_depths) < 1):
        raise SystemExit("evaluation depths must be >= 1")
    if args.command == "train" and args.eval_games < 1:
        raise SystemExit("--eval-games must be >= 1")
    if args.command == "train" and args.eval_every < 0:
        raise SystemExit("--eval-every must be >= 0")
    if args.command == "train" and args.extended_eval_every < 0:
        raise SystemExit("--extended-eval-every must be >= 0")
    if args.command == "train" and args.extended_eval_games < 1:
        raise SystemExit("--extended-eval-games must be >= 1")
    if args.command == "train" and (not args.extended_eval_depths or min(args.extended_eval_depths) < 1):
        raise SystemExit("extended evaluation depths must be >= 1")
    if args.command in ("train", "evaluate") and args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    if args.command == "benchmark" and (args.games < 1 or not args.depths or min(args.depths) < 1):
        raise SystemExit("benchmark games and search depths must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
