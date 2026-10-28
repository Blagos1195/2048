"""Self-learning 2048 agent: n-tuple TD learning with learned-value Expectimax."""
from __future__ import annotations

import argparse
import csv
import random
import time
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
    """Learned board evaluator; all table weights start at zero and change via TD updates."""
    def __init__(self, alpha: float = 0.08, gamma: float = 0.995, seed: int = 1):
        self.weights = np.zeros((PATTERN_COUNT, PATTERN_TABLE_SIZE), dtype=np.float32)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def value(self, board: np.ndarray) -> float:
        indices = pattern_indices(board)
        return float(self.weights[np.arange(PATTERN_COUNT), indices].sum(dtype=np.float64))

    def update(self, state: np.ndarray, reward: float, next_state: np.ndarray, terminal: bool) -> float:
        """Semi-gradient TD(0): fit current value to reward plus discounted next value."""
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
    """Expectimax uses the learned value function, not a fixed board-shape heuristic."""
    def __init__(self, evaluator: NTupleValue, depth: int = 1, seed: int = 1, epsilon: float = 0.0):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.evaluator = evaluator
        self.depth = int(depth)
        self.rng = random.Random(seed)
        self.epsilon = float(epsilon)
        self.nodes = 0

    def choose_action(self, board: np.ndarray) -> int:
        actions = legal_actions(board)
        if not actions:
            raise ValueError("No legal actions available")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        self.nodes = 0
        scored = [(self._player(board, self.depth, a), a) for a in actions]
        best = max(v for v, _ in scored)
        return self.rng.choice([a for v, a in scored if abs(v - best) < 1e-9])

    def _player(self, board: np.ndarray, depth: int, action: int) -> float:
        moved, merge_score = apply_action(board, action)
        if np.array_equal(moved, board):
            return -1e30
        immediate = merge_score / 128.0
        if depth <= 1:
            return immediate + self.evaluator.value(moved)
        return immediate + self._chance(moved, depth - 1)

    def _chance(self, board: np.ndarray, depth: int) -> float:
        self.nodes += 1
        empty = np.argwhere(board == 0)
        if len(empty) == 0:
            return self.evaluator.value(board) if can_move(board) else -2.0
        total = 0.0
        for r, c in empty:
            r, c = int(r), int(c)
            board[r, c] = 2
            total += 0.9 * self._best_player(board, depth)
            board[r, c] = 4
            total += 0.1 * self._best_player(board, depth)
            board[r, c] = 0
        return total / len(empty)

    def _best_player(self, board: np.ndarray, depth: int) -> float:
        actions = legal_actions(board)
        if not actions:
            return -2.0
        if depth <= 0:
            return self.evaluator.value(board)
        return max(self._player(board, depth, a) for a in actions)


def play_game(agent: ExpectimaxAgent, seed: int, train: bool = False, max_moves: int = 10000) -> dict:
    game = Game2048(seed)
    td_abs = 0.0
    td_count = 0
    while can_move(game.board) and game.moves < max_moves:
        state = game.board.copy()
        action = agent.choose_action(state)
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
    }


def evaluate(evaluator: NTupleValue, games: int = 50, depth: int = 2, seed_start: int = 900_000) -> dict:
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed_start)
    rows = [play_game(agent, seed_start + i, train=False) for i in range(games)]
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    tiles = np.asarray([r["max_tile"] for r in rows], dtype=np.float64)
    return {
        "games": games, "avg_score": float(scores.mean()), "median_score": float(np.median(scores)),
        "avg_max_tile": float(tiles.mean()), "best_max_tile": int(tiles.max()),
        "rate_2048": float(np.mean(tiles >= 2048)), "rate_4096": float(np.mean(tiles >= 4096)),
        "rate_8192": float(np.mean(tiles >= 8192)),
        "avg_moves": float(np.mean([r["moves"] for r in rows])),
    }


def train(args: argparse.Namespace) -> None:
    evaluator = NTupleValue(alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    if args.resume:
        evaluator.load(args.resume)
        print(f"Loaded {args.resume}; learned updates: {evaluator.updates}")
    agent = ExpectimaxAgent(evaluator, depth=args.depth, seed=args.seed, epsilon=args.epsilon)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "training.csv"
    mode = "a" if args.resume_metrics else "w"
    write_header = mode == "w" or not metrics_path.exists()
    with metrics_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "score", "max_tile", "moves", "epsilon", "td_abs_mean", "updates"])
        if write_header:
            writer.writeheader()
        best_eval = -float("inf")
        started = time.time()
        for episode in range(1, args.games + 1):
            agent.epsilon = max(args.min_epsilon, args.epsilon * (args.epsilon_decay ** (episode - 1)))
            result = play_game(agent, args.seed + episode, train=True)
            writer.writerow({"episode": episode, "score": result["score"], "max_tile": result["max_tile"],
                             "moves": result["moves"], "epsilon": agent.epsilon,
                             "td_abs_mean": result["td_abs_mean"], "updates": evaluator.updates})
            f.flush()
            if episode % args.log_every == 0 or episode == 1:
                elapsed = time.time() - started
                print(f"episode {episode}/{args.games} | score {result['score']} | max {result['max_tile']} | "
                      f"moves {result['moves']} | epsilon {agent.epsilon:.3f} | "
                      f"TD updates {evaluator.updates:,} | {elapsed:.1f}s", flush=True)
            if args.eval_every and episode % args.eval_every == 0:
                report = evaluate(evaluator, games=args.eval_games, depth=args.eval_depth,
                                  seed_start=args.eval_seed + episode * args.eval_games)
                print("EVAL " + " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in report.items()), flush=True)
                evaluator.save(out / "latest.npz", {"episode": episode, "evaluation": report})
                quality = report["avg_score"] + 500.0 * report["rate_2048"] + 1000.0 * report["rate_4096"]
                if quality > best_eval:
                    best_eval = quality
                    evaluator.save(out / "best.npz", {"episode": episode, "evaluation": report})
                    print(f"New champion saved: {out / 'best.npz'}", flush=True)
    evaluator.save(out / "latest.npz", {"episodes": args.games})
    print(f"Training finished. Latest model: {out / 'latest.npz'}")
    print(f"Best evaluated model: {out / 'best.npz'} (created when an evaluation interval is reached)")


def evaluate_cli(args: argparse.Namespace) -> None:
    evaluator = NTupleValue(alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    evaluator.load(args.model)
    report = evaluate(evaluator, games=args.games, depth=args.depth, seed_start=args.seed)
    for key, value in report.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-learning 2048: n-tuple TD + learned-value Expectimax")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train from scratch or resume a checkpoint")
    t.add_argument("--games", type=int, default=100_000)
    t.add_argument("--depth", type=int, default=1, help="training search depth; depth 1 is much faster")
    t.add_argument("--seed", type=int, default=20261028)
    t.add_argument("--alpha", type=float, default=0.08)
    t.add_argument("--gamma", type=float, default=0.995)
    t.add_argument("--epsilon", type=float, default=0.20)
    t.add_argument("--min-epsilon", type=float, default=0.02)
    t.add_argument("--epsilon-decay", type=float, default=0.99995)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--eval-games", type=int, default=30)
    t.add_argument("--eval-depth", type=int, default=1)
    t.add_argument("--eval-seed", type=int, default=1_000_000)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--out", default="models/selfplay")
    t.add_argument("--resume", default=None)
    t.add_argument("--resume-metrics", action="store_true")
    t.set_defaults(func=train)
    e = sub.add_parser("evaluate", help="evaluate a saved model on fixed seeds")
    e.add_argument("--model", required=True)
    e.add_argument("--games", type=int, default=100)
    e.add_argument("--depth", type=int, default=2)
    e.add_argument("--seed", type=int, default=2_000_000)
    e.add_argument("--alpha", type=float, default=0.08)
    e.add_argument("--gamma", type=float, default=0.995)
    e.set_defaults(func=evaluate_cli)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.depth < 1:
        raise SystemExit("--depth must be >= 1")
    if args.command == "train" and args.games < 1:
        raise SystemExit("--games must be >= 1")
    if args.command == "train" and args.eval_depth < 1:
        raise SystemExit("--eval-depth must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
