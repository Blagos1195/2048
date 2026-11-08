from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfplay_ai as game
from policy_ordering_ai import PolicyNet


ROOT = Path("models/policy-ordering")
TEACHER = Path("models/restored-search-target/latest.npz")
VECTORS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def transform_action(action, turns, mirrored):
    dr, dc = VECTORS[action]
    for _ in range(turns):
        dr, dc = -dc, dr
    if mirrored:
        dc = -dc
    return VECTORS.index((dr, dc))


def transform_board(board, turns, mirrored):
    result = np.rot90(np.asarray(board).reshape(4, 4), turns)
    if mirrored:
        result = np.fliplr(result)
    return result.copy()


def validate_symmetries():
    env = game.Game2048(77123)
    checked = 0
    for _ in range(40):
        board = env.board.copy()
        for action in game.legal_actions(board):
            moved, _, _ = game._apply_action(board, action)
            for turns in range(4):
                for mirrored in (False, True):
                    transformed = transform_board(board, turns, mirrored)
                    transformed_move = transform_action(action, turns, mirrored)
                    actual, _, changed = game._apply_action(transformed, transformed_move)
                    expected = transform_board(moved, turns, mirrored)
                    if not changed or not np.array_equal(actual, expected):
                        raise RuntimeError("Symmetry/action transformation check failed")
                    checked += 1
        legal = game.legal_actions(board)
        if legal:
            env.step(legal[0])
        else:
            break
    print(f"SYMMETRY_CHECK positions={checked} status=PASS", flush=True)


def collect(teacher, games=60):
    train_boards = []
    train_scores = []
    train_legal = []
    valid_boards = []
    valid_scores = []
    valid_legal = []
    for episode in range(games):
        env = game.Game2048(8_100_000 + episode)
        agent = game.ExpectimaxAgent(teacher, depth=2, seed=8_100_000 + episode)
        while game.can_move(env.board) and env.moves < 10000:
            state = env.board.copy()
            legal = game.legal_actions(state)
            scores = np.full(4, -1e9, dtype=np.float32)
            for action in legal:
                scores[action] = agent._player(state, agent.depth, action)
            best = max(float(scores[action]) for action in legal)
            action = agent.rng.choice([a for a in legal if abs(float(scores[a]) - best) < 1e-9])
            destination = (valid_boards, valid_scores, valid_legal) if episode >= games - 12 else (train_boards, train_scores, train_legal)
            boards_out, scores_out, legal_out = destination
            for turns in range(4):
                for mirrored in (False, True):
                    transformed_scores = np.full(4, -1e9, dtype=np.float32)
                    transformed_legal = np.zeros(4, dtype=bool)
                    for original in legal:
                        mapped = transform_action(original, turns, mirrored)
                        transformed_scores[mapped] = scores[original]
                        transformed_legal[mapped] = True
                    boards_out.append(transform_board(state, turns, mirrored).reshape(16))
                    scores_out.append(transformed_scores)
                    legal_out.append(transformed_legal)
            env.step(action)
        if (episode + 1) % 10 == 0:
            print(f"DATA games={episode + 1}/{games} train={len(train_boards)} validation={len(valid_boards)} score={env.score} max_tile={env.max_tile}", flush=True)
    return (
        np.asarray(train_boards, dtype=np.uint32),
        np.asarray(train_scores, dtype=np.float32),
        np.asarray(train_legal, dtype=bool),
        np.asarray(valid_boards, dtype=np.uint32),
        np.asarray(valid_scores, dtype=np.float32),
        np.asarray(valid_legal, dtype=bool),
    )


def policy_metrics(model, boards, scores, legal):
    logits = model.predict_batch(boards)
    logits = np.where(legal, logits, -1e9)
    top = np.argsort(logits, axis=1)[:, ::-1]
    best = np.max(np.where(legal, scores, -1e9), axis=1)
    best_mask = legal & (np.abs(scores - best[:, None]) < 1e-6)
    top1 = best_mask[np.arange(len(boards)), top[:, 0]]
    top2 = np.any(best_mask[np.arange(len(boards))[:, None], top[:, :2]], axis=1)
    return {"positions": int(len(boards)), "top1_teacher_agreement": float(top1.mean()), "top2_teacher_coverage": float(top2.mean())}


class PolicyOrderingAgent:
    def __init__(self, evaluator, policy, top_k, seed):
        self.evaluator = evaluator
        self.policy = policy
        self.top_k = top_k
        self.rng = random.Random(seed)
        self.depth = 2

    def choose_action(self, board):
        legal = game.legal_actions(board)
        if not legal:
            raise ValueError("No legal actions available")
        logits = self.policy.predict(board)
        ranked = sorted(legal, key=lambda action: float(logits[action]), reverse=True)[:self.top_k]
        scored = [(self.evaluator._player(board, self.depth, action), action) for action in ranked]
        best = max(value for value, _ in scored)
        return self.rng.choice([action for value, action in scored if abs(value - best) < 1e-9])


def play(teacher, policy, seeds, mode):
    rows = []
    decision_times = []
    for seed in seeds:
        env = game.Game2048(seed)
        if mode == "baseline":
            agent = game.ExpectimaxAgent(teacher, depth=2, seed=seed)
        else:
            agent = PolicyOrderingAgent(teacher, policy, int(mode), seed)
        while game.can_move(env.board) and env.moves < 10000:
            started = time.perf_counter()
            action = agent.choose_action(env.board)
            decision_times.append((time.perf_counter() - started) * 1000.0)
            env.step(action)
        rows.append((env.score, env.max_tile, env.moves))
    scores = np.asarray([row[0] for row in rows], dtype=np.float64)
    tiles = np.asarray([row[1] for row in rows], dtype=np.int64)
    return {
        "games": len(rows),
        "avg_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "rate_2048": float(np.mean(tiles >= 2048)),
        "rate_4096": float(np.mean(tiles >= 4096)),
        "rate_8192": float(np.mean(tiles >= 8192)),
        "avg_max_tile": float(tiles.mean()),
        "avg_moves": float(np.mean([row[2] for row in rows])),
        "decision_ms_mean": float(np.mean(decision_times)),
        "decision_ms_p95": float(np.percentile(decision_times, 95)),
        "decision_ms_max": float(np.max(decision_times)),
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    validate_symmetries()
    teacher = game.NTupleValue()
    teacher.load(TEACHER)
    train_boards, train_scores, train_legal, valid_boards, valid_scores, valid_legal = collect(teacher)
    model = PolicyNet()
    model.train(train_boards, train_scores, train_legal)
    validation = policy_metrics(model, valid_boards, valid_scores, valid_legal)
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    model.save(ROOT / "policy-ordering.npz")
    seeds = list(range(9_100_000, 9_100_100))
    results = {"baseline": play(teacher, model, seeds, "baseline")}
    results["top2"] = play(teacher, model, seeds, 2)
    results["top3"] = play(teacher, model, seeds, 3)
    with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", *results["baseline"].keys()])
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"model": name, **metrics})
    print(f"VALIDATION {validation}", flush=True)
    for name, metrics in results.items():
        print(f"RESULT {name} {metrics}", flush=True)


if __name__ == "__main__":
    main()
