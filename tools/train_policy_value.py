from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfplay_ai as game
from policy_value_ai import FastSelectiveAgent, PolicyValueNet, symmetries


ROOT = Path("models/policy-value")
TEACHER = Path("models/restored-search-target/latest.npz")


def collect(teacher, games=60):
    boards = []
    actions = []
    targets = []
    for episode in range(games):
        seed = 8_000_000 + episode
        env = game.Game2048(seed)
        agent = game.ExpectimaxAgent(teacher, depth=2, seed=seed)
        while game.can_move(env.board) and env.moves < 10000:
            state = env.board.copy()
            action = agent.choose_action(state)
            target = agent._player(state, agent.depth, action)
            for augmented, augmented_action in symmetries(state, action):
                boards.append(augmented)
                actions.append(augmented_action)
                targets.append(target)
            env.step(action)
        if (episode + 1) % 10 == 0:
            print(
                f"DATA games={episode + 1}/{games} positions={len(boards)} "
                f"score={env.score} max_tile={env.max_tile}",
                flush=True,
            )
    return boards, actions, targets


def play(model, seeds, baseline=False):
    rows = []
    decision_times = []
    for seed in seeds:
        env = game.Game2048(seed)
        agent = game.ExpectimaxAgent(model, depth=2, seed=seed) if baseline else FastSelectiveAgent(model)
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
    teacher = game.NTupleValue()
    teacher.load(TEACHER)
    boards, actions, targets = collect(teacher)
    np.savez_compressed(
        ROOT / "teacher-data.npz",
        boards=np.asarray(boards, dtype=np.uint32),
        actions=np.asarray(actions, dtype=np.int8),
        targets=np.asarray(targets, dtype=np.float32),
    )
    model = PolicyValueNet()
    model.train(boards, actions, targets, epochs=35)
    model.save(ROOT / "policy-value.npz")
    seeds = list(range(9_000_000, 9_000_040))
    baseline = play(teacher, seeds, baseline=True)
    candidate = play(model, seeds)
    with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", *baseline.keys()])
        writer.writeheader()
        writer.writerow({"model": "n-tuple-expectimax-depth-2", **baseline})
        writer.writerow({"model": "policy-value-selective-search", **candidate})
    print(f"BASELINE {baseline}", flush=True)
    print(f"POLICY_VALUE {candidate}", flush=True)
    print(f"COMPARISON {ROOT / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
