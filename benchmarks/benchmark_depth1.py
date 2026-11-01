import json
import statistics
import time

import numpy as np

from selfplay_ai import ExpectimaxAgent, NTupleValue, legal_actions

BOARDS = [
    [[2, 4, 8, 16], [0, 0, 4, 8], [0, 2, 16, 32], [0, 0, 0, 2]],
    [[2, 0, 4, 8], [2, 4, 8, 16], [0, 0, 16, 32], [0, 2, 4, 8]],
    [[0, 2, 4, 8], [0, 4, 8, 16], [2, 8, 16, 32], [4, 16, 32, 64]],
    [[4, 2, 0, 0], [8, 4, 2, 0], [16, 8, 4, 2], [32, 16, 8, 4]],
    [[0, 0, 2, 4], [0, 8, 4, 2], [16, 8, 16, 4], [32, 16, 8, 2]],
    [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]],
]

REPEATS = 40
DEPTH = 1

def main():
    evaluator = NTupleValue(seed=1)
    timings = []
    nodes = []
    tested_boards = 0
    for board_data in BOARDS:
        board = np.asarray(board_data, dtype=np.uint32)
        if not legal_actions(board):
            continue
        tested_boards += 1
        for i in range(REPEATS):
            agent = ExpectimaxAgent(evaluator, depth=DEPTH, seed=i)
            started = time.perf_counter()
            agent.choose_action(board.copy())
            timings.append((time.perf_counter() - started) * 1000)
            nodes.append(agent.nodes)

    ordered = sorted(timings)
    report = {
        "depth": DEPTH,
        "boards": tested_boards,
        "repeats_per_board": REPEATS,
        "samples": len(timings),
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "min_ms": min(timings),
        "max_ms": max(timings),
        "mean_search_nodes": statistics.mean(nodes),
    }
    print("BENCHMARK_RESULT=" + json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    main()
