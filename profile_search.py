from __future__ import annotations

import argparse
import cProfile
import pstats
import io
import random

import numpy as np

from selfplay_ai import ExpectimaxAgent, Game2048, NTupleValue, can_move


def run_profile(model: str, depth: int, seed: int, moves: int) -> dict:
    evaluator = NTupleValue()
    evaluator.load(model)
    game = Game2048(seed)
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed)
    total_nodes = 0
    total_hits = 0
    completed = 0

    for _ in range(moves):
        if not can_move(game.board):
            break
        action = agent.choose_action(game.board.copy())
        total_nodes += agent.nodes
        total_hits += agent.cache_hits
        changed, _, _ = game.step(action)
        if not changed:
            break
        completed += 1

    return {
        "moves": completed,
        "score": game.score,
        "max_tile": game.max_tile,
        "nodes": total_nodes,
        "cache_hits": total_hits,
        "chance_cache_entries": len(agent._chance_cache),
        "player_cache_entries": len(agent._player_cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5200000)
    parser.add_argument("--moves", type=int, default=40)
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    result = run_profile(args.model, args.depth, args.seed, args.moves)
    profiler.disable()

    for key, value in result.items():
        print(f"{key}: {value}")

    output = io.StringIO()
    pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumtime").print_stats(30)
    print("=== PROFILE BY CUMULATIVE TIME ===")
    print(output.getvalue())


if __name__ == "__main__":
    main()
