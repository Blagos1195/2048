from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path
from selfplay_ai import ExpectimaxAgent, Game2048, NTupleValue, can_move


def run_profile(model: str, depth: int, seed: int, moves: int, profiler: cProfile.Profile) -> dict:
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
        profiler.enable()
        try:
            action = agent.choose_action(game.board.copy())
        finally:
            profiler.disable()
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
    parser.add_argument("--output", default=None, help="optional text file for the profile report")
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be >= 1")
    if args.moves < 1:
        parser.error("--moves must be >= 1")

    profiler = cProfile.Profile()
    result = run_profile(args.model, args.depth, args.seed, args.moves, profiler)

    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).strip_dirs()
    stats.sort_stats("cumtime").print_stats(30)
    output.write("\\n=== PROFILE BY SELF TIME ===\\n")
    stats.sort_stats("tottime").print_stats(20)
    report = "\n".join(
        [*(f"{key}: {value}" for key, value in result.items()),
         "=== PROFILE BY CUMULATIVE TIME ===", output.getvalue()]
    )
    print(report)
    if args.output:
        profile_path = Path(args.output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(report, encoding="utf-8")
        print(f"Profile written to {profile_path}")


if __name__ == "__main__":
    main()
