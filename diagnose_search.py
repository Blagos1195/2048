from __future__ import annotations

import argparse
import time

from selfplay_ai import ExpectimaxAgent, Game2048, NTupleValue, can_move, legal_actions


def diagnose(model: str, depth: int, seed: int, moves: int) -> None:
    evaluator = NTupleValue()
    evaluator.load(model)
    game = Game2048(seed)
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed)
    total_seconds = 0.0
    total_nodes = 0
    total_hits = 0

    print(f"SEARCH DIAGNOSTIC depth={depth} seed={seed} moves={moves} cache_limit={agent.cache_size}", flush=True)

    for move_number in range(1, moves + 1):
        if not can_move(game.board):
            break
        empty_count = int((game.board == 0).sum())
        legal_count = len(legal_actions(game.board))
        started = time.perf_counter()
        action = agent.choose_action(game.board.copy())
        elapsed = time.perf_counter() - started
        nodes = agent.nodes
        hits = agent.cache_hits
        total_seconds += elapsed
        total_nodes += nodes
        total_hits += hits

        print(
            f"MOVE {move_number}: seconds={elapsed:.3f} chance_nodes={nodes} "
            f"cache_hits={hits} empty_cells={empty_count} legal_actions={legal_count} "
            f"chance_cache={len(agent._chance_cache)} player_cache={len(agent._player_cache)} "
            f"action={action}",
            flush=True,
        )

        changed, _, _ = game.step(action)
        if not changed:
            raise RuntimeError("Search selected an illegal move")

    print(
        f"DIAGNOSTIC COMPLETE moves={game.moves} seconds={total_seconds:.3f} "
        f"chance_nodes={total_nodes} cache_hits={total_hits} score={game.score} max_tile={game.max_tile}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5200000)
    parser.add_argument("--moves", type=int, default=3)
    args = parser.parse_args()
    diagnose(args.model, args.depth, args.seed, args.moves)


if __name__ == "__main__":
    main()
