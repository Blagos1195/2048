from __future__ import annotations

import csv
import os
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfplay_ai as ai


ROOT = Path("models/comparison")
MODELS = [
    ("adaptive-best", ROOT / "adaptive" / "best.npz"),
    ("search-target-best", ROOT / "search-target" / "best.npz"),
    ("search-target-latest", ROOT / "search-target" / "latest.npz"),
    ("published-champion", ROOT / "search-target" / "published-champion.npz"),
]
GAMES = 100
SEED_START = 30_000_000


def main():
    workers = max(1, os.cpu_count() or 1)
    first = ai.NTupleValue()
    first.load(MODELS[0][1])
    shared = shared_memory.SharedMemory(create=True, size=first.weights.nbytes)
    shared_weights = np.ndarray(first.weights.shape, dtype=np.float32, buffer=shared.buf)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=ai._init_parallel_worker,
        initargs=(shared.name, first.weights.shape, first.alpha, first.gamma, 2),
    )
    output = ROOT / "comparison-100-games.csv"
    fields = [
        "model", "depth", "games", "avg_score", "median_score", "avg_max_tile",
        "best_max_tile", "avg_moves", "rate_2048", "rate_4096", "rate_8192",
        "seconds", "games_per_second",
    ]
    try:
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for name, path in MODELS:
                evaluator = ai.NTupleValue()
                evaluator.load(path)
                shared_weights[:] = evaluator.weights
                print(
                    f"MODEL START name={name} checkpoint={path} workers={workers} "
                    f"games_per_depth={GAMES} seed_start={SEED_START}",
                    flush=True,
                )
                reports = ai.evaluate_depths(
                    evaluator, games=GAMES, depths=[2, 3],
                    seed_start=SEED_START, pool=pool,
                )
                for report in reports:
                    row = {"model": name, "games": GAMES, **report}
                    writer.writerow(row)
                    file.flush()
                    print(
                        f"RESULT model={name} "
                        + " ".join(f"{key}={value}" for key, value in report.items()),
                        flush=True,
                    )
        print(f"COMPARISON COMPLETE output={output}", flush=True)
    finally:
        pool.shutdown(wait=True)
        shared.close()
        shared.unlink()


if __name__ == "__main__":
    main()
