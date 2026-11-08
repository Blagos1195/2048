from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfplay_ai as ai


ROOT = Path("models/comparison")
ADAPTIVE = ROOT / "adaptive" / "best.npz"
SEARCH_TARGET = ROOT / "search-target" / "latest.npz"
GAMES = 100
SEED_START = 31_000_000
BLEND_RATIOS = (0.25, 0.5, 0.75)


def main():
    adaptive = ai.NTupleValue()
    adaptive.load(ADAPTIVE)
    search_target = ai.NTupleValue()
    search_target.load(SEARCH_TARGET)

    workers = max(1, os.cpu_count() or 1)
    shared = shared_memory.SharedMemory(create=True, size=adaptive.weights.nbytes)
    shared_weights = np.ndarray(adaptive.weights.shape, dtype=np.float32, buffer=shared.buf)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=ai._init_parallel_worker,
        initargs=(shared.name, adaptive.weights.shape, adaptive.alpha, adaptive.gamma, 2),
    )

    output = ROOT / "blend-comparison-100-games.csv"
    models = [("adaptive-best", adaptive)]
    for ratio in BLEND_RATIOS:
        blended = ai.NTupleValue(alpha=adaptive.alpha, gamma=adaptive.gamma)
        blended.weights[:] = (
            adaptive.weights * (1.0 - ratio) + search_target.weights * ratio
        )
        blended.updates = max(adaptive.updates, search_target.updates)
        models.append((f"blend-{ratio:.2f}", blended))
    models.append(("search-target-latest", search_target))

    fields = [
        "model", "search_target_weight", "depth", "games", "avg_score",
        "median_score", "avg_max_tile", "best_max_tile", "avg_moves",
        "rate_2048", "rate_4096", "rate_8192", "seconds", "games_per_second",
    ]
    try:
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for name, evaluator in models:
                shared_weights[:] = evaluator.weights
                ratio = 0.0 if name == "adaptive-best" else (
                    1.0 if name == "search-target-latest" else float(name.split("-")[1])
                )
                print(
                    f"MODEL START name={name} workers={workers} games_per_depth={GAMES} "
                    f"seed_start={SEED_START}",
                    flush=True,
                )
                reports = ai.evaluate_depths(
                    evaluator, games=GAMES, depths=[2, 3],
                    seed_start=SEED_START, pool=pool,
                )
                for report in reports:
                    writer.writerow({
                        "model": name,
                        "search_target_weight": ratio,
                        "games": GAMES,
                        **report,
                    })
                    file.flush()
                    print(
                        f"RESULT model={name} "
                        + " ".join(f"{key}={value}" for key, value in report.items()),
                        flush=True,
                    )
        print(f"BLEND COMPARISON COMPLETE output={output}", flush=True)
    finally:
        pool.shutdown(wait=True)
        shared.close()
        shared.unlink()


if __name__ == "__main__":
    main()
