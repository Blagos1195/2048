from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfplay_ai as ai


STOP = False


def stop_training(signum, frame):
    global STOP
    STOP = True


def atomic_copy(source: Path, destination: Path):
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def update_search_target(evaluator, state, transitions, index, horizon, depth):
    total = 0.0
    discount = 1.0
    terminal = False
    next_state = None
    for offset in range(horizon):
        if index + offset >= len(transitions):
            break
        _, reward, next_bytes, terminal = transitions[index + offset]
        total += discount * reward
        discount *= evaluator.gamma
        next_state = np.frombuffer(next_bytes, dtype=np.uint32).reshape(4, 4)
        if terminal:
            break
    if not terminal and next_state is not None:
        search = ai.ExpectimaxAgent(evaluator, depth=depth, seed=index + 1)
        total += discount * search._best_player(next_state.copy(), depth)
    indices = ai.pattern_indices(state)
    current = float(evaluator.weights[np.arange(ai.PATTERN_COUNT), indices].sum(dtype=np.float64))
    delta = total - current
    evaluator.weights[np.arange(ai.PATTERN_COUNT), indices] += np.float32(
        evaluator.alpha * delta / ai.PATTERN_COUNT
    )
    evaluator.updates += 1
    return abs(delta)


def apply_episode(evaluator, transitions, target_interval, horizon, target_depth):
    errors = []
    for index, (state_bytes, reward, next_bytes, terminal) in enumerate(transitions):
        state = np.frombuffer(state_bytes, dtype=np.uint32).reshape(4, 4)
        if index % target_interval == 0:
            errors.append(update_search_target(
                evaluator, state, transitions, index, horizon, target_depth
            ))
        else:
            next_state = np.frombuffer(next_bytes, dtype=np.uint32).reshape(4, 4)
            errors.append(abs(evaluator.update(state, reward, next_state, terminal)))
    return float(np.mean(errors)) if errors else 0.0


def evaluate_model(pool, shared_weights, evaluator, weights, depths, games, seed):
    shared_weights[:] = weights
    return ai.evaluate_depths(evaluator, games, depths, seed, pool=pool)


def quality(report):
    return report["avg_score"] + 500 * report["rate_2048"] + 1000 * report["rate_4096"]


def benchmark_models(pool, shared_weights, evaluator, models, output, episode):
    candidate_weights = evaluator.weights.copy()
    fields = [
        "episode", "model", "depth", "games", "avg_score", "median_score",
        "avg_max_tile", "rate_2048", "rate_4096", "rate_8192",
        "seconds", "games_per_second",
    ]
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for name, weights in models:
            reports = evaluate_model(
                pool, shared_weights, evaluator, weights,
                depths=[2, 3], games=10, seed=20_000_000,
            )
            for report in reports:
                writer.writerow({"episode": episode, "model": name, **report})
                print(
                    f"BENCHMARK episode={episode} model={name} "
                    + " ".join(f"{key}={value}" for key, value in report.items()),
                    flush=True,
                )
        file.flush()
    shared_weights[:] = candidate_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--out", default="models/search-target")
    parser.add_argument("--hours", type=float, default=5.75)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--target-interval", type=int, default=32)
    parser.add_argument("--target-horizon", type=int, default=3)
    parser.add_argument("--target-depth", type=int, default=3)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, stop_training)
    signal.signal(signal.SIGTERM, stop_training)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    starting_path = out / "starting.npz"
    champion_path = out / "published-champion.npz"
    latest_path = out / "latest.npz"
    best_path = out / "best.npz"
    atomic_copy(Path(args.resume), starting_path)
    atomic_copy(Path(args.resume), latest_path)
    atomic_copy(Path(args.resume), best_path)
    atomic_copy(Path(args.champion), champion_path)

    evaluator = ai.NTupleValue()
    evaluator.load(latest_path)
    starting = ai.NTupleValue()
    starting.load(starting_path)
    champion = ai.NTupleValue()
    champion.load(champion_path)

    workers = args.workers or (os.cpu_count() or 1)
    workers = max(1, workers)
    shared = shared_memory.SharedMemory(create=True, size=evaluator.weights.nbytes)
    shared_weights = np.ndarray(evaluator.weights.shape, dtype=np.float32, buffer=shared.buf)
    shared_weights[:] = evaluator.weights
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=ai._init_parallel_worker,
        initargs=(shared.name, evaluator.weights.shape, evaluator.alpha, evaluator.gamma, 2),
    )

    training_fields = ["episode", "score", "max_tile", "moves", "td_abs_mean", "updates"]
    training_path = out / "training.csv"
    benchmark_path = out / "benchmark.csv"
    episode_base = 0
    episode = 0
    completed = False
    try:
        with np.load(latest_path, allow_pickle=False) as checkpoint:
            import ast
            metadata = ast.literal_eval(str(checkpoint["metadata"].item()))
            episode_base = int(metadata.get("episode", metadata.get("episodes", 0)))

        initial = ai.evaluate_depths(
            evaluator, games=50, depths=[2], seed_start=1_000_000, pool=pool
        )[0]
        best_quality = quality(initial)
        print(f"Initial validation quality: {best_quality:.2f}", flush=True)

        started = time.monotonic()
        last_validation = 0
        last_benchmark = 0
        with training_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=training_fields)
            writer.writeheader()
            while time.monotonic() - started < args.hours * 3600 and not STOP:
                batch_size = workers * 2
                tasks = [
                    (20261106 + episode_base + episode + offset, 0.06, 0.15, 1024)
                    for offset in range(1, batch_size + 1)
                ]
                collected_batch = list(pool.map(ai._collect_episode, tasks))
                for collected in collected_batch:
                    td_mean = apply_episode(
                        evaluator, collected["transitions"], args.target_interval,
                        args.target_horizon, args.target_depth,
                    )
                    episode += 1
                    absolute_episode = episode_base + episode
                    writer.writerow({
                        "episode": absolute_episode,
                        "score": collected["score"],
                        "max_tile": collected["max_tile"],
                        "moves": collected["moves"],
                        "td_abs_mean": td_mean,
                        "updates": evaluator.updates,
                    })
                file.flush()

                if episode % 100 == 0:
                    print(
                        f"episode={episode_base + episode} additional={episode} "
                        f"workers={workers} updates={evaluator.updates:,} "
                        f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                        flush=True,
                    )

                if episode // 1000 > last_validation // 1000:
                    last_validation = episode
                    evaluator.save(latest_path, {
                        "episode": absolute_episode, "updates": evaluator.updates,
                        "target_depth": args.target_depth,
                        "target_interval": args.target_interval,
                        "target_horizon": args.target_horizon,
                    })
                    validation = ai.evaluate_depths(
                        evaluator, games=50, depths=[2], seed_start=1_000_000, pool=pool
                    )[0]
                    if quality(validation) > best_quality:
                        best_quality = quality(validation)
                        evaluator.save(best_path, {
                            "episode": absolute_episode, "validation": validation,
                            "target_depth": args.target_depth,
                        })
                        print(f"New experiment best at episode {absolute_episode}", flush=True)

                if episode // 5000 > last_benchmark // 5000:
                    last_benchmark = episode
                    benchmark_models(
                        pool, shared_weights, evaluator,
                        [
                            ("search-target", evaluator.weights.copy()),
                            ("starting-latest", starting.weights),
                            ("published-champion", champion.weights),
                        ],
                        benchmark_path, absolute_episode,
                    )

        evaluator.save(latest_path, {
            "episode": episode_base + episode, "updates": evaluator.updates,
            "target_depth": args.target_depth,
            "target_interval": args.target_interval,
            "target_horizon": args.target_horizon,
            "stopped": STOP,
        })
        experiment_best = ai.NTupleValue()
        experiment_best.load(best_path)
        benchmark_models(
            pool, shared_weights, evaluator,
            [
                ("experiment-best", experiment_best.weights),
                ("starting-latest", starting.weights),
                ("published-champion", champion.weights),
                ("experiment-latest", evaluator.weights.copy()),
            ],
            benchmark_path, episode_base + episode,
        )
        print(
            f"Training complete: {episode} additional episodes; "
            f"best validation quality={best_quality:.2f}",
            flush=True,
        )
        completed = True
    finally:
        if not completed:
            evaluator.save(latest_path, {"episode": episode_base + episode, "updates": evaluator.updates,
                                         "target_depth": args.target_depth, "interrupted": True})
        pool.shutdown(wait=True, cancel_futures=True)
        shared.close()
        shared.unlink()


if __name__ == "__main__":
    main()
