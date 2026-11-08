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


def quality(reports):
    if isinstance(reports, dict):
        reports = [reports]
    return sum(
        report["avg_score"]
        + 500 * report["rate_2048"]
        + 1000 * report["rate_4096"]
        + 2000 * report["rate_8192"]
        for report in reports
    ) / len(reports)


def benchmark_models(pool, shared_weights, evaluator, models, output, episode):
    candidate_weights = evaluator.weights.copy()
    fields = [
        "episode", "model", "depth", "games", "avg_score", "median_score",
        "avg_max_tile", "best_max_tile", "avg_moves", "rate_2048", "rate_4096",
        "rate_8192", "seconds", "games_per_second",
    ]
    existing_rows = []
    if output.exists() and output.stat().st_size:
        with output.open("r", newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            existing_rows = list(reader)
            if reader.fieldnames != fields:
                with output.open("w", newline="", encoding="utf-8") as rewritten:
                    writer = csv.DictWriter(rewritten, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(existing_rows)
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
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
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--late-epsilon", type=float, default=0.18)
    parser.add_argument("--validation-games", type=int, default=20)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, stop_training)
    signal.signal(signal.SIGTERM, stop_training)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    starting_path = out / "starting.npz"
    champion_path = out / "published-champion.npz"
    latest_path = out / "latest.npz"
    best_path = out / "best.npz"
    if not latest_path.exists():
        atomic_copy(Path(args.resume), starting_path)
        atomic_copy(Path(args.resume), latest_path)
        atomic_copy(Path(args.resume), best_path)
    if not starting_path.exists():
        atomic_copy(Path(args.resume), starting_path)
    if not best_path.exists():
        atomic_copy(Path(args.resume), best_path)
    if not champion_path.exists():
        atomic_copy(Path(args.champion), champion_path)

    evaluator = ai.NTupleValue()
    evaluator.load(latest_path)
    starting = ai.NTupleValue()
    starting.load(starting_path)
    champion = ai.NTupleValue()
    champion.load(champion_path)
    experiment_best = ai.NTupleValue()
    experiment_best.load(best_path)

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
            evaluator, games=args.validation_games, depths=[2, 3],
            seed_start=1_000_000, pool=pool
        )
        if np.array_equal(evaluator.weights, experiment_best.weights):
            best_quality = quality(initial)
        else:
            best_validation = ai.evaluate_depths(
                experiment_best, games=args.validation_games, depths=[2, 3],
                seed_start=1_000_000, pool=pool
            )
            best_quality = quality(best_validation)
        print(
            f"Initial latest composite quality: {quality(initial):.2f}; "
            f"retained best composite quality: {best_quality:.2f}; "
            + " | ".join(
                f"depth={report['depth']} avg={report['avg_score']:.1f} "
                f"2048={report['rate_2048']:.1%} 4096={report['rate_4096']:.1%} "
                f"8192={report['rate_8192']:.1%}"
                for report in initial
            ),
            flush=True,
        )

        started = time.monotonic()
        last_validation = 0
        last_benchmark = 0
        with training_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=training_fields)
            if file.tell() == 0:
                writer.writeheader()
            recent_games = []
            while time.monotonic() - started < args.hours * 3600 and not STOP:
                batch_size = workers * 2
                tasks = [
                    (20261108 + episode_base + episode + offset, args.epsilon, args.late_epsilon, 1024)
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
                    recent_games.append(collected | {"td_abs_mean": td_mean})

                    if len(recent_games) >= 100:
                        elapsed = time.monotonic() - started
                        recent_scores = np.asarray([game["score"] for game in recent_games], dtype=np.float64)
                        recent_tiles = np.asarray([game["max_tile"] for game in recent_games], dtype=np.float64)
                        print(
                            f"PROGRESS episode={absolute_episode} additional={episode} window=100 "
                            f"avg_score={recent_scores.mean():.1f} median_score={np.median(recent_scores):.1f} "
                            f"rate_2048={np.mean(recent_tiles >= 2048):.1%} "
                            f"rate_4096={np.mean(recent_tiles >= 4096):.1%} "
                            f"avg_max_tile={recent_tiles.mean():.1f} "
                            f"avg_moves={np.mean([game['moves'] for game in recent_games]):.1f} "
                            f"td_abs_mean={np.mean([game['td_abs_mean'] for game in recent_games]):.6g} "
                            f"updates={evaluator.updates:,} "
                            f"updates_per_second={evaluator.updates / max(elapsed, 1):.1f} "
                            f"episodes_per_minute={episode / max(elapsed / 60, 1e-9):.1f} "
                            f"elapsed={elapsed / 60:.1f}m",
                            flush=True,
                        )
                        recent_games.clear()
                file.flush()

                if episode // 2000 > last_validation // 2000:
                    last_validation = episode
                    evaluator.save(latest_path, {
                        "episode": absolute_episode, "updates": evaluator.updates,
                        "target_depth": args.target_depth,
                        "target_interval": args.target_interval,
                        "target_horizon": args.target_horizon,
                    })
                    validation = ai.evaluate_depths(
                        evaluator, games=args.validation_games, depths=[2, 3],
                        seed_start=1_000_000, pool=pool
                    )
                    validation_quality = quality(validation)
                    print(
                        f"VALIDATION episode={absolute_episode} composite={validation_quality:.2f} "
                        + " | ".join(
                            f"depth={report['depth']} avg={report['avg_score']:.1f} "
                            f"median={report['median_score']:.1f} "
                            f"2048={report['rate_2048']:.1%} "
                            f"4096={report['rate_4096']:.1%} "
                            f"8192={report['rate_8192']:.1%} "
                            f"max={report['best_max_tile']}"
                            for report in validation
                        )
                        + f" best_before={best_quality:.2f}",
                        flush=True,
                    )
                    if validation_quality > best_quality:
                        best_quality = validation_quality
                        evaluator.save(best_path, {
                            "episode": absolute_episode, "validation": validation,
                            "target_depth": args.target_depth,
                            "target_interval": args.target_interval,
                            "target_horizon": args.target_horizon,
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
