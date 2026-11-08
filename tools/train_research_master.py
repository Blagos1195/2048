from __future__ import annotations

import argparse
import csv
import os
import random
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfplay_ai as ai
import research_ai as research


STOP = False


def stop_training(signum, frame):
    global STOP
    STOP = True


def apply_episode(evaluator, transitions, target_interval, horizon, target_depth):
    errors = []
    for index, (state_bytes, reward, next_bytes, terminal) in enumerate(transitions):
        state = np.frombuffer(state_bytes, dtype=np.uint32).reshape(4, 4)
        if index % target_interval == 0:
            total = 0.0
            discount = 1.0
            next_state = None
            ended = False
            for offset in range(horizon):
                position = index + offset
                if position >= len(transitions):
                    break
                _, step_reward, next_bytes, ended = transitions[position]
                total += discount * step_reward
                discount *= evaluator.gamma
                next_state = np.frombuffer(next_bytes, dtype=np.uint32).reshape(4, 4)
                if ended:
                    break
            if not ended and next_state is not None:
                search = ai.ExpectimaxAgent(evaluator, depth=target_depth, seed=index + 1)
                total += discount * search._best_player(next_state.copy(), target_depth)
            errors.append(abs(evaluator.update_target(state, total)))
        else:
            next_state = np.frombuffer(next_bytes, dtype=np.uint32).reshape(4, 4)
            errors.append(abs(evaluator.update(state, reward, next_state, terminal)))
    return float(np.mean(errors)) if errors else 0.0


def run_validation(pool, shared_weights, evaluator, games, seed, depths=(2, 3)):
    shared_weights[:] = evaluator.weights
    return research.evaluate_depths(
        evaluator, games=games, depths=list(depths), seed_start=seed, pool=pool
    )


def save_comparison(path, reports):
    fields = ["model"] + list(reports[0][1][0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, model_reports in reports:
            for report in model_reports:
                writer.writerow({"model": name, **report})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--out", default="models/research-mastery")
    parser.add_argument("--hours", type=float, default=5.75)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--late-epsilon", type=float, default=0.08)
    parser.add_argument("--late-tile-threshold", type=int, default=1024)
    parser.add_argument("--target-interval", type=int, default=32)
    parser.add_argument("--target-horizon", type=int, default=4)
    parser.add_argument("--target-depth", type=int, default=2)
    parser.add_argument("--validation-games", type=int, default=20)
    parser.add_argument("--comparison-games", type=int, default=100)
    parser.add_argument("--deep-comparison-games", type=int, default=10)
    parser.add_argument("--optimism", type=float, default=2500.0)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    if args.hours <= 0 or args.workers < 0 or args.max_episodes < 0:
        raise SystemExit("hours and worker settings are invalid")
    if not 0 <= args.epsilon <= 1 or not 0 <= args.late_epsilon <= 1:
        raise SystemExit("exploration rates must be between 0 and 1")
    if args.target_interval < 1 or args.target_horizon < 1 or args.target_depth < 1:
        raise SystemExit("search-target settings must be positive")

    signal.signal(signal.SIGINT, stop_training)
    signal.signal(signal.SIGTERM, stop_training)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest_path = out / "latest.npz"
    best_path = out / "best.npz"
    baseline_path = out / "baseline.npz"
    training_path = out / "training.csv"
    benchmark_path = out / "comparison.csv"

    if latest_path.exists():
        evaluator = research.MultiStageNTupleValue()
        evaluator.load(latest_path)
    else:
        evaluator = research.MultiStageNTupleValue.from_checkpoint(
            args.resume, optimism=args.optimism
        )
        evaluator.save(latest_path, {"source": args.resume, "initialised": True})
        evaluator.save(best_path, {"source": args.resume, "initialised": True})

    baseline = research.MultiStageNTupleValue.from_checkpoint(args.resume, optimism=0.0)
    baseline.save(baseline_path, {"source": args.resume, "role": "baseline"})

    workers = max(1, args.workers or (os.cpu_count() or 1))
    memory = shared_memory.SharedMemory(create=True, size=evaluator.weights.nbytes)
    shared_weights = np.ndarray(
        evaluator.weights.shape, dtype=np.float32, buffer=memory.buf
    )
    shared_weights[:] = evaluator.weights
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=research.init_worker,
        initargs=(
            memory.name, evaluator.weights.shape,
            evaluator.alpha, evaluator.gamma, args.target_depth,
        ),
    )

    fields = [
        "episode", "kind", "start_stage", "score", "max_tile", "moves",
        "td_abs_mean", "updates", "elapsed_seconds",
    ]
    reservoirs = {}
    rng = random.Random(args.seed)
    episode = 0
    seed_counter = 0
    last_progress = 0
    best_quality = float("-inf")
    started = time.monotonic()
    last_validation = 0
    recent = []
    completed = False

    try:
        initial = run_validation(
            pool, shared_weights, evaluator, args.validation_games, 40_000_000
        )
        best_quality = research.composite_quality(initial)
        evaluator.save(best_path, {
            "source": args.resume, "initial_quality": best_quality,
            "stage_thresholds": research.STAGE_THRESHOLDS.tolist(),
        })
        print(
            f"INITIAL composite={best_quality:.2f} workers={workers} "
            f"stages={research.STAGE_THRESHOLDS.tolist()} "
            f"optimism={args.optimism} target_depth={args.target_depth}",
            flush=True,
        )

        with training_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if file.tell() == 0:
                writer.writeheader()

            while (
                time.monotonic() - started < args.hours * 3600
                and not STOP
                and (not args.max_episodes or episode < args.max_episodes)
            ):
                batch_size = workers
                tasks = [
                    (
                        args.seed + seed_counter + offset,
                        args.epsilon,
                        args.late_epsilon,
                        args.late_tile_threshold,
                        args.target_depth,
                        None,
                    )
                    for offset in range(1, batch_size + 1)
                ]
                seed_counter += batch_size
                collected_batch = list(pool.map(research.collect_episode, tasks))

                work = [("normal", collected) for collected in collected_batch]
                for collected in collected_batch:
                    research.add_restart_states(
                        reservoirs, collected["restart_states"], 1024, rng
                    )

                restart_count = min(max(1, workers // 2), len(reservoirs))
                restart_tasks = []
                for offset in range(restart_count):
                    selected = research.choose_restart_state(reservoirs, rng)
                    if selected is None:
                        break
                    state_bytes, stage = selected
                    restart_tasks.append((
                        args.seed + seed_counter + offset + 1,
                        args.epsilon,
                        args.late_epsilon,
                        args.late_tile_threshold,
                        args.target_depth,
                        state_bytes,
                    ))
                seed_counter += len(restart_tasks)
                if restart_tasks:
                    restarted = list(pool.map(research.collect_episode, restart_tasks))
                    work.extend(("restart", item) for item in restarted)
                    for _, collected in work:
                        research.add_restart_states(
                            reservoirs, collected["restart_states"], 1024, rng
                        )

                for kind, collected in work:
                    td_mean = apply_episode(
                        evaluator, collected["transitions"],
                        args.target_interval, args.target_horizon, args.target_depth,
                    )
                    episode += 1
                    elapsed = time.monotonic() - started
                    row = {
                        "episode": episode,
                        "kind": kind,
                        "start_stage": collected["start_stage"],
                        "score": collected["score"],
                        "max_tile": collected["max_tile"],
                        "moves": collected["moves"],
                        "td_abs_mean": td_mean,
                        "updates": evaluator.updates,
                        "elapsed_seconds": round(elapsed, 1),
                    }
                    writer.writerow(row)
                    recent.append(row)

                shared_weights[:] = evaluator.weights
                file.flush()

                if episode // 100 > last_progress // 100:
                    last_progress = episode
                    window = recent[-100:]
                    scores = np.asarray([row["score"] for row in window], dtype=np.float64)
                    tiles = np.asarray([row["max_tile"] for row in window], dtype=np.uint32)
                    print(
                        f"PROGRESS episode={episode} window={len(window)} "
                        f"avg_score={scores.mean():.1f} median_score={np.median(scores):.1f} "
                        + " ".join(
                            f"rate_{tile}={np.mean(tiles >= tile):.1%}"
                            for tile in research.MILESTONES
                        )
                        + f" restarts={sum(row['kind'] == 'restart' for row in window)} "
                        f"updates={evaluator.updates:,} "
                        f"updates_per_second={evaluator.updates / max(elapsed, 1):.1f} "
                        f"elapsed={elapsed / 60:.1f}m",
                        flush=True,
                    )
                    recent = []

                if episode // 2000 > last_validation // 2000:
                    last_validation = episode
                    evaluator.save(latest_path, {
                        "episode": episode,
                        "updates": evaluator.updates,
                        "stage_thresholds": research.STAGE_THRESHOLDS.tolist(),
                        "optimism": args.optimism,
                    })
                    validation = run_validation(
                        pool, shared_weights, evaluator,
                        args.validation_games, 40_000_000,
                    )
                    score = research.composite_quality(validation)
                    print(
                        f"VALIDATION episode={episode} composite={score:.2f} "
                        f"best={best_quality:.2f}",
                        flush=True,
                    )
                    if score > best_quality:
                        best_quality = score
                        evaluator.save(best_path, {
                            "episode": episode,
                            "updates": evaluator.updates,
                            "validation": validation,
                            "composite_quality": score,
                            "stage_thresholds": research.STAGE_THRESHOLDS.tolist(),
                        })
                        print(f"NEW BEST episode={episode} quality={score:.2f}", flush=True)
                    shared_weights[:] = evaluator.weights

        evaluator.save(latest_path, {
            "episode": episode,
            "updates": evaluator.updates,
            "stopped": STOP,
            "stage_thresholds": research.STAGE_THRESHOLDS.tolist(),
        })
        best = research.MultiStageNTupleValue()
        best.load(best_path)

        baseline_reports = run_validation(
            pool, shared_weights, baseline,
            args.comparison_games, 50_000_000, depths=(2,),
        )
        baseline_reports.extend(run_validation(
            pool, shared_weights, baseline,
            args.deep_comparison_games, 50_000_000, depths=(3,),
        ))
        best_reports = run_validation(
            pool, shared_weights, best,
            args.comparison_games, 50_000_000, depths=(2,),
        )
        best_reports.extend(run_validation(
            pool, shared_weights, best,
            10, 50_000_000, depths=(3,),
        ))
        save_comparison(benchmark_path, [
            ("baseline", baseline_reports),
            ("research-best", best_reports),
        ])
        baseline_quality = research.composite_quality(baseline_reports)
        best_comparison_quality = research.composite_quality(best_reports)
        print(
            f"FINAL COMPARISON games={args.comparison_games} "
            f"baseline={baseline_quality:.2f} research_best={best_comparison_quality:.2f} "
            f"improvement={best_comparison_quality - baseline_quality:+.2f}",
            flush=True,
        )
        completed = True
    finally:
        if not completed:
            evaluator.save(latest_path, {
                "episode": episode, "updates": evaluator.updates, "interrupted": True,
                "stage_thresholds": research.STAGE_THRESHOLDS.tolist(),
            })
        pool.shutdown(wait=True, cancel_futures=True)
        memory.close()
        memory.unlink()


if __name__ == "__main__":
    main()
