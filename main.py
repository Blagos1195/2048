"""2048 DQN training runner. Strategy is learned, never encoded as board heuristics."""
import glob
import json
import os
import random
import time
import threading
import csv
import numpy as np
import pygame

from ai import NeuralNetwork
from game import Game2048
from ui import (
    BACKGROUND_COLOR, PANEL_COLOR, init_tile_cache, draw_tiny_board,
    play_model_showcase, play_replay_showcase, play_larp_mode
)

# Viewing modes retained:
# 0=headless + terminal showcase hotkey, 1=multi-board, 2=slower multi-board,
# 3=single-game showcase, 4=LARP mode.
RUN_MODE = 0
RESUME_FROM_SAVED = True
MODEL_TO_LOAD = ""
PRINT_EVERY_N_EPISODES = 50
EVAL_EVERY_N_EPISODES = 100
EVAL_GAMES = 4
DEEP_EVAL_EVERY_N_EPISODES = 500
DEEP_EVAL_GAMES = 50
FINAL_EVAL_GAMES = 100
RANDOM_BASELINE_GAMES = 20
STATE_ENCODING = "tile_planes"  # One-hot tile categories, with symmetry augmentation
EXPERIMENT_NAME = "tile_planes_symmetry_v2"  # Fresh run; preserves the previous experiment
REWARD_SCALE = 128.0
SURVIVAL_REWARD = 0.01
TERMINAL_PENALTY = 1.0
CHECKPOINT_EVERY_N_EPISODES = 50
TRAIN_EVERY_N_MOVES = 4
BATCH_SIZE = 64
MAX_EPISODES = 0  # 0 means run until interrupted
RANDOM_SEED = None
ACTION_NAMES = Game2048.ACTION_NAMES

MODEL_DIR = os.path.join("models", EXPERIMENT_NAME)
LATEST_PATH = os.path.join(MODEL_DIR, "latest_checkpoint.npz")
BEST_PATH = os.path.join(MODEL_DIR, "best_ever_model.npz")
REPLAY_PATH = os.path.join(MODEL_DIR, "best_game_replay.npz")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.csv")


def save_model_with_meta(ai, filepath, generation, max_tile, score, extra=None):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    ai.save(filepath, include_training_state=("checkpoint" in os.path.basename(filepath)))
    metadata = {
        "generation": int(generation),
        "episode": int(generation),
        "max_tile": int(max_tile),
        "score": float(score),
        "timestamp": time.time(),
        "algorithm": "DQN",
        "architecture": [int(size) for size in ai.layer_sizes],
        "training_steps": int(ai.train_steps),
        "environment_steps": int(ai.environment_steps),
        "epsilon": float(ai.epsilon),
    }
    if extra:
        metadata.update(extra)
    with open(filepath + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_meta(filepath):
    try:
        with open(filepath + ".json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_agent(filepath, training_state=False):
    agent = NeuralNetwork()
    try:
        # Reject the old pre-DQN checkpoint format. It has no target network and
        # was previously saturated, so it must not silently become a training seed.
        with np.load(filepath, allow_pickle=False) as archive:
            if "TW1" not in archive or "Tb1" not in archive:
                raise ValueError("legacy non-DQN model format")
        agent.load(filepath, load_training_state=training_state)
        return agent
    except Exception as exc:
        print(f"[Model] Could not load {filepath}: {exc}")
        return None


def play_one_episode(ai, seed=None, explore=True, train=True, keep_history=False):
    game = Game2048(seed=seed)
    history = [game.get_state()] if keep_history else None
    episode_td_errors = []
    q_samples = []
    moves = 0

    while not game.game_over:
        state = game.get_state()
        if moves % 25 == 0:
            q_samples.extend(ai.q_values(state).astype(float).tolist())
        valid = game.valid_actions()
        if not valid:
            game.game_over = True
            break
        action = ai.select_action(state, valid, explore=explore)
        moved, score_gain = game.move(action)
        next_state = game.get_state()
        done = game.game_over
        # Only neutral, game-derived signals: actual merge score, a tiny survival
        # incentive for each legal move, and a terminal loss. No positional bonuses.
        reward = float(score_gain) / REWARD_SCALE + SURVIVAL_REWARD
        if done:
            reward -= TERMINAL_PENALTY
        if train:
            ai.store_experience(state, action, reward, next_state, done, game.valid_actions())
            if ai.environment_steps % TRAIN_EVERY_N_MOVES == 0:
                err = ai.learn(batch_size=BATCH_SIZE)
                if err:
                    episode_td_errors.append(err)
        if keep_history:
            history.append(next_state.copy())
        moves += 1
        if not moved:  # Should be unreachable because actions are legality-masked.
            break

    return {
        "score": int(game.score),
        "max_tile": int(np.max(game.board)),
        "moves": moves,
        "empty_cells": int(np.sum(game.board == 0)),
        "reached_2048": bool(np.max(game.board) >= 2048),
        "history": history,
        "mean_td_error": float(np.mean(episode_td_errors)) if episode_td_errors else 0.0,
        "q_value_min": float(np.min(q_samples)) if q_samples else 0.0,
        "q_value_max": float(np.max(q_samples)) if q_samples else 0.0,
        "q_value_mean_abs": float(np.mean(np.abs(q_samples))) if q_samples else 0.0,
    }


def evaluate_agent(ai, episode, games=EVAL_GAMES, seed_offset=100_000):
    """Frozen, reproducible evaluation. It never updates weights or exploration."""
    old_epsilon, old_steps = ai.epsilon, ai.environment_steps
    results = [play_one_episode(ai, seed=seed_offset + i, explore=False,
                                train=False, keep_history=True) for i in range(games)]
    ai.epsilon, ai.environment_steps = old_epsilon, old_steps
    scores = np.asarray([r["score"] for r in results], dtype=np.float64)
    tiles = np.asarray([r["max_tile"] for r in results], dtype=np.int64)
    summary = {
        "episode": int(episode), "games": int(games),
        "average_score": float(np.mean(scores)), "median_score": float(np.median(scores)),
        "average_max_tile": float(np.mean(tiles)), "best_max_tile": int(np.max(tiles)),
        "reached_2048_rate": float(np.mean([r["reached_2048"] for r in results])),
        "average_moves": float(np.mean([r["moves"] for r in results])),
    }
    best_run = max(results, key=lambda r: (r["score"], r["max_tile"], r["moves"]))
    summary.update({"best_history": best_run["history"], "best_run_score": best_run["score"],
                    "best_run_max_tile": best_run["max_tile"]})
    summary["evaluation_quality"] = (summary["average_score"]
        + 2000.0 * summary["reached_2048_rate"]
        + 100.0 * np.log2(max(summary["average_max_tile"], 2.0)))
    return summary


def evaluate_random_baseline(games=RANDOM_BASELINE_GAMES):
    """Evaluate a random legal-move policy as a simple, honest baseline."""
    results = []
    rng = random.Random(20261027)
    for _ in range(games):
        game = Game2048(rng=rng)
        moves = 0
        while not game.game_over:
            valid = game.valid_actions()
            if not valid:
                break
            game.move(rng.choice(valid))
            moves += 1
        results.append((game.score, int(np.max(game.board)), moves, int(np.max(game.board) >= 2048)))
    return {"games": games, "average_score": float(np.mean([r[0] for r in results])),
            "median_score": float(np.median([r[0] for r in results])),
            "average_max_tile": float(np.mean([r[1] for r in results])),
            "reached_2048_rate": float(np.mean([r[3] for r in results])),
            "average_moves": float(np.mean([r[2] for r in results]))}


METRIC_FIELDS = ["record_type", "episode", "games", "score", "max_tile", "moves",
                 "empty_cells", "reached_2048", "mean_td_error", "epsilon", "replay_size",
                 "train_steps", "environment_steps", "episode_seconds", "moves_per_second",
                 "q_value_min", "q_value_max", "q_value_mean_abs", "average_score", "median_score",
                 "average_max_tile", "best_max_tile", "reached_2048_rate", "average_moves",
                 "best_run_score", "best_run_max_tile", "evaluation_quality", "average_td_error",
                 "state_encoding", "reward_scale", "survival_reward", "terminal_penalty"]

def append_metrics(row):
    """Write metrics without allowing a locked CSV to stop training or final saves."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Excel and some sync tools lock CSV files on Windows. If the main log is
    # unavailable, preserve the row in a separate recovery log instead.
    recovery_path = os.path.join(MODEL_DIR, "training_metrics_recovery.csv")
    paths_to_try = (METRICS_PATH, recovery_path)
    errors = []

    for path in paths_to_try:
        try:
            needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=METRIC_FIELDS, extrasaction="ignore"
                )
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            if path != METRICS_PATH:
                print(
                    f"[Metrics] Main CSV is locked or unavailable; "
                    f"row saved to {path}"
                )
            return
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    # Logging is useful, but it must never prevent the checkpoint from saving.
    print("[Metrics] Could not write this row: " + " | ".join(errors))


def load_checkpoint(ai, path):
    if not os.path.exists(path):
        return 0, 2, -1.0
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "TW1" not in archive or "Tb1" not in archive:
                raise ValueError("checkpoint is not in the DQN format")
            saved_encoding = (
                str(archive["state_encoding"].item())
                if "state_encoding" in archive else "log"
            )
            if saved_encoding != ai.state_encoding:
                raise RuntimeError(
                    f"Checkpoint encoding mismatch: saved model uses "
                    f"'{saved_encoding}', but this run is configured for "
                    f"'{ai.state_encoding}'. Use a new EXPERIMENT_NAME or "
                    f"restore the matching STATE_ENCODING. The checkpoint "
                    f"has not been loaded or overwritten."
                )
        ai.load(path, load_training_state=True)
        meta = load_meta(path)
        episode = int(meta.get("episode", meta.get("generation", 0)))
        print(f"[Resume] Loaded DQN checkpoint at episode {episode}, "
              f"step {ai.train_steps}, epsilon {ai.epsilon:.3f}, "
              f"replay {len(ai.memory_buffer):,} transitions.")
        return episode, int(meta.get("max_tile", 2)), float(meta.get("best_evaluation_quality", -1.0))
    except RuntimeError as exc:
        # An encoding mismatch must stop startup: silently training fresh into
        # the same experiment directory could overwrite the incompatible model.
        if "Checkpoint encoding mismatch" in str(exc):
            raise
        print(f"[Resume] Checkpoint unavailable ({exc}); starting with a fresh network.")
        return 0, 2, -1.0
    except Exception as exc:
        print(f"[Resume] Checkpoint unavailable ({exc}); starting with a fresh network.")
        return 0, 2, -1.0


def run_evolution():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Showcase-only modes prefer a genuinely trained model. The old saturated model
    # remains untouched on disk as a historical artifact, but is not a training seed.
    if RUN_MODE in (3, 4):
        candidates = [LATEST_PATH, BEST_PATH]
        if MODEL_TO_LOAD:
            candidates.insert(0, MODEL_TO_LOAD)
        agent = None
        meta = {}
        for path in candidates:
            if os.path.exists(path):
                agent = load_agent(path, training_state=False)
                if agent is not None:
                    meta = load_meta(path)
                    break
        if agent is None and os.path.exists("best_ever_model.npz"):
            # Keep the old showcase/LARP mode usable before new training has produced
            # a checkpoint. This legacy network is read-only and never seeds training.
            try:
                agent = NeuralNetwork()
                agent.load("best_ever_model.npz", load_training_state=False)
                meta = load_meta("best_ever_model.npz")
                print("[Showcase] Using the preserved legacy model for display only; "
                      "training still starts from a fresh DQN.")
            except Exception as exc:
                print(f"[Showcase] Legacy display model could not be loaded: {exc}")
                agent = None
        if agent is None:
            print("[Showcase] No compatible model found. Run training first.")
            return
        if RUN_MODE == 3:
            play_model_showcase(agent, meta.get("episode", meta.get("generation", 0)), meta.get("max_tile", 2), 0, 0)
        else:
            play_larp_mode(agent)
        return

    seed = RANDOM_SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ai = NeuralNetwork(seed=seed, state_encoding=STATE_ENCODING)
    episode, max_tile_ever, best_evaluation_quality = 0, 2, -1.0
    if RESUME_FROM_SAVED and os.path.exists(LATEST_PATH):
        episode, max_tile_ever, best_evaluation_quality = load_checkpoint(ai, LATEST_PATH)
    elif MODEL_TO_LOAD and os.path.exists(MODEL_TO_LOAD):
        # Explicitly requested compatible DQN checkpoints only. The old best-ever
        # weights are not auto-loaded, because they were saturated and pre-DQN.
        loaded = load_agent(MODEL_TO_LOAD, training_state=True)
        if loaded is not None:
            ai = loaded
            meta = load_meta(MODEL_TO_LOAD)
            episode = int(meta.get("episode", 0))
            max_tile_ever = int(meta.get("max_tile", 2))
            print(f"[Resume] Explicit checkpoint loaded: {MODEL_TO_LOAD}")

    print(f"[DQN] Experiment {EXPERIMENT_NAME} | Mode {RUN_MODE} | Architecture {ai.layer_sizes} | "
          f"Replay {ai.memory_buffer.capacity:,} | Batch {BATCH_SIZE} | "
          f"Target sync every 500 updates | No hand-coded strategy")
    print("[DQN] Training starts with a fresh network unless a DQN checkpoint is resumed.")
    baseline = evaluate_random_baseline()
    print(f"[Baseline: random legal moves] {baseline}")
    append_metrics({"record_type": "random_baseline", **baseline, "state_encoding": ai.state_encoding, "reward_scale": REWARD_SCALE, "survival_reward": SURVIVAL_REWARD, "terminal_penalty": TERMINAL_PENALTY})

    terminal_choice = "1"
    terminal_trigger = threading.Event()

    def terminal_listener():
        nonlocal terminal_choice
        while True:
            try:
                input()
                terminal_choice = input("Showcase: [1] live model, [2] best replay: ").strip()
                terminal_trigger.set()
            except EOFError:
                return

    if RUN_MODE == 0:
        threading.Thread(target=terminal_listener, daemon=True).start()
        print("[Headless] Training. Press Enter for a showcase menu.")

    screen = None
    clock = pygame.time.Clock()
    font = small_font = None
    if RUN_MODE in (1, 2):
        pygame.init()
        screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
        pygame.display.set_caption("2048 DQN | SPACE: Live Showcase | R: Replay | S: Save")
        font = pygame.font.SysFont("arial", 16, bold=True)
        small_font = pygame.font.SysFont("arial", 13, bold=True)
        init_tile_cache(pygame.font.SysFont("arial", 20, bold=True))

    live_games = [Game2048() for _ in range(4)]
    best_history = []
    best_train_key = (-1, 2)
    recent_scores = []
    recent_tiles = []
    recent_td_errors = []
    session_start = time.time()
    last_eval = {}
    running = True

    try:
        while running and (MAX_EPISODES == 0 or episode < MAX_EPISODES):
            episode += 1
            episode_start = time.time()
            episode_seed = int(ai.rng.integers(0, 2**31 - 1))
            result = play_one_episode(ai, seed=episode_seed, explore=True, train=True, keep_history=True)
            episode_seconds = max(time.time() - episode_start, 1e-6)
            recent_scores.append(result["score"])
            recent_tiles.append(result["max_tile"])
            recent_td_errors.append(result["mean_td_error"])
            recent_scores = recent_scores[-100:]
            recent_tiles = recent_tiles[-100:]
            recent_td_errors = recent_td_errors[-100:]
            max_tile_ever = max(max_tile_ever, result["max_tile"])
            if (result["score"], result["max_tile"]) > best_train_key:
                best_train_key = (result["score"], result["max_tile"])
                if not last_eval:
                    best_history = result["history"]

            if episode % EVAL_EVERY_N_EPISODES == 0 or episode == 1:
                last_eval = evaluate_agent(ai, episode, EVAL_GAMES, 100_000)
                print(f"[Quick eval {last_eval['games']} games] Ep {episode} | avg {last_eval['average_score']:.1f} | "
                      f"median {last_eval['median_score']:.1f} | avg max tile {last_eval['average_max_tile']:.0f} | "
                      f"2048 rate {last_eval['reached_2048_rate']:.0%} | moves {last_eval['average_moves']:.0f}")
                append_metrics({"record_type": "quick_eval", **{k:v for k,v in last_eval.items() if k != "best_history"},
                                "state_encoding": ai.state_encoding, "epsilon": ai.epsilon,
                                "train_steps": ai.train_steps, "environment_steps": ai.environment_steps,
                                "reward_scale": REWARD_SCALE, "survival_reward": SURVIVAL_REWARD,
                                "terminal_penalty": TERMINAL_PENALTY})

            # Larger validation set every 100 episodes is used for model selection.
            if episode % DEEP_EVAL_EVERY_N_EPISODES == 0:
                deep = evaluate_agent(ai, episode, DEEP_EVAL_GAMES, 200_000)
                is_best = deep["evaluation_quality"] > best_evaluation_quality
                append_metrics({"record_type": "deep_eval", **{k:v for k,v in deep.items() if k != "best_history"},
                                "state_encoding": ai.state_encoding, "epsilon": ai.epsilon,
                                "train_steps": ai.train_steps, "environment_steps": ai.environment_steps,
                                "reward_scale": REWARD_SCALE, "survival_reward": SURVIVAL_REWARD,
                                "terminal_penalty": TERMINAL_PENALTY})
                print(f"[Deep eval {DEEP_EVAL_GAMES} games] Ep {episode} | avg {deep['average_score']:.1f} | "
                      f"median {deep['median_score']:.1f} | 2048 rate {deep['reached_2048_rate']:.0%} | "
                      f"avg max tile {deep['average_max_tile']:.0f}")
                if is_best:
                    best_evaluation_quality = deep["evaluation_quality"]
                    best_history = deep["best_history"]
                    save_model_with_meta(ai, BEST_PATH, episode, deep["best_max_tile"], deep["average_score"],
                                         {**{k:v for k,v in deep.items() if k != "best_history"}, "state_encoding": ai.state_encoding, "reward_scale": REWARD_SCALE, "survival_reward": SURVIVAL_REWARD, "terminal_penalty": TERMINAL_PENALTY})
                    np.savez_compressed(REPLAY_PATH, boards=np.asarray(best_history, dtype=np.int64))
                    print(f"[BEST] New best frozen-evaluation model saved at episode {episode}.")

            if episode % CHECKPOINT_EVERY_N_EPISODES == 0:
                save_model_with_meta(ai, LATEST_PATH, episode, max_tile_ever,
                                     float(np.mean(recent_scores)),
                                     {"recent_average_score": float(np.mean(recent_scores)),
                                      "best_evaluation_quality": best_evaluation_quality,
                                      "state_encoding": ai.state_encoding, "reward_scale": REWARD_SCALE})

            if episode % PRINT_EVERY_N_EPISODES == 0 or episode == 1:
                elapsed = max(time.time() - session_start, 0.001)
                print(f"Ep {episode:6d} | train avg {np.mean(recent_scores):7.1f} | "
                      f"median {np.median(recent_scores):7.1f} | "
                      f"avg max tile {np.mean(recent_tiles):6.0f} | "
                      f"epsilon {ai.epsilon:.3f} | replay {len(ai.memory_buffer):5d} | "
                      f"TD {np.mean(recent_td_errors):.3f} | "
                      f"{ai.environment_steps/elapsed:.0f} moves/s")

            append_metrics({"record_type": "train_episode", "episode": episode,
                            "score": result["score"], "max_tile": result["max_tile"],
                            "moves": result["moves"], "empty_cells": result["empty_cells"],
                            "reached_2048": int(result["reached_2048"]),
                            "mean_td_error": result["mean_td_error"], "q_value_min": result["q_value_min"],
                            "q_value_max": result["q_value_max"], "q_value_mean_abs": result["q_value_mean_abs"],
                            "episode_seconds": episode_seconds, "moves_per_second": result["moves"] / episode_seconds,
                            "epsilon": ai.epsilon, "replay_size": len(ai.memory_buffer),
                            "train_steps": ai.train_steps, "environment_steps": ai.environment_steps,
                            "state_encoding": ai.state_encoding,
                            "reward_scale": REWARD_SCALE, "survival_reward": SURVIVAL_REWARD,
                            "terminal_penalty": TERMINAL_PENALTY})

            if RUN_MODE == 0 and terminal_trigger.is_set():
                terminal_trigger.clear()
                if terminal_choice == "2" and best_history:
                    play_replay_showcase(best_history, episode, max_tile_ever, 0, 0)
                else:
                    play_model_showcase(ai, episode, max_tile_ever, 0, 0)

            if screen is not None:
                # Keep the original multi-board showcase concept. Four parallel games
                # display the current frozen greedy policy; only the main episode trains.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            play_model_showcase(ai, episode, max_tile_ever, 0, 0)
                            pygame.init()
                            screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
                            init_tile_cache(pygame.font.SysFont("arial", 20, bold=True))
                        elif event.key == pygame.K_r:
                            play_replay_showcase(best_history, episode, max_tile_ever, 0, 0)
                            pygame.init()
                            screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
                            init_tile_cache(pygame.font.SysFont("arial", 20, bold=True))
                        elif event.key == pygame.K_s:
                            save_model_with_meta(ai, LATEST_PATH, episode, max_tile_ever,
                                                 float(np.mean(recent_scores)))
                screen.fill(BACKGROUND_COLOR)
                pygame.draw.rect(screen, PANEL_COLOR, (0, 0, screen.get_width(), 52))
                banner = (f"DQN | Ep {episode} | Best tile {max_tile_ever} | "
                          f"Avg score {np.mean(recent_scores):.0f} | "
                          f"ε {ai.epsilon:.2f} | [SPACE] Live [R] Replay [S] Save")
                screen.blit(small_font.render(banner, True, (255, 255, 255)), (16, 18))
                cols, rows = 2, 2
                panel_w = (screen.get_width() - 36) / cols
                panel_h = (screen.get_height() - 68) / rows
                for i, game in enumerate(live_games):
                    if game.game_over:
                        live_games[i] = game = Game2048()
                    valid = game.valid_actions()
                    if valid:
                        action = ai.select_action(game.get_state(), valid, explore=False)
                        game.move(action)
                    r, c = divmod(i, cols)
                    x, y = 12 + c * panel_w, 60 + r * panel_h
                    pygame.draw.rect(screen, PANEL_COLOR,
                                     (x, y, panel_w - 8, panel_h - 8), border_radius=6)
                    draw_tiny_board(screen, game, x + 18, y + 42, min(panel_w - 44, panel_h - 58))
                    header = f"Greedy demo {i+1} | score {game.score} | max {np.max(game.board)}"
                    screen.blit(small_font.render(header, True, (255, 255, 255)), (x + 12, y + 12))
                pygame.display.flip()
                clock.tick(12 if RUN_MODE == 2 else 30)

    except KeyboardInterrupt:
        print("\n[Exit] Saving resumable training state.")
    finally:
        if ai.environment_steps > 0:
            final_eval = evaluate_agent(ai, episode, FINAL_EVAL_GAMES, 300_000)
            append_metrics({"record_type": "final_eval", **{k:v for k,v in final_eval.items() if k != "best_history"},
                            "state_encoding": ai.state_encoding, "epsilon": ai.epsilon,
                            "train_steps": ai.train_steps, "environment_steps": ai.environment_steps})
            print(f"[Final eval {FINAL_EVAL_GAMES} unseen seeds] avg {final_eval['average_score']:.1f} | "
                  f"median {final_eval['median_score']:.1f} | 2048 rate {final_eval['reached_2048_rate']:.0%} | "
                  f"avg max tile {final_eval['average_max_tile']:.0f}")
        save_model_with_meta(ai, LATEST_PATH, episode, max_tile_ever,
                             float(np.mean(recent_scores)) if recent_scores else 0.0,
                             {"recent_average_score": float(np.mean(recent_scores)) if recent_scores else 0.0,
                              "best_evaluation_quality": best_evaluation_quality,
                              "state_encoding": ai.state_encoding, "reward_scale": REWARD_SCALE})
        if pygame.get_init():
            pygame.display.quit()
            pygame.quit()
        print(f"[Saved] Latest checkpoint: {LATEST_PATH}")
        if os.path.exists(BEST_PATH):
            print(f"[Saved] Best frozen-evaluation model: {BEST_PATH}")


if __name__ == "__main__":
    run_evolution()
