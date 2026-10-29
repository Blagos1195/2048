"""Pygame views for the self-learning n-tuple 2048 agent.

Modes:
  1: train while viewing four parallel games
  2: same, slower animation
  3: watch a saved model play
  4: LARP mode: press any key to let the AI make one move
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pygame

from selfplay_ai import ACTIONS, ExpectimaxAgent, Game2048, NTupleValue, can_move, legal_actions

BG = (250, 248, 239)
PANEL = (187, 173, 160)
EMPTY = (205, 193, 180)
TEXT = (119, 110, 101)
TILE_COLORS = {
    2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
    16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59),
    128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
    1024: (237, 197, 63), 2048: (237, 194, 46),
}
ACTION_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")


def draw_board(screen, board, rect, font):
    x, y, w, h = rect
    pygame.draw.rect(screen, PANEL, rect, border_radius=8)
    gap = 7
    cell = (min(w, h) - gap * 5) / 4
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            cx = x + gap + c * (cell + gap)
            cy = y + gap + r * (cell + gap)
            color = EMPTY if value == 0 else TILE_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(screen, color, (cx, cy, cell, cell), border_radius=5)
            if value:
                ink = TEXT if value <= 4 else (255, 255, 255)
                size = max(12, int(cell * (0.37 if value >= 1024 else 0.44)))
                tile_font = pygame.font.SysFont("arial", size, bold=True)
                label = tile_font.render(str(value), True, ink)
                screen.blit(label, label.get_rect(center=(cx + cell / 2, cy + cell / 2)))


def load_evaluator(path: str | None) -> NTupleValue:
    evaluator = NTupleValue()
    candidates = [Path(path)] if path else [
        Path("models/selfplay/best.npz"),
        Path("models/selfplay/latest.npz"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            evaluator.load(str(candidate))
            print(f"[View] Loaded learned value function: {candidate}")
            return evaluator
    raise FileNotFoundError(
        "No self-learning checkpoint found. Train first with "
        "'python main.py --mode 0 --games 100000', or pass --model PATH."
    )


def train_view(mode: int, games_target: int, depth: int, out: str, seed: int):
    pygame.init()
    screen = pygame.display.set_mode((900, 650), pygame.RESIZABLE)
    pygame.display.set_caption("2048 | Training view")
    font = pygame.font.SysFont("arial", 20, bold=True)
    small = pygame.font.SysFont("arial", 15)
    clock = pygame.time.Clock()

    evaluator = NTupleValue(seed=seed)
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed, epsilon=0.08)
    games = [Game2048(seed + i) for i in range(4)]
    episode_scores = []
    episode = 0
    running = True
    last_save = time.monotonic()
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    speed = 1 if mode == 2 else 4
    print("[Training view] Esc closes the view; checkpoints save periodically.")

    while running and episode < games_target:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(640, event.w), max(480, event.h)), pygame.RESIZABLE)

        for _ in range(speed):
            for i, game in enumerate(games):
                if episode >= games_target:
                    break
                agent.epsilon = max(0.02, 0.08 * (0.9999 ** episode))
                if not can_move(game.board):
                    episode += 1
                    episode_scores.append(game.score)
                    if episode % 100 == 0:
                        avg = float(np.mean(episode_scores[-100:]))
                        print(f"episode {episode}/{games_target} | recent average score {avg:.1f} | "
                              f"max tile {max(g.max_tile for g in games)} | TD updates {evaluator.updates:,}")
                    games[i] = Game2048(seed + episode * 7 + i)
                    continue
                state = game.board.copy()
                action = agent.choose_action(state)
                moved, reward, terminal = game.step(action)
                if moved:
                    evaluator.update(state, reward / 128.0, game.board.copy(), terminal)
                elif not can_move(game.board):
                    episode += 1
                    episode_scores.append(game.score)
                    games[i] = Game2048(seed + episode * 7 + i)

        now = time.monotonic()
        if now - last_save >= 20:
            evaluator.save(out_path / "latest.npz", {"episodes": episode, "updates": evaluator.updates})
            last_save = now

        screen.fill(BG)
        header = font.render(
            f"SELF-LEARNING TRAINING | Episodes {episode:,}/{games_target:,} | TD updates {evaluator.updates:,}",
            True, TEXT)
        screen.blit(header, (20, 15))
        width, height = screen.get_size()
        margin, top = 20, 55
        board_w = (width - margin * 3) // 2
        board_h = min(board_w, (height - top - 60 - margin * 3) // 2)
        for i, game in enumerate(games):
            x = margin + (i % 2) * (board_w + margin)
            y = top + (i // 2) * (board_h + margin + 25)
            label = small.render(f"Game {i+1} | Score {game.score:,} | Max tile {game.max_tile}", True, TEXT)
            screen.blit(label, (x, y))
            draw_board(screen, game.board, (x, y + 22, board_w, board_h), font)
        footer = small.render("Esc: close | Models save to " + str(out_path), True, TEXT)
        screen.blit(footer, (20, height - 28))
        pygame.display.flip()
        clock.tick(30 if mode == 1 else 12)

    evaluator.save(out_path / "latest.npz", {"episodes": episode, "updates": evaluator.updates})
    pygame.quit()


def showcase(mode: int, model: str | None, depth: int, seed: int):
    """Play a conventional-looking 2048 game with a precomputed AI move in LARP mode."""
    import threading

    evaluator = load_evaluator(model)
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed)
    pygame.init()
    screen = pygame.display.set_mode((520, 700), pygame.RESIZABLE)
    pygame.display.set_caption("2048")
    clock = pygame.time.Clock()
    game = Game2048(seed)
    auto = mode == 3
    game_over = False
    running = True
    best_score = 0
    last_action = "Ready"
    stats_visible = False
    stats_path = Path("player_stats.json")
    player_stats = {"games_played": 0, "total_score": 0, "highest_score": 0, "highest_tile": 2}
    try:
        if stats_path.exists():
            saved_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            for key in player_stats:
                if isinstance(saved_stats.get(key), (int, float)) and saved_stats[key] >= 0:
                    player_stats[key] = saved_stats[key]
    except (OSError, ValueError, TypeError):
        print("[View] Could not read player_stats.json; starting with fresh stats.")

    current_game_recorded = False
    move_plan = {"board": None, "action": None, "busy": False}
    plan_lock = threading.Lock()
    # A single worker processes only the latest requested board. This prevents
    # spam-clicking/restarting from launching concurrent searches on one agent.
    worker_state = {
        "active": False,
        "version": 0,
        "requested_board": None,
        "requested_key": None,
    }
    last_move_time = 0.0
    move_debounce_seconds = 0.08

    title_font = pygame.font.SysFont("arial", 30, bold=True)
    score_font = pygame.font.SysFont("arial", 17, bold=True)
    small = pygame.font.SysFont("arial", 15)
    tiny = pygame.font.SysFont("arial", 13)
    button_font = pygame.font.SysFont("arial", 16, bold=True)

    def schedule_plan():
        """Request a move without starting duplicate search threads."""
        if game_over or not can_move(game.board):
            cancel_plan()
            return
        board = game.board.copy()
        board_key = board.tobytes()
        with plan_lock:
            worker_state["version"] += 1
            worker_state["requested_board"] = board
            worker_state["requested_key"] = board_key
            move_plan["board"] = None
            move_plan["action"] = None
            move_plan["busy"] = True
            if worker_state["active"]:
                return
            worker_state["active"] = True

        def worker_loop():
            while True:
                with plan_lock:
                    requested_board = worker_state["requested_board"]
                    requested_key = worker_state["requested_key"]
                    version = worker_state["version"]
                    if requested_board is None:
                        worker_state["active"] = False
                        return
                    # Consume this request. Newer requests replace it while thinking.
                    worker_state["requested_board"] = None
                    worker_state["requested_key"] = None

                try:
                    action = agent.choose_action(requested_board)
                except Exception as exc:
                    # Keep an AI/search failure from killing the UI event loop.
                    print(f"[View] AI move calculation failed: {exc}")
                    action = None

                with plan_lock:
                    if (version == worker_state["version"]
                            and requested_key == game.board.tobytes()
                            and not game_over):
                        legal = legal_actions_safe(game.board)
                        if action in legal:
                            move_plan["board"] = requested_key
                            move_plan["action"] = action
                            move_plan["busy"] = False
                        else:
                            move_plan["board"] = None
                            move_plan["action"] = None
                            move_plan["busy"] = False
                    # If another board was requested during this search, loop and
                    # calculate only the newest request. Otherwise release worker.
                    if worker_state["requested_board"] is None:
                        worker_state["active"] = False
                        return

        threading.Thread(target=worker_loop, daemon=True, name="2048-ai-planner").start()

    def cancel_plan():
        """Invalidate any result for an old board and stop pending requests."""
        with plan_lock:
            worker_state["version"] += 1
            worker_state["requested_board"] = None
            worker_state["requested_key"] = None
            move_plan["board"] = None
            move_plan["action"] = None
            move_plan["busy"] = False

    def legal_actions_safe(board):
        # Only accept a known legal action for the exact current board.
        return legal_actions(board)

    def record_game():
        """Save this game once as the player's achievement."""
        nonlocal current_game_recorded
        if current_game_recorded or game.moves == 0:
            return
        player_stats["games_played"] += 1
        player_stats["total_score"] += int(game.score)
        player_stats["highest_score"] = max(int(player_stats["highest_score"]), int(game.score))
        player_stats["highest_tile"] = max(int(player_stats["highest_tile"]), int(game.max_tile))
        current_game_recorded = True
        try:
            stats_path.write_text(json.dumps(player_stats, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"[View] Could not save player stats: {exc}")

    def restart():
        nonlocal game, game_over, last_action, current_game_recorded
        record_game()
        cancel_plan()
        game = Game2048(seed + int(time.time() * 1000) % 1000000)
        game_over = False
        current_game_recorded = False
        last_action = "Ready"
        schedule_plan()

    def draw_button(rect, label, enabled=True):
        color = (119, 110, 101) if enabled else (180, 174, 166)
        pygame.draw.rect(screen, color, rect, border_radius=5)
        text = button_font.render(label, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=rect.center))

    schedule_plan()
    while running:
        width, height = screen.get_size()
        margin = max(12, min(22, width // 24))
        header_h = 92
        footer_h = 48
        button_h = 38
        gap = 10
        board_size = max(180, min(width - margin * 2, height - header_h - footer_h - button_h - gap * 3))
        board_x = (width - board_size) // 2
        board_y = header_h + 12
        new_width = (width - margin * 2 - gap) // 2
        new_rect = pygame.Rect(margin, height - footer_h - button_h - 4, new_width, button_h)
        info_rect = pygame.Rect(new_rect.right + gap, new_rect.y, (width - margin * 2 - gap) // 2, button_h)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(360, event.w), max(480, event.h)), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if stats_visible and mode == 4:
                    panel_w = min(width - 32, 320)
                    panel_h = 230
                    stats_panel = pygame.Rect((width - panel_w) // 2, (height - panel_h) // 2, panel_w, panel_h)
                    close_rect = pygame.Rect(stats_panel.right - 34, stats_panel.y + 8, 26, 26)
                    if close_rect.collidepoint(event.pos) or not stats_panel.collidepoint(event.pos):
                        stats_visible = False
                    continue
                if new_rect.collidepoint(event.pos):
                    record_game()
                    best_score = max(best_score, game.score)
                    cancel_plan()
                    game = Game2048(seed + int(time.time() * 1000) % 1000000)
                    game_over = False
                    current_game_recorded = False
                    last_action = "Ready"
                    stats_visible = False
                    schedule_plan()
                elif mode == 3 and info_rect.collidepoint(event.pos):
                    auto = not auto
                elif mode == 4 and info_rect.collidepoint(event.pos):
                    stats_visible = not stats_visible
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    record_game()
                    best_score = max(best_score, game.score)
                    cancel_plan()
                    game = Game2048(seed + int(time.time() * 1000) % 1000000)
                    game_over = False
                    current_game_recorded = False
                    last_action = "Ready"
                    stats_visible = False
                    schedule_plan()
                elif stats_visible and mode == 4:
                    # Don't let keys intended to close/interact with the overlay
                    # accidentally advance the hidden game.
                    continue
                elif not game_over:
                    if mode == 3:
                        auto = not auto
                    else:
                        # LARP: any ordinary key advances one AI-selected move.
                        with plan_lock:
                            planned_board = move_plan["board"]
                            planned_action = move_plan["action"]
                        # Ignore key repeats while the planner is busy or the
                        # board has changed. Never run a second search on the UI thread.
                        now = time.monotonic()
                        if (not game_over and planned_board == game.board.tobytes()
                                and planned_action in legal_actions_safe(game.board)
                                and now - last_move_time >= move_debounce_seconds):
                            last_move_time = now
                            moved, _, terminal = game.step(planned_action)
                            if moved:
                                last_action = ACTION_NAMES[planned_action]
                                best_score = max(best_score, game.score)
                                game_over = terminal or not can_move(game.board)
                                if game_over:
                                    cancel_plan()
                                    record_game()
                                else:
                                    schedule_plan()

        if auto and not game_over:
            with plan_lock:
                planned_board = move_plan["board"]
                planned_action = move_plan["action"]
            if planned_board == game.board.tobytes() and planned_action in legal_actions_safe(game.board):
                moved, _, terminal = game.step(planned_action)
                if moved:
                    last_action = ACTION_NAMES[planned_action]
                    best_score = max(best_score, game.score)
                    game_over = terminal or not can_move(game.board)
                    if game_over:
                        cancel_plan()
                        record_game()
                    else:
                        schedule_plan()

        screen.fill(BG)
        # Familiar 2048 header and compact score cards.
        screen.blit(title_font.render("2048", True, TEXT), (margin, 14))
        score_w = max(82, (width - margin * 2 - 8) // 2)
        score_x = width - margin - score_w * 2 - 8
        for i, (label, value) in enumerate((("SCORE", game.score), ("BEST", max(int(player_stats["highest_score"]), best_score, game.score)))):
            rect = pygame.Rect(score_x + i * (score_w + 8), 12, score_w, 54)
            pygame.draw.rect(screen, PANEL, rect, border_radius=5)
            screen.blit(tiny.render(label, True, (238, 228, 218)),
                        tiny.render(label, True, (238, 228, 218)).get_rect(center=(rect.centerx, rect.y + 13)))
            value_text = score_font.render(f"{value:,}", True, (255, 255, 255))
            screen.blit(value_text, value_text.get_rect(center=(rect.centerx, rect.y + 36)))

        width, height = screen.get_size()
        board_size = max(180, min(width - margin * 2, height - header_h - footer_h - button_h - gap * 3))
        board_x = (width - board_size) // 2
        board_y = max(header_h, (height - board_size - footer_h - button_h - gap) // 2)
        draw_board(screen, game.board, (board_x, board_y, board_size, board_size), title_font)

        new_width = (width - margin * 2 - gap) // 2
        new_rect = pygame.Rect(margin, height - footer_h - button_h - 4, new_width, button_h)
        info_rect = pygame.Rect(new_rect.right + gap, new_rect.y, (width - margin * 2 - gap) // 2, button_h)
        draw_button(new_rect, "New Game")
        if mode == 3:
            draw_button(info_rect, "Pause AI" if auto else "Resume AI")
        else:
            draw_button(info_rect, "Your Stats")
        footer_text = ("AI thinking..." if move_plan["busy"] else "AI ready") if not game_over else "Game over"
        if mode == 3:
            hint_surface = tiny.render(footer_text, True, TEXT)
            screen.blit(hint_surface, hint_surface.get_rect(center=(width // 2, height - 12)))

        if stats_visible and mode == 4:
            panel_w = min(width - 32, 320)
            panel_h = 230
            panel = pygame.Rect((width - panel_w) // 2, (height - panel_h) // 2, panel_w, panel_h)
            pygame.draw.rect(screen, (250, 248, 239), panel, border_radius=10)
            pygame.draw.rect(screen, PANEL, panel, width=2, border_radius=10)
            close_rect = pygame.Rect(panel.right - 34, panel.y + 8, 26, 26)
            pygame.draw.rect(screen, (238, 228, 218), close_rect, border_radius=5)
            close_font = pygame.font.SysFont("arial", 20, bold=True)
            close_label = close_font.render("x", True, TEXT)
            screen.blit(close_label, close_label.get_rect(center=close_rect.center))
            heading = score_font.render("2048 STATS", True, TEXT)
            screen.blit(heading, heading.get_rect(center=(panel.centerx, panel.y + 26)))
            average = (player_stats["total_score"] / player_stats["games_played"]
                       if player_stats["games_played"] else 0)
            lines = [
                f"Games played: {int(player_stats['games_played']):,}",
                f"Highest score: {int(player_stats['highest_score']):,}",
                f"Average score: {average:,.1f}",
                f"Highest tile: {int(player_stats['highest_tile']):,}",
                "",
                "Stats are saved between sessions",
            ]
            for i, line in enumerate(lines):
                surf = tiny.render(line, True, TEXT)
                screen.blit(surf, surf.get_rect(center=(panel.centerx, panel.y + 58 + i * 27)))

        if game_over:
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((40, 35, 30, 155))
            screen.blit(overlay, (0, 0))
            panel_w = min(width - 32, 330)
            panel_h = min(height - 40, 300)
            panel = pygame.Rect((width - panel_w) // 2, (height - panel_h) // 2, panel_w, panel_h)
            pygame.draw.rect(screen, (250, 248, 239), panel, border_radius=10)
            heading = title_font.render("Game over!", True, TEXT)
            screen.blit(heading, heading.get_rect(center=(panel.centerx, panel.y + 38)))
            lines = [
                f"Final score: {game.score:,}",
                f"Best score: {max(best_score, game.score):,}",
                f"Largest tile: {game.max_tile:,}",
                f"Moves played: {game.moves:,}",
            ]
            for i, line in enumerate(lines):
                surf = small.render(line, True, TEXT)
                screen.blit(surf, surf.get_rect(center=(panel.centerx, panel.y + 88 + i * 27)))
            hint = small.render("Choose New Game to play again", True, TEXT)
            screen.blit(hint, hint.get_rect(center=(panel.centerx, panel.bottom - 24)))

        pygame.display.flip()
        clock.tick(30 if mode == 3 else 60)

    record_game()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Self-learning 2048 graphical modes")
    parser.add_argument("--mode", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--model", default=None, help="Checkpoint .npz; defaults to best then latest")
    parser.add_argument("--games", type=int, default=100000, help="Training episodes for modes 1 and 2")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--out", default="models/selfplay")
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.mode in (1, 2):
        train_view(args.mode, args.games, args.depth, args.out, args.seed)
    else:
        showcase(args.mode, args.model, args.depth, args.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[View] Keyboard interrupt received. Closing cleanly...")
    finally:
        if pygame.get_init():
            pygame.quit()
