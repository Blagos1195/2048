"""Pygame views for the self-learning n-tuple 2048 agent.

Modes:
  1: train while viewing four parallel games
  2: same, slower animation
  3: watch a saved model play
  4: LARP mode: press any key to let the AI make one move
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pygame

from selfplay_ai import ACTIONS, ExpectimaxAgent, Game2048, NTupleValue, can_move

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
    screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
    pygame.display.set_caption("2048 Self-learning AI | Training view")
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

        for _ in range(speed):
            for i, game in enumerate(games):
                if episode >= games_target:
                    break
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
    evaluator = load_evaluator(model)
    agent = ExpectimaxAgent(evaluator, depth=depth, seed=seed)
    pygame.init()
    screen = pygame.display.set_mode((850, 790), pygame.RESIZABLE)
    pygame.display.set_caption("2048 Self-learning AI" + (" | LARP mode" if mode == 4 else " | Showcase"))
    font = pygame.font.SysFont("arial", 22, bold=True)
    small = pygame.font.SysFont("arial", 16)
    clock = pygame.time.Clock()
    game = Game2048(seed)
    running = True
    last_action = "Waiting for first move"
    auto = mode == 3

    while running:
        should_move = auto
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif mode == 4 and not should_move:
                    should_move = True
                elif event.key == pygame.K_r:
                    game = Game2048(seed + int(time.time()) % 100000)
                    last_action = "New game"
        if should_move and can_move(game.board):
            action = agent.choose_action(game.board.copy())
            moved, _, _ = game.step(action)
            last_action = ACTION_NAMES[action] if moved else "No legal move"
        screen.fill(BG)
        screen.blit(font.render("2048 | " + ("LARP MODE" if mode == 4 else "AI SHOWCASE"), True, TEXT), (22, 18))
        screen.blit(small.render(f"Score: {game.score:,}    Max tile: {game.max_tile}    Last move: {last_action}", True, TEXT), (22, 52))
        width, height = screen.get_size()
        size = min(width - 44, height - 150)
        draw_board(screen, game.board, ((width - size) / 2, 90, size, size), font)
        hint = "Press any key to let the AI make one move | R: restart | Esc: quit" if mode == 4 else "AI playing automatically | R: restart | Esc: quit"
        screen.blit(small.render(hint, True, TEXT), (22, height - 35))
        if not can_move(game.board):
            screen.blit(font.render("GAME OVER", True, (160, 70, 55)), (width - 180, 18))
            auto = False
        pygame.display.flip()
        clock.tick(8 if mode == 3 else 30)
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
    main()
