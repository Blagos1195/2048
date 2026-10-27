import pygame
import numpy as np
from game import Game2048

BACKGROUND_COLOR = (187, 173, 160)
EMPTY_TILE = (204, 192, 179)
TEXT_COLOR = (119, 110, 101)
PANEL_COLOR = (143, 122, 102)

TILE_COLORS = {
    2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
    16: (245, 149, 99), 32: (245, 121, 77), 64: (245, 93, 55),
    128: (237, 206, 113), 256: (237, 204, 97), 512: (237, 200, 80),
    1024: (237, 197, 63), 2048: (237, 194, 46), 4096: (60, 58, 50)
}

TILE_SURF_CACHE = {}

def init_tile_cache(font):
    TILE_SURF_CACHE.clear()
    for val in TILE_COLORS:
        t_color = TEXT_COLOR if val <= 4 else (255, 255, 255)
        TILE_SURF_CACHE[val] = font.render(str(val), True, t_color)


def get_tile_surface(value, font):
    value = int(value)
    if value not in TILE_SURF_CACHE:
        color = TEXT_COLOR if value <= 4 else (255, 255, 255)
        TILE_SURF_CACHE[value] = font.render(str(value), True, color)
    return TILE_SURF_CACHE[value]

def draw_tiny_board(screen, game, x, y, size):
    margin = 4
    total_margin = margin * 5
    cell_size = (size - total_margin) / 4.0
    pygame.draw.rect(screen, (160, 145, 132), (x, y, size, size), border_radius=4)
    for r in range(4):
        for c in range(4):
            val = game.board[r, c]
            cx = x + margin + c * (cell_size + margin)
            cy = y + margin + r * (cell_size + margin)
            color = TILE_COLORS.get(val, (60, 58, 50)) if val > 0 else EMPTY_TILE
            pygame.draw.rect(screen, color, (cx, cy, cell_size, cell_size), border_radius=3)
            if val > 0:
                txt_surf = get_tile_surface(int(val), pygame.font.SysFont("arial", max(10, int(cell_size * 0.42)), bold=True))
                txt_rect = txt_surf.get_rect(center=(cx + cell_size/2, cy + cell_size/2))
                screen.blit(txt_surf, txt_rect)

def draw_neural_network(screen, ai, state, x, y, width, height):
    pygame.draw.rect(screen, PANEL_COLOR, (x, y, width, height), border_radius=6)
    
    weights = ai.get_weights()
    layer_sizes = weights.get('layer_sizes', [16, 48, 32, 4])
    # Use the model's own encoding so the diagram also works with tile-plane inputs.
    x_in = ai.encode(state).reshape(-1, 1)
    
    activations = [x_in.flatten()]
    a = x_in
    num_weights = len(layer_sizes) - 1
    
    for i in range(num_weights - 1):
        w = weights[f'W{i+1}']
        b = weights[f'b{i+1}']
        z = np.dot(w, a) + b
        a = np.maximum(0, z)
        activations.append(a.flatten())
    
    w_final = weights[f'W{num_weights}']
    b_final = weights[f'b{num_weights}']
    z_final = np.dot(w_final, a) + b_final
    outputs = z_final.flatten()
    activations.append(outputs)

    layers = []
    for idx, size in enumerate(layer_sizes):
        if idx == 0:
            layers.append([f"In {i+1}" for i in range(size)])
        elif idx == len(layer_sizes) - 1:
            layers.append(["LEFT", "DOWN", "RIGHT", "UP"])
        else:
            layers.append([f"H{idx}_{i}" for i in range(size)])

    col_spacing = width / (len(layers) + 1)
    layer_coords = []

    for l_idx, layer in enumerate(layers):
        cx = x + (l_idx + 1) * col_spacing
        node_coords = []
        max_nodes_to_draw = min(len(layer), 22)
        node_spacing = height / (max_nodes_to_draw + 1)

        for n_idx in range(max_nodes_to_draw):
            cy = y + (n_idx + 1) * node_spacing
            node_coords.append((cx, cy))
            
            act_val = activations[l_idx][n_idx] if n_idx < len(activations[l_idx]) else 0
            intensity = int(min(255, max(50, abs(act_val) * 50)))
            n_color = (intensity, 200, 255) if act_val >= 0 else (255, 100, 100)
            
            pygame.draw.circle(screen, n_color, (int(cx), int(cy)), 4)

        layer_coords.append(node_coords)

    for l_idx in range(len(layer_coords) - 1):
        prev_layer = layer_coords[l_idx]
        curr_layer = layer_coords[l_idx + 1]
        for p_pt in prev_layer[::4]:
            for c_pt in curr_layer[::2]:
                pygame.draw.line(screen, (130, 115, 102), p_pt, c_pt, 1)

    font = pygame.font.SysFont("arial", 12, bold=True)
    arch_str = " - ".join(map(str, layer_sizes))
    title_surf = font.render(f"DQN Q-VALUES: [{arch_str}]", True, (255, 255, 255))
    screen.blit(title_surf, (x + 15, y + 10))
    q_values = ai.forward(state)
    q_line = "  ".join(f"{name}: {q_values[i]:+.2f}" for i, name in enumerate(("LEFT", "DOWN", "RIGHT", "UP")))
    screen.blit(font.render(q_line, True, (255, 255, 255)), (x + 15, y + 28))

def draw_showcase_screen(screen, font, small_font, board, generation, max_tile_ever, live_gps, avg_gps, ai, state, status_text=""):
    width = screen.get_width()
    height = screen.get_height()
    screen.fill(BACKGROUND_COLOR)
    
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, width, 60))
    gen_txt = small_font.render(f"Gen: {generation}", True, (255, 255, 255))
    max_txt = small_font.render(f"Max: {max_tile_ever}", True, (237, 194, 46))
    gps_txt = small_font.render(f"Live: {live_gps:.1f} | Avg: {avg_gps:.1f} GPS", True, (255, 255, 255))
    
    screen.blit(gen_txt, (20, 20))
    screen.blit(max_txt, (150, 20))
    screen.blit(gps_txt, (300, 20))

    if status_text:
        s_txt = small_font.render(status_text, True, (255, 255, 0))
        screen.blit(s_txt, (580, 20))

    board_area_w = width * 0.55
    margin = 12
    board_size = min(board_area_w - 60, height - 140)
    start_x = 30 + (board_area_w - 60 - board_size) / 2
    start_y = 80 + (height - 80 - board_size) / 2

    pygame.draw.rect(screen, PANEL_COLOR, (start_x, start_y, board_size, board_size), border_radius=6)
    cell_size = (board_size - (margin * 5)) / 4.0

    for r in range(4):
        for c in range(4):
            val = board[r, c]
            x = start_x + margin + c * (cell_size + margin)
            y = start_y + margin + r * (cell_size + margin)
            color = TILE_COLORS.get(val, (60, 58, 50)) if val > 0 else EMPTY_TILE
            pygame.draw.rect(screen, color, (x, y, cell_size, cell_size), border_radius=4)
            if val > 0:
                txt_surf = get_tile_surface(int(val), font)
                screen.blit(txt_surf, txt_surf.get_rect(center=(x + cell_size/2, y + cell_size/2)))

    brain_x = board_area_w + 10
    brain_y = 80
    brain_w = width - board_area_w - 30
    brain_h = height - 100
    draw_neural_network(screen, ai, state, brain_x, brain_y, brain_w, brain_h)
    pygame.display.flip()

def draw_single_board_simple(screen, font, small_font, board, generation, max_tile_ever, live_gps, avg_gps, status_text=""):
    width = screen.get_width()
    height = screen.get_height()
    screen.fill(BACKGROUND_COLOR)
    
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, width, 60))
    gen_txt = small_font.render(f"Gen: {generation}", True, (255, 255, 255))
    max_txt = small_font.render(f"Max: {max_tile_ever}", True, (237, 194, 46))
    gps_txt = small_font.render(f"Live: {live_gps:.1f} | Avg: {avg_gps:.1f} GPS", True, (255, 255, 255))
    
    screen.blit(gen_txt, (20, 20))
    screen.blit(max_txt, (150, 20))
    screen.blit(gps_txt, (300, 20))

    if status_text:
        s_txt = small_font.render(status_text, True, (255, 255, 0))
        screen.blit(s_txt, (650, 20))

    margin = 12
    board_size = min(width - 80, height - 140)
    start_x = (width - board_size) / 2
    start_y = 80 + (height - 80 - board_size) / 2

    pygame.draw.rect(screen, PANEL_COLOR, (start_x, start_y, board_size, board_size), border_radius=6)
    cell_size = (board_size - (margin * 5)) / 4.0

    for r in range(4):
        for c in range(4):
            val = board[r, c]
            x = start_x + margin + c * (cell_size + margin)
            y = start_y + margin + r * (cell_size + margin)
            color = TILE_COLORS.get(val, (60, 58, 50)) if val > 0 else EMPTY_TILE
            pygame.draw.rect(screen, color, (x, y, cell_size, cell_size), border_radius=4)
            if val > 0:
                txt_surf = get_tile_surface(int(val), font)
                screen.blit(txt_surf, txt_surf.get_rect(center=(x + cell_size/2, y + cell_size/2)))
    pygame.display.flip()

def play_model_showcase(ai, generation, max_tile_ever, live_gps, avg_gps):
    if ai is None:
        print("[Showcase] No active model available yet.")
        return

    pygame.init()
    screen = pygame.display.set_mode((1200, 720), pygame.RESIZABLE)
    pygame.display.set_caption("2048 AI - Live Showcase & Scrying Glass")
    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 14, bold=True)
    init_tile_cache(font)
    clock = pygame.time.Clock()

    game = Game2048()
    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                pygame.quit()
                return

        state = game.get_state()
        q = ai.forward(state)
        for act in np.argsort(q)[::-1]:
            m, _ = game.move(act)
            if m:
                break
        else:
            game.game_over = True
            break

        draw_showcase_screen(screen, font, small_font, game.board, generation, max_tile_ever, live_gps, avg_gps, ai, state, "[LIVE SHOWCASE - Press Space to Close]")
        clock.tick(8)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                waiting = False

        draw_showcase_screen(screen, font, small_font, game.board, generation, max_tile_ever, live_gps, avg_gps, ai, game.get_state(), "[GAME OVER - Press Any Key to Close]")
        clock.tick(10)
    pygame.quit()

def play_replay_showcase(board_history, generation, max_tile_ever, live_gps, avg_gps):
    if board_history is None or len(board_history) == 0:
        print("[Showcase] No recorded game replay in RAM yet.")
        return

    pygame.init()
    screen = pygame.display.set_mode((1000, 680), pygame.RESIZABLE)
    pygame.display.set_caption("2048 AI - Session Best Game Replay")
    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 14, bold=True)
    init_tile_cache(font)
    clock = pygame.time.Clock()

    idx = 0
    speed = 15
    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    idx = min(len(board_history) - 1, idx + 1)
                    paused = True
                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 1)
                    paused = True
                elif event.key == pygame.K_UP:
                    speed = min(60, speed + 5)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 5)

        status = (f"REPLAY {idx + 1}/{len(board_history)} | "
                  f"{'PAUSED' if paused else 'PLAYING'} {speed} FPS | "
                  "Space pause | ←/→ step | ↑/↓ speed | Esc close")
        draw_single_board_simple(screen, font, small_font, board_history[idx],
                                 generation, max_tile_ever, live_gps, avg_gps, status)
        if not paused:
            if idx < len(board_history) - 1:
                idx += 1
            else:
                paused = True
        clock.tick(speed)
    pygame.quit()


def play_larp_mode(ai):
    if ai is None:
        print("[LARP Mode] No model found to pilot!")
        return

    pygame.init()
    screen = pygame.display.set_mode((800, 700), pygame.RESIZABLE)
    pygame.display.set_caption("2048 - Master Terminal (Totally You Playing)")
    font = pygame.font.SysFont("arial", 28, bold=True)
    small_font = pygame.font.SysFont("arial", 16, bold=True)
    init_tile_cache(font)
    clock = pygame.time.Clock()

    game = Game2048()
    status_msg = "SMASH ANY KEY TO CAST MOVES"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif not game.game_over:
                    state = game.get_state()
                    q = ai.forward(state)
                    moved = False
                    for act in np.argsort(q)[::-1]:
                        m, _ = game.move(act)
                        if m:
                            moved = True
                            break
                    if not moved:
                        game.game_over = True
                        status_msg = "GAME OVER - LEGENDARY RUN"

        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, screen.get_width(), 60))
        
        score_txt = small_font.render(f"Score: {game.score}", True, (255, 255, 255))
        mode_txt = small_font.render(status_msg, True, (237, 194, 46))
        screen.blit(score_txt, (30, 20))
        screen.blit(mode_txt, (300, 20))

        margin = 12
        board_size = min(screen.get_width() - 80, screen.get_height() - 140)
        start_x = (screen.get_width() - board_size) / 2
        start_y = 80 + (screen.get_height() - 80 - board_size) / 2

        pygame.draw.rect(screen, PANEL_COLOR, (start_x, start_y, board_size, board_size), border_radius=6)
        cell_size = (board_size - (margin * 5)) / 4.0

        for r in range(4):
            for c in range(4):
                val = game.board[r, c]
                x = start_x + margin + c * (cell_size + margin)
                y = start_y + margin + r * (cell_size + margin)
                color = TILE_COLORS.get(val, (60, 58, 50)) if val > 0 else EMPTY_TILE
                pygame.draw.rect(screen, color, (x, y, cell_size, cell_size), border_radius=4)
                if val > 0:
                    txt_surf = get_tile_surface(int(val), font)
                    screen.blit(txt_surf, txt_surf.get_rect(center=(x + cell_size/2, y + cell_size/2)))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()