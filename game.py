"""Neutral 2048 rules and reproducible random-number support."""
import random
import numpy as np


class Game2048:
    # Action indices are deliberately consistent throughout the project.
    ACTION_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")

    def __init__(self, seed=None, rng=None):
        self.rng = rng if rng is not None else random.Random(seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.game_over = False
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = self.rng.choice(empty_cells)
            self.board[r, c] = 4 if self.rng.random() < 0.1 else 2

    @staticmethod
    def compress(row):
        values = row[row != 0]
        return np.pad(values, (0, 4 - len(values)), 'constant')

    @staticmethod
    def merge(row):
        row = row.copy()
        score_gained = 0
        i = 0
        while i < 3:
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                score_gained += int(row[i])
                row[i + 1] = 0
                i += 2
            else:
                i += 1
        return row, score_gained

    def slide_left(self, board):
        result = np.zeros((4, 4), dtype=np.int64)
        score_gained = 0
        for i in range(4):
            merged, score = self.merge(self.compress(board[i]))
            result[i] = self.compress(merged)
            score_gained += score
        return result, score_gained

    def board_after_move(self, direction):
        if direction not in range(4):
            raise ValueError("direction must be 0=left, 1=down, 2=right, 3=up")
        # 0=left, 1=down, 2=right, 3=up. Rotating clockwise makes
        # a downward move become a leftward move in the rotated board.
        rotated = np.rot90(self.board, -direction)
        shifted, score = self.slide_left(rotated)
        return np.rot90(shifted, direction).copy(), score

    def valid_actions(self):
        return [action for action in range(4)
                if not np.array_equal(self.board_after_move(action)[0], self.board)]

    def move(self, direction):
        if self.game_over:
            return False, 0
        new_board, score_gained = self.board_after_move(direction)
        moved = not np.array_equal(self.board, new_board)
        if moved:
            self.board = new_board
            self.score += score_gained
            self.spawn_tile()
            if not self.can_move():
                self.game_over = True
        return moved, score_gained

    def can_move(self):
        if np.any(self.board == 0):
            return True
        return bool(np.any(self.board[:, :-1] == self.board[:, 1:]) or
                    np.any(self.board[:-1, :] == self.board[1:, :]))

    def get_state(self):
        return self.board.copy()
