from typing import Tuple
import numpy as np
import random
from abc import ABC, abstractmethod

PLAYER1 = 1
PLAYER2 = 2
WIN = 1
TIE = 0.2
ONGOING = -1


class Player(ABC):
    @abstractmethod
    def play(self, board: np.ndarray) -> int:
        pass


class RandomPlayer(Player):
    def play(self, board):
        return random.choices([i for i, _ in enumerate(board) if board[i] == 0])[0]


class GameBoard:
    def __init__(self):
        self.reset_board()

    def get_board(self) -> "GameBoard":
        return self.board

    def set_cell(self, index: int, value: int) -> None:
        self.board[index] = value

    def get_cell(self, index: int) -> int:
        return self.board[index]

    def is_action_valid(self, action: int) -> bool:
        return action in [i for i, _ in enumerate(self.board) if self.board[i] == 0]

    def reset_board(self) -> np.ndarray:
        self.board = np.zeros(shape=(9,), dtype=np.int32)


class GameState:
    def __init__(self, opponent: Player, player: int) -> None:
        self.board = GameBoard()
        self.player = player
        self.opponent = opponent
        self.reset_state(toggle=False)

    def reset_state(self, toggle: bool = True) -> None:
        self.turn = 0
        self.board.reset_board()
        if self.player == PLAYER2:
            self.make_opponent_move()

    def get_opponent(self) -> int:
        return PLAYER1 if self.player == PLAYER2 else PLAYER2

    def get_board_state(self) -> np.ndarray:
        return self.board.get_board()

    def evaluate_status(self) -> float:
        if self.turn > 8:
            return TIE

        for i in range(3):
            if i == 0 or i == 2:
                # Diagonals
                if (
                    self.board.get_cell(i) == self.board.get_cell(4)
                    and self.board.get_cell(8 - i) == self.board.get_cell(i)
                    and self.board.get_cell(i) != 0
                ):
                    return WIN
            # Columns
            if (
                self.board.get_cell(i) == self.board.get_cell(i + 3)
                and self.board.get_cell(i + 6) == self.board.get_cell(i)
                and self.board.get_cell(i) != 0
            ):
                return WIN
            # Rows
            row = i * 3
            if (
                self.board.get_cell(row) == self.board.get_cell(row + 1)
                and self.board.get_cell(row + 2) == self.board.get_cell(row)
                and self.board.get_cell(row) != 0
            ):
                return WIN

        return ONGOING

    def make_move(self, action) -> Tuple[np.ndarray, float, bool]:
        if self.board.is_action_valid(action):
            self.board.set_cell(action, self.player)
            self.turn += 1
            reward = self.evaluate_status()
            if reward != ONGOING:
                return self.get_board_state(), reward, True

        self.make_opponent_move()
        reward = self.evaluate_status()
        if reward != ONGOING:
            return self.get_board_state(), -reward if reward == WIN else reward, True

        return self.get_board_state(), 0, False

    def make_opponent_move(self) -> None:
        action = self.opponent.play(self.get_board_state())
        if self.board.is_action_valid(action):
            self.board.set_cell(action, self.get_opponent())
            self.turn += 1
