"""Tic-Tac-Toe environment for reinforcement learning."""

import numpy as np
from typing import Tuple, List


class TicTacToeEnv:
    """Tic-Tac-Toe game environment.
    
    The board is represented as a 1D array of 9 elements:
    - 1: Human player (X)
    - -1: AI player (O)
    - 0: Empty cell
    
    Actions are integers 0-8 representing board positions:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    """

    def __init__(self):
        """Initialize the environment."""
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = 1  # 1 for human/agent1, -1 for agent2
        self.terminal = False
        self.reward = 0.0
        self._winning_lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

    def reset(self):
        """Reset the environment."""
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = 1
        self.terminal = False
        self.reward = 0.0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get current board state."""
        return self.board.copy()

    def get_valid_moves(self) -> List[int]:
        """Get list of valid moves."""
        return list(np.where(self.board == 0)[0])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return state, reward, done."""
        if action not in self.get_valid_moves():
            raise ValueError(f"Invalid action: {action}")

        # Place piece
        self.board[action] = self.current_player

        # Check terminal state
        winner = self._check_winner()
        if winner is not None:
            self.terminal = True
            if winner == self.current_player:
                self.reward = 1.0
            else:
                self.reward = -1.0
        elif len(self.get_valid_moves()) == 0:
            self.terminal = True
            self.reward = 0.0

        # Switch player
        self.current_player *= -1

        return self.get_state(), self.reward, self.terminal

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.terminal

    def _check_winner(self) -> int | None:
        """Check if there's a winner. Returns 1, -1, or None."""
        for line in self._winning_lines:
            values = self.board[line]
            if values[0] != 0 and values[0] == values[1] == values[2]:
                return values[0]
        return None

    def render(self):
        """Render the board."""
        symbols = {0: " ", 1: "X", -1: "O"}
        board = self.board.reshape(3, 3)
        print("\n   0   1   2")
        for i in range(3):
            print(f"{i}  {symbols[board[i, 0]]} | {symbols[board[i, 1]]} | {symbols[board[i, 2]]}")
            if i < 2:
                print("  -----------")
        print()
