"""RL agents for Tic-Tac-Toe."""

import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for RL agents."""

    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action given state."""
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update agent based on transition."""
        pass


class QLearningAgent(BaseAgent):
    """Q-Learning agent for Tic-Tac-Toe."""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.99):
        """Initialize Q-Learning agent.
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[tuple, np.ndarray] = {}

    def _state_key(self, state: np.ndarray) -> tuple:
        """Convert state to hashable key."""
        return tuple(state)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy."""
        state_key = self._state_key(state)

        # Get valid moves
        valid_moves = np.where(state == 0)[0]

        if np.random.random() < epsilon:
            # Explore
            return np.random.choice(valid_moves)
        else:
            # Exploit
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(9)

            q_values = self.q_table[state_key].copy()
            # Set invalid moves to very low value
            # q_values[state != 0] = -np.inf
            return np.argmax(q_values)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Update Q-values using Q-learning update rule."""
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(9)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            # Get max Q-value for next state (valid moves only)
            valid_moves = np.where(next_state == 0)[0]
            if len(valid_moves) > 0:
                next_q_values = self.q_table[next_state_key].copy()
                next_q_values[next_state != 0] = -np.inf
                max_next_q = np.max(next_q_values)
            else:
                max_next_q = 0
            target_q = reward + self.discount_factor * max_next_q

        self.q_table[state_key][action] = current_q + self.learning_rate * (
            target_q - current_q
        )


class NeuralNetworkAgent(BaseAgent):
    """Neural Network based agent (placeholder)."""

    def __init__(self, hidden_size: int = 128):
        """Initialize neural network agent."""
        self.hidden_size = hidden_size
        # TODO: Implement neural network agent
        pass

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using neural network."""
        # TODO: Implement
        pass

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update network."""
        # TODO: Implement
        pass
