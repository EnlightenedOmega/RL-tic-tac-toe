"""Trainer for RL agents."""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.agents import QLearningAgent


class Trainer:
    """Trainer for two Q-Learning agents playing Tic-Tac-Toe via self-play."""

    def __init__(
        self,
        config: Dict[str, Any],
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """Initialize trainer.

        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.config = config
        self.verbose = verbose
        self.training_config = config.get("training", {})

        if seed is not None:
            np.random.seed(seed)

        # Initialize environment
        self.env = TicTacToeEnv()

        # Initialize two separate agents
        lr = self.training_config.get("learning_rate", 0.1)
        gamma = self.training_config.get("discount_factor", 0.99)

        self.agent_x = QLearningAgent(learning_rate=lr, discount_factor=gamma)
        self.agent_o = QLearningAgent(learning_rate=lr, discount_factor=gamma)

        # Training state
        self.epsilon = self.training_config.get("epsilon_start", 1.0)
        self.episode_count = 0

        # Stats tracked over full training
        self.total_stats = {"x_wins": 0, "o_wins": 0, "draws": 0}

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run self-play training for configured number of episodes."""
        num_episodes = self.training_config.get("num_iterations", 5000)
        epsilon_end = self.training_config.get("epsilon_end", 0.01)
        epsilon_decay = self.training_config.get("epsilon_decay", 0.995)
        log_interval = self.training_config.get("log_interval", 500)

        if self.verbose:
            print(f"Starting training for {num_episodes} episodes")
            print(f"  learning_rate  : {self.training_config.get('learning_rate', 0.1)}")
            print(f"  discount_factor: {self.training_config.get('discount_factor', 0.99)}")
            print(f"  epsilon_start  : {self.epsilon}")
            print(f"  epsilon_end    : {epsilon_end}")
            print(f"  epsilon_decay  : {epsilon_decay}")
            print("-" * 60)

        # Stats for the current logging window
        window_stats = {"x_wins": 0, "o_wins": 0, "draws": 0}

        for episode in range(num_episodes):
            result = self._run_episode()

            # Accumulate stats
            window_stats[result] += 1
            self.total_stats[result] += 1
            self.episode_count += 1

            # Decay epsilon after every episode
            self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

            # Logging
            if self.verbose and (episode + 1) % log_interval == 0:
                self._log_progress(episode + 1, num_episodes, window_stats)
                window_stats = {"x_wins": 0, "o_wins": 0, "draws": 0}

        if self.verbose:
            print("-" * 60)
            print("Training complete.")
            print(f"  Total X wins : {self.total_stats['x_wins']}")
            print(f"  Total O wins : {self.total_stats['o_wins']}")
            print(f"  Total draws  : {self.total_stats['draws']}")
            print(f"  Q-table size (X): {len(self.agent_x.q_table)} states")
            print(f"  Q-table size (O): {len(self.agent_o.q_table)} states")

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(self) -> str:
        self.env.reset()
        done = False

        last_x = None
        last_o = None
        result = "draws"

        while not done:

            state = self.env.get_state()
            action_x = self.agent_x.select_action(state, epsilon=self.epsilon)
            
            # DEBUG
            print(f"[X] board: {self.env.get_state()}, action: {action_x}, valid: {self.env.get_valid_moves()}")
            
            last_x = (state, action_x)
            next_state, reward, done = self.env.step(action_x)

            if done:
                self.agent_x.update(state, action_x, reward, next_state, done=True)
                if last_o is not None:
                    self.agent_o.update(last_o[0], last_o[1], -reward, next_state, done=True)
                result = "x_wins" if reward > 0 else "draws"
                break

            self.agent_x.update(state, action_x, reward, next_state, done=False)
            state = self.env.get_state()

            action_o = self.agent_o.select_action(state, epsilon=self.epsilon)
            
            # DEBUG
            print(f"[O] board: {self.env.get_state()}, action: {action_o}, valid: {self.env.get_valid_moves()}")
            
            last_o = (state, action_o)
            next_state, reward, done = self.env.step(action_o)

            if done:
                self.agent_o.update(state, action_o, reward, next_state, done=True)
                if last_x is not None:
                    self.agent_x.update(last_x[0], last_x[1], -reward, next_state, done=True)
                result = "o_wins" if reward > 0 else "draws"
                break

            self.agent_o.update(state, action_o, reward, next_state, done=False)

        return result 
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_progress(
        self,
        episode: int,
        total: int,
        window_stats: Dict[str, int],
    ):
        """Print a progress line for the current logging window.

        Args:
            episode: Current episode number (1-indexed)
            total: Total number of episodes
            window_stats: Win/draw counts since last log
        """
        n = sum(window_stats.values())
        x_pct = 100 * window_stats["x_wins"] / n if n else 0
        o_pct = 100 * window_stats["o_wins"] / n if n else 0
        d_pct = 100 * window_stats["draws"] / n if n else 0

        print(
            f"Episode {episode:>6}/{total} | "
            f"epsilon: {self.epsilon:.4f} | "
            f"X wins: {window_stats['x_wins']:>4} ({x_pct:5.1f}%) | "
            f"O wins: {window_stats['o_wins']:>4} ({o_pct:5.1f}%) | "
            f"Draws: {window_stats['draws']:>4} ({d_pct:5.1f}%)"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        """Save both trained agents to disk.

        Args:
            path: File path to save the model (pkl)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_x": self.agent_x,
            "agent_o": self.agent_o,
            "config": self.config,
            "total_stats": self.total_stats,
            "episode_count": self.episode_count,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        if self.verbose:
            print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load agents from a previously saved file.

        Args:
            path: File path to load the model from
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.agent_x = payload["agent_x"]
        self.agent_o = payload["agent_o"]
        self.total_stats = payload.get("total_stats", self.total_stats)
        self.episode_count = payload.get("episode_count", 0)

        if self.verbose:
            print(f"Model loaded from {path}")
            print(f"  Resumed from episode {self.episode_count}")