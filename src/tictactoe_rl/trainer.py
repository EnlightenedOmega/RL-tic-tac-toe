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
        
        # History for plotting
        self.history = {
            "x_wins": [],
            "o_wins": [],
            "draws": [],
            "win_rate_x": [],
            "win_rate_o": [],
            "draw_rate": [],
        }

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

            # Logging and history tracking
            if (episode + 1) % log_interval == 0:
                if self.verbose:
                    self._log_progress(episode + 1, num_episodes, window_stats)
                
                # Track history
                n = sum(window_stats.values())
                self.history["x_wins"].append(window_stats["x_wins"])
                self.history["o_wins"].append(window_stats["o_wins"])
                self.history["draws"].append(window_stats["draws"])
                self.history["win_rate_x"].append(100 * window_stats["x_wins"] / n if n else 0)
                self.history["win_rate_o"].append(100 * window_stats["o_wins"] / n if n else 0)
                self.history["draw_rate"].append(100 * window_stats["draws"] / n if n else 0)
                
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

    def get_training_history(self) -> Dict[str, list]:
        """Get training history for plotting.
        
        Returns:
            Dictionary with metrics tracked during training
        """
        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        """Save both trained agents to separate disk files.

        Args:
            path: Base file path for the models (pkl). Agent X will be saved as
                  path_x.pkl and Agent O as path_o.pkl. E.g., if path is 
                  'artifacts/models/agent.pkl', files will be 'agent_x.pkl' and 'agent_o.pkl'
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Split path into base and extension
        base_path = str(path).rsplit(".", 1)[0] if "." in str(path) else str(path)
        path_x = f"{base_path}_x.pkl"
        path_o = f"{base_path}_o.pkl"
        
        # Save agent_x
        with open(path_x, "wb") as f:
            pickle.dump(self.agent_x, f)
        
        # Save agent_o
        with open(path_o, "wb") as f:
            pickle.dump(self.agent_o, f)

        if self.verbose:
            print(f"Models saved to {path_x} and {path_o}")

    def load_model(self, path: str):
        """Load agents from separately saved pickle files.

        Args:
            path: Base file path to load the models from. Will look for files
                  named path_x.pkl and path_o.pkl
        """
        # Split path into base and extension
        base_path = str(path).rsplit(".", 1)[0] if "." in str(path) else str(path)
        path_x = f"{base_path}_x.pkl"
        path_o = f"{base_path}_o.pkl"
        
        with open(path_x, "rb") as f:
            self.agent_x = pickle.load(f)
        
        with open(path_o, "rb") as f:
            self.agent_o = pickle.load(f)

        if self.verbose:
            print(f"Models loaded from {path_x} and {path_o}")
            print(f"  Resumed from episode {self.episode_count}")