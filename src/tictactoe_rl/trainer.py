"""Trainer for RL agents."""

import pickle
from pathlib import Path
from typing import Dict, Any

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.agents import QLearningAgent


class Trainer:
    """Trainer for RL agents."""

    def __init__(self, config: Dict[str, Any], seed: int = None, verbose: bool = False):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
            seed: Random seed
            verbose: Verbose output
        """
        self.config = config
        self.verbose = verbose

        if seed is not None:
            import numpy as np
            np.random.seed(seed)

        # Initialize environment and agent
        self.env = TicTacToeEnv()
        self.agent = QLearningAgent(
            learning_rate=config.get("training", {}).get("learning_rate", 0.1),
            discount_factor=config.get("training", {}).get("discount_factor", 0.99),
        )

        self.training_config = config.get("training", {})

    def train(self):
        """Train the agent."""
        num_episodes = self.training_config.get("num_iterations", 1000)
        epsilon_start = self.training_config.get("epsilon_start", 1.0)
        epsilon_end = self.training_config.get("epsilon_end", 0.01)
        epsilon_decay = self.training_config.get("epsilon_decay", 0.995)

        epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                # Agent 1 move
                action = self.agent.select_action(state, epsilon=epsilon)
                next_state, reward, done = self.env.step(action)
                
                # For two-player game, flip reward for agent 2's perspective
                self.agent.update(state, action, reward, next_state, done)

                if done:
                    break

                # Agent 2 move (random opponent for now)
                valid_moves = self.env.get_valid_moves()
                if valid_moves:
                    action2 = valid_moves[0]  # Simple deterministic opponent
                    next_state, _, done = self.env.step(action2)

                state = next_state

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if self.verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, epsilon: {epsilon:.4f}")

        if self.verbose:
            print("Training completed!")

    def save_model(self, path: str):
        """Save trained agent.
        
        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.agent, f)
