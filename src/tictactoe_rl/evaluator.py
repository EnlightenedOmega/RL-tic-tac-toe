"""Evaluator for trained agents."""

import numpy as np
from typing import Dict, Any

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.agents import BaseAgent


class Evaluator:
    """Evaluator for RL agents."""

    def __init__(self, agent: BaseAgent, verbose: bool = False):
        """Initialize evaluator.
        
        Args:
            agent: Agent to evaluate
            verbose: Verbose output
        """
        self.agent = agent
        self.verbose = verbose

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate agent against random opponent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        wins = 0
        losses = 0
        draws = 0

        for episode in range(num_episodes):
            env = TicTacToeEnv()
            state = env.reset()
            done = False

            while not done:
                # Agent move
                action = self.agent.select_action(state, epsilon=0.0)
                next_state, reward, done = env.step(action)

                if done:
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    break

                # Random opponent move
                valid_moves = env.get_valid_moves()
                if valid_moves:
                    opponent_action = np.random.choice(valid_moves)
                    next_state, _, done = env.step(opponent_action)

                state = next_state

            if self.verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")

        total = wins + losses + draws
        return {
            "win_rate": wins / total if total > 0 else 0.0,
            "loss_rate": losses / total if total > 0 else 0.0,
            "draw_rate": draws / total if total > 0 else 0.0,
            "total_episodes": total,
        }
