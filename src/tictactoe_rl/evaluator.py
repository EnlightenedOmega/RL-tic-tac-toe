"""Evaluator for trained agents."""

import numpy as np
from typing import Dict, Any, Optional

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.agents import BaseAgent


class Evaluator:
    """Evaluator for RL agents."""

    def __init__(self, agent: Optional[BaseAgent] = None, agent_x: Optional[BaseAgent] = None, 
                 agent_o: Optional[BaseAgent] = None, verbose: bool = False):
        """Initialize evaluator.
        
        Args:
            agent: Single agent to evaluate (for backward compatibility)
            agent_x: Agent X for evaluation
            agent_o: Agent O for evaluation
            verbose: Verbose output
        """
        # Support both single agent and dual agent initialization
        if agent is not None:
            self.agent = agent
            self.agent_x = agent
            self.agent_o = None
        else:
            self.agent = agent_x
            self.agent_x = agent_x
            self.agent_o = agent_o
        
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

    def evaluate_agent_vs_random(self, agent: BaseAgent, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate a specific agent against random opponent.
        
        Args:
            agent: Agent to evaluate
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
                # Agent move (X)
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, done = env.step(action)

                if done:
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    break

                # Random opponent move (O)
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

    def evaluate_both_agents(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate both agent_x and agent_o individually against random opponent.
        
        Args:
            num_episodes: Number of episodes to evaluate each agent
            
        Returns:
            Dictionary with results for both agents
        """
        if self.agent_x is None or self.agent_o is None:
            raise ValueError("Both agent_x and agent_o must be provided for this evaluation")
        
        x_results = self.evaluate_agent_vs_random(self.agent_x, num_episodes)
        o_results = self.evaluate_agent_vs_random(self.agent_o, num_episodes)
        
        return {
            "agent_x": x_results,
            "agent_o": o_results,
        }

