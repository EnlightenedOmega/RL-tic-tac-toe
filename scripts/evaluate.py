"""Evaluation script for trained RL agent."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tictactoe_rl.evaluator import Evaluator
from tictactoe_rl.utils import load_model
from tictactoe_rl.plotting import plot_evaluation_results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    
    parser.add_argument(
        "--plot_output",
        type=str,
        default="artifacts\\plots\\evaluation_results.png",
        help="Path to save evaluation plot",
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load trained agent_x model
    agent_x = load_model(args.model)

    # Initialize evaluator
    evaluator = Evaluator(agent=agent_x, verbose=args.verbose)

    # Evaluate agent
    results = evaluator.evaluate(num_episodes=args.num_episodes)
    plot_evaluation_results(results, output_path=args.plot_output)

    print("\n=== Evaluation Results ===")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Loss rate: {results['loss_rate']:.2%}")
    print(f"Draw rate: {results['draw_rate']:.2%}")


if __name__ == "__main__":
    main()
