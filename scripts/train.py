"""Training script for RL agent."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tictactoe_rl.trainer import Trainer
from tictactoe_rl.utils import load_config


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/models/agent.pkl",
        help="Path to save trained model",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize trainer
    trainer = Trainer(config=config, seed=args.seed, verbose=args.verbose)

    # Train agent
    trainer.train()

    # Save trained model
    trainer.save_model(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
