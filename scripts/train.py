"""Training script for RL agent."""

import argparse

from tictactoe_rl.trainer import Trainer
from tictactoe_rl.utils import load_config
from tictactoe_rl.plotting import plot_training_history


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
    
    # Plot training history
    history = trainer.get_training_history()
    plot_output = args.output.replace(".pkl", "_history.png")
    plot_training_history(history, output_path=plot_output)

    # Save trained model (splits into agent_x.pkl and agent_o.pkl)
    trainer.save_model(args.output)
    base_path = args.output.rsplit(".", 1)[0] if "." in args.output else args.output
    print(f"Models saved to {base_path}_x.pkl and {base_path}_o.pkl")


if __name__ == "__main__":
    main()
