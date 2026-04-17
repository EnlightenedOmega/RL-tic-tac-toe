"""Interactive script to play Tic-Tac-Toe against trained agent."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.utils import load_model


def print_board(env):
    """Print current board state."""
    board = env.board.reshape(3, 3)
    symbols = {0: " ", 1: "X", -1: "O"}
    print("\n   0   1   2")
    for i in range(3):
        print(f"{i}  {symbols[board[i, 0]]} | {symbols[board[i, 1]]} | {symbols[board[i, 2]]}")
        if i < 2:
            print("  -----------")
    print()


def get_human_move(env):
    """Get human player move."""
    valid_moves = env.get_valid_moves()
    while True:
        try:
            move = int(input("Enter your move (0-8): "))
            if move in valid_moves:
                return move
            print(f"Invalid move. Valid moves: {valid_moves}")
        except ValueError:
            print("Please enter a valid number.")


def main():
    """Main entry point for human vs agent game."""
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe against trained agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play")

    args = parser.parse_args()

    # Load trained model
    agent = load_model(args.model)

    # Play games
    for game_num in range(args.num_games):
        print(f"\n=== Game {game_num + 1} ===")
        env = TicTacToeEnv()

        while not env.is_terminal():
            print_board(env)

            # Human move
            human_move = get_human_move(env)
            env.step(human_move)

            if env.is_terminal():
                break

            # Agent move
            state = env.get_state()
            agent_move = agent.select_action(state, epsilon=0.0)
            env.step(agent_move)

        print_board(env)
        reward = env.reward
        if reward > 0:
            print("You won!")
        elif reward < 0:
            print("Agent won!")
        else:
            print("It's a draw!")


if __name__ == "__main__":
    main()
