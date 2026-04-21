"""Interactive script to play Tic-Tac-Toe against trained agent."""

import argparse
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
    parser.add_argument(
        "--player",
        type=str,
        choices=["x", "o"],
        default=None,
        help="Play as X or O (default: prompt user)",
    )
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play")

    args = parser.parse_args()

    # Load trained agent_x model
    agent = load_model(args.model)

    # Determine player choice
    player_symbol = args.player
    if player_symbol is None:
        while True:
            choice = input("Do you want to play as X or O? (x/o): ").strip().lower()
            if choice in ["x", "o"]:
                player_symbol = choice
                break
            print("Invalid choice. Please enter 'x' or 'o'.")

    human_is_x = player_symbol == "x"

    # Play games
    for game_num in range(args.num_games):
        print(f"\n=== Game {game_num + 1} ===")
        if human_is_x:
            print("You play as X (first), Agent plays as O")
        else:
            print("Agent plays as X (first), You play as O")
        
        env = TicTacToeEnv()

        while not env.is_terminal():
            state = env.get_state()
            
            if human_is_x:
                # Human plays as X
                if env.current_player == 1:
                    human_move = get_human_move(env)
                    env.step(human_move)
                else:
                    # Agent plays as O
                    agent_move = agent.select_action(state, epsilon=0.0)
                    print(f"Agent plays: {agent_move}")
                    env.step(agent_move)
                    print_board(env)
            else:
                # Agent plays as X, human plays as O
                if env.current_player == 1:
                    agent_move = agent.select_action(state, epsilon=0.0)
                    print(f"Agent plays: {agent_move}")
                    env.step(agent_move)
                    print_board(env)
                else:
                    human_move = get_human_move(env)
                    env.step(human_move)

        print_board(env)
        reward = env.reward
        
        if human_is_x:
            # Human is X, so positive reward means human won
            if reward > 0:
                print("You won!")
            elif reward < 0:
                print("Agent won!")
            else:
                print("It's a draw!")
        else:
            # Human is O, so negative reward means human won
            if reward > 0:
                print("Agent won!")
            elif reward < 0:
                print("You won!")
            else:
                print("It's a draw!")


if __name__ == "__main__":
    main()
