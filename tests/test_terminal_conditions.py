"""Additional tests for terminal conditions in TicTacToe."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from tictactoe_rl.env import TicTacToeEnv


class TestTerminalConditionsDetailed:
    """Detailed tests for terminal conditions."""

    def test_all_horizontal_wins(self):
        """Test all possible horizontal wins."""
        for row in range(3):
            env = TicTacToeEnv()
            positions = [row * 3 + col for col in range(3)]
            
            # Create opponent moves
            env.step(positions[0])  # Player 1
            env.step((row + 1) * 3)  # Player 2 (different row)
            env.step(positions[1])  # Player 1
            env.step((row + 1) * 3 + 1)  # Player 2
            
            # Winning move
            _, reward, done = env.step(positions[2])  # Player 1
            
            assert done, f"Game should end for row {row}"
            assert reward == 1.0, f"Player 1 should win in row {row}"

    def test_all_vertical_wins(self):
        """Test all possible vertical wins."""
        for col in range(3):
            env = TicTacToeEnv()
            positions = [row * 3 + col for row in range(3)]
            
            # Create opponent moves
            env.step(positions[0])  # Player 1
            env.step(col)  # Player 2 (different column)
            env.step(positions[1])  # Player 1
            env.step(col + 3)  # Player 2
            
            # Winning move
            _, reward, done = env.step(positions[2])  # Player 1
            
            assert done, f"Game should end for column {col}"
            assert reward == 1.0, f"Player 1 should win in column {col}"

    def test_all_diagonal_wins(self):
        """Test all possible diagonal wins."""
        # Main diagonal (0, 4, 8)
        env = TicTacToeEnv()
        env.step(0)  # Player 1
        env.step(1)  # Player 2
        env.step(4)  # Player 1
        env.step(2)  # Player 2
        _, reward, done = env.step(8)  # Player 1
        
        assert done
        assert reward == 1.0

        # Anti-diagonal (2, 4, 6)
        env = TicTacToeEnv()
        env.step(2)  # Player 1
        env.step(0)  # Player 2
        env.step(4)  # Player 1
        env.step(1)  # Player 2
        _, reward, done = env.step(6)  # Player 1
        
        assert done
        assert reward == 1.0

    def test_no_early_termination(self):
        """Test that game doesn't terminate prematurely."""
        env = TicTacToeEnv()
        
        # Make some moves without completing a line
        env.step(0)  # Player 1
        env.step(1)  # Player 2
        env.step(3)  # Player 1
        
        assert not env.is_terminal()
        assert len(env.get_valid_moves()) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
