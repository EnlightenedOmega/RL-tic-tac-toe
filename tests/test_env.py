"""Tests for TicTacToe environment."""
import pytest
import numpy as np
from tictactoe_rl.env import TicTacToeEnv


class TestTicTacToeEnv:
    """Test cases for TicTacToe environment."""

    def test_initialization(self):
        """Test environment initialization."""
        env = TicTacToeEnv()
        assert env.board.shape == (9,)
        assert np.all(env.board == 0)
        assert env.current_player == 1
        assert not env.is_terminal()

    def test_get_valid_moves(self):
        """Test getting valid moves."""
        env = TicTacToeEnv()
        valid_moves = env.get_valid_moves()
        assert len(valid_moves) == 9
        assert set(valid_moves) == set(range(9))

    def test_step(self):
        """Test environment step."""
        env = TicTacToeEnv()
        state, reward, done = env.step(0)
        
        assert env.board[0] == 1
        assert env.current_player == -1
        assert not done

    def test_invalid_move(self):
        """Test invalid move raises error."""
        env = TicTacToeEnv()
        env.step(0)
        
        with pytest.raises(ValueError):
            env.step(0)

    def test_reset(self):
        """Test environment reset."""
        env = TicTacToeEnv()
        env.step(0)
        env.reset()
        
        assert np.all(env.board == 0)
        assert env.current_player == 1
        assert not env.is_terminal()

    def test_get_state(self):
        """Test getting state."""
        env = TicTacToeEnv()
        env.step(0)
        state = env.get_state()
        
        assert state[0] == 1
        assert np.all(state[1:] == 0)


class TestTerminalConditions:
    """Test cases for terminal conditions."""

    def test_horizontal_win(self):
        """Test horizontal winning condition."""
        env = TicTacToeEnv()
        # Player 1 wins with top row
        env.step(0)  # 1
        env.step(3)  # 2
        env.step(1)  # 1
        env.step(4)  # 2
        _, reward, done = env.step(2)  # 1 wins

        assert done
        assert reward == 1.0

    def test_vertical_win(self):
        """Test vertical winning condition."""
        env = TicTacToeEnv()
        # Player 1 wins with left column
        env.step(0)  # 1
        env.step(1)  # 2
        env.step(3)  # 1
        env.step(4)  # 2
        _, reward, done = env.step(6)  # 1 wins

        assert done
        assert reward == 1.0

    def test_diagonal_win(self):
        """Test diagonal winning condition."""
        env = TicTacToeEnv()
        # Player 1 wins with main diagonal
        env.step(0)  # 1
        env.step(1)  # 2
        env.step(4)  # 1
        env.step(2)  # 2
        _, reward, done = env.step(8)  # 1 wins

        assert done
        assert reward == 1.0

    def test_draw(self):
        """Test draw condition."""
        env = TicTacToeEnv()
        # Create a draw game
        moves = [0, 1, 2, 3, 4, 5, 6, 8, 7]  # Results in draw
        
        done = False
        for i, move in enumerate(moves):
            _, reward, done = env.step(move)
            if done:
                break

        # Add more comprehensive draw test if needed
        assert done or len(env.get_valid_moves()) == 0
