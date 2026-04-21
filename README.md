# Tic-Tac-Toe Reinforcement Learning Agents

A comprehensive reinforcement learning implementation where two Q-Learning agents learn to play Tic-Tac-Toe through competitive self-play training. The agents learn optimal strategies through exploration and exploitation, converging toward near-perfect play.

## Project Structure

```
tic-tac-toe-rl/
├── configs/                          # Configuration files
│   ├── default.yaml                 # Default settings
│   └── train.yaml                   # Training-specific hyperparameters
├── scripts/                         # Entry point scripts
│   ├── train.py                     # Train agents via self-play
│   ├── evaluate.py                  # Evaluate agent against random opponent
│   └── play_human.py                # Interactive human vs agent gameplay
├── src/tictactoe_rl/               # Main source code
│   ├── __init__.py
│   ├── agents.py                    # Q-Learning agent implementation
│   ├── env.py                       # Tic-Tac-Toe environment
│   ├── trainer.py                   # Self-play training orchestration
│   ├── evaluator.py                 # Agent evaluation utilities
│   ├── plotting.py                  # Visualization utilities
│   └── utils.py                     # Configuration and model I/O
├── tests/                           # Unit tests
│   ├── test_env.py                  # Environment tests
│   └── test_terminal_conditions.py  # Terminal state detection tests
├── artifacts/                       # Output directory
│   ├── models/                      # Trained agent models (split pickles)
│   ├── logs/                        # Training logs
│   └── plots/                       # Training history plots
├── notes.md                         # Technical documentation
├── poetry.toml                      # Poetry configuration
├── pyproject.toml                   # Project dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or Poetry

### Using Poetry (Recommended)
```bash
poetry install
poetry shell
```

### Using pip
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training

Train two agents (agent_x and agent_o) via self-play:

```bash
# Using default config
python scripts/train.py

# With custom config
python scripts/train.py --config configs/train.yaml --output artifacts/models/agent.pkl --verbose

# With specific random seed
python scripts/train.py --seed 42 --verbose
```

Output:
- `artifacts/models/agent_x.pkl` - Trained Agent X (first player)
- `artifacts/models/agent_o.pkl` - Trained Agent O (second player)
- `artifacts/models/agent_history.png` - Training history plot

### 2. Evaluation

Evaluate a trained agent against a random opponent:

```bash
# Evaluate agent_x
python scripts/evaluate.py --model artifacts/models/agent.pkl --num_episodes 100 --verbose

# With custom plot output
python scripts/evaluate.py --model artifacts/models/agent.pkl --plot_output artifacts/plots/eval_results.png
```

### 3. Play Against Trained Agent

Interactive gameplay where you can choose to play as X or O:

```bash
# Interactive prompt for player selection
python scripts/play_human.py --model artifacts/models/agent.pkl --num_games 3

# Explicitly play as X (first move)
python scripts/play_human.py --model artifacts/models/agent.pkl --player x

# Play as O (second move)
python scripts/play_human.py --model artifacts/models/agent.pkl --player o
```

During gameplay, enter moves as numbers 0-8:
```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

## Configuration

Training hyperparameters are configured via YAML files in `configs/`:

### Training Configuration (configs/train.yaml)

```yaml
training:
  algorithm: q_learning              # Learning algorithm
  learning_rate: 0.05                # Alpha - weight of TD update (0.01-0.1)
  discount_factor: 0.99              # Gamma - importance of future rewards
  epsilon_start: 1.0                 # Initial exploration rate (0-1)
  epsilon_end: 0.01                  # Final exploration rate after decay
  epsilon_decay: 0.995               # Decay multiplier per episode
  num_iterations: 5000               # Number of self-play episodes
  log_interval: 50                   # Episodes between progress logs
```

### Key Parameters Explained

- **learning_rate**: Controls how much new information overrides old Q-values. Higher = faster learning but less stability.
- **discount_factor**: How much future rewards are valued. 0.99 = long-term planning, 0.0 = immediate rewards only.
- **epsilon_decay**: How quickly exploration decreases. Higher (closer to 1.0) = slower transition to exploitation.
- **num_iterations**: More episodes = better convergence but longer training time. Tic-Tac-Toe typically converges around 5000 episodes.

## Architecture Details

### State Representation

Board state: 1D numpy array of 9 elements

```
Values: 1 (Agent X), -1 (Agent O), 0 (Empty)
Positions:
  0 | 1 | 2
  ---------
  3 | 4 | 5
  ---------
  6 | 7 | 8
```

### Game Flow

1. X plays first (current_player = 1)
2. After X moves, board state checked for terminal conditions
3. If not terminal, switch to O (current_player = -1)
4. After O moves, board state checked for terminal conditions
5. Both agents updated with appropriate rewards
6. Episode ends or continues

### Terminal Conditions

Game ends on any of:
- Any player completes 3-in-a-row (winning line: rows, columns, diagonals)
- All 9 board positions filled with no winner (draw)

## Learning Algorithm

### Q-Learning

Agents use Q-Learning, a model-free reinforcement learning algorithm that learns the value (Q-value) of each action in each state.

Update Rule:
```
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
```

### Training Dynamics

- Both agents play against each other continuously (self-play)
- Each agent explores new moves with probability epsilon (decreasing over time)
- Each agent exploits best-known moves with probability 1 - epsilon
- Zero-sum rewards: winner gets +1, loser gets -1, draw = 0 for both
- Agents converge toward Nash equilibrium (approximately 80-95% draws)

### Expected Training Outcomes

After 5000 episodes of self-play:

- **Draw Rate**: 80-95% (game-theoretic equilibrium)
- **Agent X Win Rate**: 5-15% (first-mover advantage)
- **Agent O Win Rate**: 0-5% (plays second)

Agent X advantage is inherent to Tic-Tac-Toe; with perfect play from both agents, the game always results in a draw.

## Model Persistence

Models are saved as two separate pickle files:

```
artifacts/models/agent_x.pkl    # Agent X Q-table
artifacts/models/agent_o.pkl    # Agent O Q-table
```

This split design:
- Allows independent loading of each agent
- Enables easier model inspection and debugging
- Supports backward compatibility with legacy formats

## Development

### Setting Up Development Environment

```bash
# Install with development dependencies
poetry install --with dev

# Activate virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_env.py

# Run with coverage report
pytest tests/ --cov=src/tictactoe_rl --cov-report=html
```

### Current Test Coverage

- `test_env.py`: Environment mechanics, board state transitions, valid move generation
- `test_terminal_conditions.py`: Win condition detection for all 8 winning lines, draw detection

### Code Quality and Linting

```bash
# Format code with Black
black src/ scripts/ tests/

# Lint with Ruff
ruff check src/ scripts/ tests/ --fix

# Type checking with Mypy (if configured)
mypy src/
```

### Project Architecture

#### Core Components

- **agents.py**: QLearningAgent class with Q-table management and epsilon-greedy policy
- **env.py**: TicTacToeEnv implementing game mechanics and win detection
- **trainer.py**: Trainer class orchestrating self-play episodes and learning
- **evaluator.py**: Evaluator for testing trained agents against random opponents
- **utils.py**: Configuration loading, model persistence, and utilities
- **plotting.py**: Visualization of training history and evaluation results

#### Key Classes

- `QLearningAgent`: Q-Learning agent with `select_action()` and `update()` methods
- `TicTacToeEnv`: Game environment with `step()` and terminal state checking
- `Trainer`: Self-play orchestration with configurable hyperparameters
- `Evaluator`: Agent evaluation and performance metrics

## Documentation

Detailed technical documentation available in `notes.md`:
- Approach and methodology
- Challenges overcome
- Current limitations
- Future enhancements

- Future enhancements

## Troubleshooting

### Agent Performs Randomly After Loading

Ensure you're loading the correct model file. Use `--verbose` flag to check file paths:
```bash
python scripts/evaluate.py --model artifacts/models/agent.pkl --verbose
```

### Training Too Slow

Reduce `num_iterations` in config for faster iteration during development:
```yaml
training:
  num_iterations: 1000  # Instead of 5000
```

### No Output During Training

Use `--verbose` flag to see training progress:
```bash
python scripts/train.py --verbose
```

## Contributing

Contributions welcome! Please ensure:
1. Code passes all tests: `pytest tests/`
2. Code is formatted: `black src/ scripts/ tests/`
3. No lint issues: `ruff check src/ scripts/ tests/`

## References

- Q-Learning: Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3), 279-292.
- Tic-Tac-Toe Complexity: There are 5,478 unique board positions (not counting symmetries) and 255,168 possible games.

## License

MIT License - see LICENSE file for details

## Authors

Reinforcement Learning implementation for Tic-Tac-Toe
