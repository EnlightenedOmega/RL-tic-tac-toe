# RL Template - Tic-Tac-Toe Reinforcement Learning Agent

A reinforcement learning project template for training and evaluating agents to play Tic-Tac-Toe using q learning.

## Project Structure

```
rl-template/
├── configs/              # Configuration files (YAML)
├── scripts/              # Entry point scripts (train, evaluate, play)
├── src/tictactoe_rl/    # Main source code
├── tests/                # Unit tests
├── artifacts/            # Output directory for models, logs, and plots
│   ├── models/          # Trained agent models
│   ├── logs/            # Training logs
│   └── plots/           # Visualization plots
└── README.md            # This file
```

## Installation

### Using Poetry
```bash
poetry install
```

### Using UV
```bash
uv sync
```

### Using pip
```bash
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
python scripts/train.py --config configs/train.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --model artifacts/models/agent.pkl
```

### Play with Human
```bash
python scripts/play_human.py --model artifacts/models/agent.pkl
```

## Configuration

Configuration files are located in the `configs/` directory:
- `default.yaml` - Default configuration
- `train.yaml` - Training-specific configuration

## Development

### Running Tests
```bash
pytest tests/
```

### Running Tests with Coverage
```bash
pytest tests/ --cov=src/tictactoe_rl
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details
