# Tic-Tac-Toe Reinforcement Learning Implementation Notes

## Project Overview

This project implements a self-play reinforcement learning system where two Q-Learning agents (agent_x and agent_o) learn to play Tic-Tac-Toe through competitive training. The agents are trained using the Q-Learning algorithm with epsilon-greedy exploration policies.

---

## Approach and Methodology

### Architecture

The system is built on a modular architecture with clear separation of concerns:

- **Environment (env.py)**: Implements the Tic-Tac-Toe game mechanics with 1D array representation (9 positions: 0-8)
- **Agents (agents.py)**: Implements Q-Learning agents with Q-tables mapping states to action values
- **Trainer (trainer.py)**: Orchestrates self-play training between two agents with reward signals
- **Evaluator (evaluator.py)**: Evaluates trained agents against random opponents and each other
- **Utilities (utils.py)**: Handles model persistence (pickle-based), configuration loading, and common functions

### Learning Algorithm

Q-Learning Update Rule:
Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))

Where:
- alpha: learning_rate (default: 0.05)
- gamma: discount_factor (default: 0.99)
- r: reward signal (1.0 for win, -1.0 for loss, 0.0 for draw)

### Exploration Strategy

Epsilon-Greedy Policy:
- Epsilon start: 1.0 (full exploration)
- Epsilon end: 0.01 (minimal exploration)
- Decay rate: 0.995 per episode
- Ensures agents explore early and exploit learned policies later

### Reward Structure

Zero-Sum Game:
- Winner gets +1.0 reward
- Loser gets -1.0 reward
- Draw gives 0.0 to both
- Agents update retroactively: if X wins, O also sees negative reward for its last move

---

## Technical Implementation Details

### State Representation

Board is represented as 1D numpy array with 9 elements:
- 1: Agent X (maximizing player)
- -1: Agent O (minimizing player)
- 0: Empty cell

Example: `[1, 0, -1, 0, 1, 0, -1, 0, 0]` represents a partially filled board

### Action Space

Valid moves are positions 0-8 corresponding to board positions:
```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

Invalid moves (occupied cells) are filtered out during action selection using `-np.inf` assignment

### Q-Table Storage

Q-table is a dictionary mapping state tuples to 9-dimensional numpy arrays:
- Key: tuple representation of board state (immutable, hashable)
- Value: Q-values for each of 9 possible actions

Only visited states are stored (sparse representation)

### Episode Structure

Single training episode flow:
1. Reset environment to empty board, set X as current player
2. Loop until terminal:
   - X selects action using epsilon-greedy policy
   - X updates its Q-table
   - Environment steps, checks terminal
   - If terminal: update both agents with final rewards and break
   - O selects action using epsilon-greedy policy
   - O updates its Q-table
   - Environment steps, checks terminal
   - If terminal: update both agents with final rewards and break
3. Return episode result (x_wins, o_wins, or draws)

### Model Persistence

Split Pickle Architecture:
- agent_x saved to `{base}_x.pkl`
- agent_o saved to `{base}_o.pkl`
- Each pickle contains only the agent object with its Q-table
- Supports backward compatibility through auto-detection in load_model()

---

## Implementation Details

### Self-Play Mechanism

The key insight of self-play training:
- Both agents play against each other continuously
- Each agent learns from wins and losses
- No fixed opponent bias after convergence
- Leads to symmetric strategies in zero-sum games

### Retroactive Updates

When an episode ends:
- Winning agent gets +1 reward for its last move
- Losing agent gets -1 reward for its last move
- Both agents retroactively update Q-values for final states
- Non-final moves updated with intermediate rewards (typically 0)

### Epsilon Decay

Gradual transition from exploration to exploitation:
- Formula: epsilon = max(epsilon_end, epsilon * decay_rate)
- Ensures consistent learning without sudden policy shifts
- Configurable parameters allow tuning exploration-exploitation trade-off

---

## Challenges Faced and Overcome

### Challenge 1: First-Mover Advantage

Problem: Agent X always plays first, creating structural advantage in game state distribution.

Solution: Accepted as inherent game asymmetry. The training correctly reflects real Tic-Tac-Toe dynamics where first player has advantage even with perfect play.

### Challenge 2: State Representation Ambiguity

Problem: Initial implementation didn't distinguish between game states that are identical in board configuration but reached via different move sequences.

Solution: Used tuple-based state representation which is deterministic based only on current board configuration, not history. This is sufficient for Markovian decision processes.

### Challenge 3: Invalid Move Handling

Problem: Q-learning could assign positive values to illegal moves (occupied cells).

Solution: Implemented filtering by setting Q-values of invalid moves to `-np.inf` during action selection. This ensures `argmax` never selects illegal moves.

### Challenge 4: Reward Signal for Non-Terminal States

Problem: How to reward intermediate moves that don't immediately result in game end?

Solution: Implemented zero-intermediate rewards with only final states providing terminal rewards. This allows bootstrapping through value iteration and is standard in Q-Learning for this domain.

### Challenge 5: Model Serialization Across Runs

Problem: Original single-pickle design stored metadata (config, stats, episode_count) that became stale across runs.

Solution: Split into two separate pickles containing only the learned Q-tables (agent_x and agent_o). Metadata is regenerated per training session.

### Challenge 6: Backward Compatibility During Refactoring

Problem: Converting from single pickle to split pickles broke existing scripts and saved models.

Solution: Implemented smart `load_model()` function that auto-detects file format:
- First tries split files (modern format)
- Falls back to single pickle (legacy format)
- Handles both old dict format and new direct agent format

### Challenge 7: Human Interaction Player Selection

Problem: play_human.py hardcoded agent as X, limiting user experience.

Solution: Added command-line parameter `--player` with interactive fallback. Implemented separate game loops for human_as_x and human_as_o cases with correct reward interpretation.

---

## Current Limitations

### Limitation 1: Tic-Tac-Toe Solved Game

The game is fully solved. With perfect play, all games end in draws. The training will converge to this equilibrium after sufficient episodes. This is not a limitation of the implementation but of the game domain itself.

Expected convergence: ~80-95% draws, X wins 5-15%, O wins 0-5%

### Limitation 2: Q-Table Scalability

Current Q-Table approach stores every visited state explicitly in memory. For Tic-Tac-Toe (3^9 = 19683 possible states), this is manageable. However:
- For larger board sizes (5x5, 7x7), this becomes infeasible
- For continuous state spaces, tabular Q-Learning is unsuitable

Mitigation: Placeholder for neural network agent exists (NeuralNetworkAgent class) but not implemented.

### Limitation 3: No Symmetry Exploitation

The Q-table doesn't exploit board symmetries (rotations, reflections). This means:
- Learning is less efficient than theoretically possible
- Larger Q-tables than necessary
- More episodes needed for convergence

Note: This is intentional for simplicity. Symmetry-aware implementation would require state canonicalization.

### Limitation 4: Evaluation Limited to Random Opponents

The Evaluator only evaluates against random opponents, not against:
- Other trained agents
- Known opening systems
- Minimax-optimal players

This limits understanding of agent performance on meaningful baselines.

### Limitation 5: No Curriculum Learning

All training uses identical initial epsilon and learning rate throughout. No curriculum strategy to:
- Start with easier opponents (random, then weak agents)
- Gradually increase difficulty
- Use learning rate scheduling

### Limitation 6: Single Seed Control

Random seed can only be set globally. No per-agent or per-episode seeding for reproducible but independent randomness.

### Limitation 7: No Convergence Metrics

Training doesn't measure:
- TD-error over time
- Q-value estimates convergence
- Value function stability

Makes it difficult to determine optimal training duration vs. overfitting.

### Limitation 8: Board Representation Inefficiency

1D array with values {-1, 0, 1} uses 64 bits per cell. Could use bitboards for 2-3x memory efficiency, but not implemented for readability trade-off.

---

## Future Enhancement Opportunities

1. Implement DQN (Deep Q-Network) using neural networks for larger state spaces
2. Add parallel self-play training across multiple processes
3. Implement multi-agent environments (3+ players)
4. Add state canonicalization for symmetry exploitation
5. Implement curriculum learning with difficulty progression
6. Add tournament-style evaluation against multiple agents
7. Implement experience replay buffer for Q-Learning
8. Add training checkpoints and resumable training
9. Implement policy gradient methods as alternative to Q-Learning
10. Add tensorboard/wandb integration for training visualization

---

## Configuration Parameters

Default training configuration (configs/train.yaml):

- algorithm: q_learning
- learning_rate: 0.05
- discount_factor: 0.99
- epsilon_start: 1.0
- epsilon_end: 0.01
- epsilon_decay: 0.995
- num_iterations: 5000
- log_interval: 50
- validation_interval: 500

These can be modified in the config file or programmatically in Trainer initialization.

---

## Testing and Validation

Current test coverage:
- test_env.py: Environment mechanics and move validation
- test_terminal_conditions.py: Win condition detection for all 8 winning lines

Tests verify:
- Valid move generation
- Board state transitions
- Win detection accuracy
- Draw detection

No tests currently for:
- Agent learning correctness
- Q-value convergence
- Trainer episode mechanics
