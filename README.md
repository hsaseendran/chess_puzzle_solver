# Chess Puzzle Solver using Reinforcement Learning

A chess puzzle solver that combines reinforcement learning with Stockfish evaluation to learn tactical patterns and solve chess puzzles. The system uses Proximal Policy Optimization (PPO) and neural networks to learn from puzzle positions.

## Features

- **Neural Network Architecture**: Separate policy and value networks with attention mechanism for move selection
- **Stockfish Integration**: Uses Stockfish for position evaluation and move validation
- **Feature Extraction**: Comprehensive chess position features including piece placement, attack maps, material count, pawn structure, and more
- **Reinforcement Learning**: PPO algorithm with experience replay buffer
- **Puzzle Dataset Support**: Works with Lichess puzzle database format
- **Performance Analysis**: Detailed evaluation metrics and visualization tools

## Requirements

- Python 3.8+
- PyTorch 2.0+
- python-chess
- Stockfish chess engine
- See `requirements.txt` for complete list

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chess-puzzle-solver.git
cd chess-puzzle-solver
```

2. Install the package:
```bash
pip install -e .
```

3. Download and install Stockfish:
   - Visit https://stockfishchess.org/download/
   - Download the appropriate version for your OS
   - Update the `stockfish_path` in `src/config.py` or pass it as a command line argument

4. Download the Lichess puzzle database:
   - Visit https://database.lichess.org/#puzzles
   - Download the puzzle CSV file
   - Place it in the `data/` directory

## Usage

### Training

Basic training:
```bash
python scripts/train.py --data data/lichess_db_puzzle.csv
```

Training with custom configuration:
```bash
python scripts/train.py --data data/lichess_db_puzzle.csv --num-epochs 50 --batch-size 128
```

Resume training from checkpoint:
```bash
python scripts/train.py --data data/lichess_db_puzzle.csv --resume models/checkpoint_epoch_10.pt
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --model models/best_model.pt --data data/lichess_db_puzzle.csv
```

Evaluate with difficulty analysis:
```bash
python scripts/evaluate.py --model models/best_model.pt --data data/lichess_db_puzzle.csv --by-difficulty
```

## Configuration

The system uses a configuration class defined in `src/config.py`. Key parameters include:

- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `policy_lr`: Learning rate for policy network
- `value_lr`: Learning rate for value network
- `stockfish_depth`: Depth for Stockfish analysis
- `clip_epsilon`: PPO clipping parameter
- `correct_move_bonus`: Reward bonus for correct puzzle moves

## Model Architecture

### Policy Network
- Multi-layer neural network with attention mechanism
- Takes board features and legal move features as input
- Outputs probability distribution over legal moves

### Value Network
- Estimates the value of the current position
- Used for advantage estimation in PPO

### Feature Extraction
- 8x8x12 piece placement matrix
- Attack and defense maps
- Material count
- Castling rights
- King safety features
- Pawn structure evaluation
- Piece mobility
- Center control

## Evaluation Metrics

The model is evaluated on:
- Puzzle solving accuracy
- Average reward per puzzle
- Agreement with Stockfish recommendations
- Performance by puzzle difficulty rating

## Visualizations

The evaluation script generates:
- Training metrics plots (accuracy, loss, entropy)
- Performance by difficulty rating
- Puzzle distribution analysis

## Project Structure

```
chess-puzzle-solver/
├── src/
│   ├── __init__.py
│   ├── board_manager.py      # Chess board operations
│   ├── feature_extractor.py  # Feature extraction from positions
│   ├── networks.py          # Neural network architectures
│   ├── stockfish_evaluator.py # Stockfish integration
│   ├── rl_components.py     # RL components (PPO, replay buffer)
│   ├── puzzle_trainer.py    # Main training logic
│   └── config.py           # Configuration settings
├── scripts/
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── data/                   # Puzzle datasets
├── models/                 # Saved models
├── requirements.txt
├── setup.py
└── README.md
```

## Future Enhancements

1. **Multi-task Learning**: Train on both puzzles and regular games
2. **Curriculum Learning**: Progressive difficulty increase
3. **Tree Search**: Implement MCTS for deeper tactical analysis
4. **Self-play**: Generate additional training data
5. **Explainable AI**: Attention visualization for move selection

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Lichess.org for the puzzle database
- Stockfish team for the chess engine
- PyTorch team for the deep learning framework
- python-chess library authors

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{chess-puzzle-solver,
  author = {Your Name},
  title = {Chess Puzzle Solver using Reinforcement Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/chess-puzzle-solver}
}
```