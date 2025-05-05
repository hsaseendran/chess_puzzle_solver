"""
Chess Puzzle Solver - Proven Architecture
"""

from .model import ChessPuzzleNet, ResidualBlock
from .dataset import ChessPuzzleDataset
from .trainer import PuzzleTrainer
from .utils import (
    board_to_fen,
    visualize_board_tensor,
    get_move_san,
    calculate_material_balance,
    is_tactical_position,
    analyze_puzzle_difficulty,
    create_augmented_positions,
    puzzle_complexity_score,
    create_curriculum_batches,
    validate_puzzle_data,
    log_gpu_memory,
    setup_deterministic_training
)

__version__ = "1.0.0"
__author__ = "Chess AI Team"

__all__ = [
    "ChessPuzzleNet",
    "ResidualBlock",
    "ChessPuzzleDataset",
    "PuzzleTrainer",
    "board_to_fen",
    "visualize_board_tensor",
    "get_move_san",
    "calculate_material_balance",
    "is_tactical_position",
    "analyze_puzzle_difficulty",
    "create_augmented_positions",
    "puzzle_complexity_score",
    "create_curriculum_batches",
    "validate_puzzle_data",
    "log_gpu_memory",
    "setup_deterministic_training"
]