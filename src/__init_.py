"""
Chess Puzzle Solver using Reinforcement Learning
"""

from .board_manager import BoardManager
from .feature_extractor import FeatureExtractor
from .networks import PolicyNetwork, ValueNetwork
from .stockfish_evaluator import StockfishEvaluator
from .rl_components import PPOTrainer, ReplayBuffer
from .puzzle_trainer import PuzzleTrainer
from .config import TrainingConfig, EvaluationConfig

__version__ = "0.1.0"

__all__ = [
    "BoardManager",
    "FeatureExtractor",
    "PolicyNetwork",
    "ValueNetwork",
    "StockfishEvaluator",
    "PPOTrainer",
    "ReplayBuffer",
    "PuzzleTrainer",
    "TrainingConfig",
    "EvaluationConfig",
]