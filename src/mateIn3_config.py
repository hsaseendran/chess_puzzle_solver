from dataclasses import dataclass
import torch
import os
from src.config import TrainingConfig


@dataclass
class MateIn3Config(TrainingConfig):
    """Specialized configuration for mateIn3 puzzle training"""
    
    # Override defaults for mateIn3 training
    batch_size: int = 32  # Smaller batch size for quicker iterations
    num_epochs: int = 20  # More epochs since we have fewer puzzles
    stockfish_depth: int = 10  # Reduced depth since mates are forced
    
    # Reward settings optimized for mate puzzles
    correct_move_bonus: float = 2.0  # Higher bonus for finding the mate
    stockfish_weight: float = 0.3  # Less weight on Stockfish eval
    mate_bonus: float = 3.0  # High bonus for achieving checkmate
    
    # PPO parameters adjusted for mate training
    clip_epsilon: float = 0.1  # Smaller clip range for more stable training
    entropy_coef: float = 0.005  # Less exploration needed for forced mates
    
    # Training settings
    save_frequency: int = 2
    eval_frequency: int = 50  # More frequent evaluation
    min_buffer_size: int = 500  # Smaller buffer requirement
    
    # Feature extractor settings
    use_fast_features: bool = True  # Use simplified features for mates
    
    def __post_init__(self):
        super().__post_init__()
        # Create specific directories for mateIn3
        self.model_dir = os.path.join(self.model_dir, "matein3")
        self.log_dir = os.path.join(self.log_dir, "matein3")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)