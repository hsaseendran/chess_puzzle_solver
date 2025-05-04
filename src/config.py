from dataclasses import dataclass
import torch
import os


@dataclass
class TrainingConfig:
    """Configuration for training the chess puzzle solver"""
    
    # Model architecture
    board_feature_dim: int = 832  # Based on feature extractor
    move_feature_dim: int = 16    # Based on feature extractor
    
    # Training parameters
    batch_size: int = 64
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    
    # PPO parameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Stockfish parameters
    stockfish_path: str = "/usr/games/stockfish"  # Default Linux path
    stockfish_depth: int = 15
    
    # Training settings
    num_epochs: int = 100
    save_frequency: int = 5
    log_frequency: int = 10
    eval_frequency: int = 100
    
    # Dataset settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Buffer settings
    replay_buffer_capacity: int = 100000
    min_buffer_size: int = 1000
    
    # Reward settings
    correct_move_bonus: float = 1.0
    stockfish_weight: float = 0.5
    mate_bonus: float = 2.0
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Adjust Stockfish path based on OS
        if os.name == 'nt':  # Windows
            self.stockfish_path = "C:/Program Files/Stockfish/stockfish.exe"
        elif os.name == 'posix' and os.uname().sysname == 'Darwin':  # macOS
            self.stockfish_path = "/usr/local/bin/stockfish"
        
        # Verify Stockfish path exists
        if not os.path.exists(self.stockfish_path):
            print(f"Warning: Stockfish not found at {self.stockfish_path}")
            print("Please update the stockfish_path in config.py")


@dataclass
class EvaluationConfig:
    """Configuration for evaluating the model"""
    
    # Evaluation settings
    num_puzzles: int = 1000
    timeout_seconds: int = 30
    max_attempts: int = 3
    
    # Difficulty-based evaluation
    evaluate_by_rating: bool = True
    rating_buckets: list = None
    
    def __post_init__(self):
        if self.rating_buckets is None:
            self.rating_buckets = [
                (0, 1000),
                (1000, 1500),
                (1500, 2000),
                (2000, 2500),
                (2500, 3000)
            ]


def get_config(config_type: str = "training") -> TrainingConfig:
    """Get configuration object"""
    if config_type == "training":
        return TrainingConfig()
    elif config_type == "evaluation":
        return EvaluationConfig()
    else:
        raise ValueError(f"Unknown config type: {config_type}")