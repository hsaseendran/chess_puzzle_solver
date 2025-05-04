#!/usr/bin/env python3
"""
Train the chess puzzle solver model
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# Fix import path - add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now we can import from src
from src.config import TrainingConfig
from src.puzzle_trainer import PuzzleTrainer


def load_puzzle_dataset(csv_path: str, config: TrainingConfig):
    """Load and preprocess puzzle dataset"""
    logging.info(f"Loading puzzle dataset from {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out puzzles with missing data
    df = df.dropna(subset=['FEN', 'Moves'])
    
    # Convert to list of dictionaries
    puzzles = df.to_dict('records')
    
    # Shuffle the dataset
    np.random.shuffle(puzzles)
    
    # Split into train/val/test
    n_puzzles = len(puzzles)
    train_size = int(n_puzzles * config.train_split)
    val_size = int(n_puzzles * config.val_split)
    
    train_puzzles = puzzles[:train_size]
    val_puzzles = puzzles[train_size:train_size + val_size]
    test_puzzles = puzzles[train_size + val_size:]
    
    logging.info(f"Dataset split: Train={len(train_puzzles)}, Val={len(val_puzzles)}, Test={len(test_puzzles)}")
    
    return train_puzzles, val_puzzles, test_puzzles


def main():
    parser = argparse.ArgumentParser(description="Train chess puzzle solver")
    parser.add_argument("--data", type=str, required=True, help="Path to puzzle CSV file")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--num-epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--stockfish-path", type=str, help="Override Stockfish path")
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    
    # Override config with command line arguments
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.stockfish_path:
        config.stockfish_path = args.stockfish_path
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting chess puzzle solver training")
    logging.info(f"Configuration: {config}")
    
    # Load dataset
    train_puzzles, val_puzzles, test_puzzles = load_puzzle_dataset(args.data, config)
    
    # Initialize trainer
    trainer = PuzzleTrainer(config)
    
    # Load checkpoint if resuming
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    try:
        trainer.train(train_puzzles, val_puzzles)
        
        # Final evaluation on test set
        logging.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_puzzles)
        
        logging.info(f"Test Results:")
        logging.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"  Average Reward: {test_metrics['average_reward']:.4f}")
        logging.info(f"  Stockfish Agreement: {test_metrics['stockfish_agreement']:.4f}")
        
        # Save final model
        final_model_path = os.path.join(config.model_dir, f'final_model_{timestamp}.pt')
        trainer.save_checkpoint(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        
        # Save test results
        results_path = os.path.join(config.log_dir, f'test_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        
        # Save interrupt checkpoint
        interrupt_path = os.path.join(config.model_dir, f'interrupt_checkpoint_{timestamp}.pt')
        trainer.save_checkpoint(interrupt_path)
        logging.info(f"Interrupt checkpoint saved to {interrupt_path}")
    
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()