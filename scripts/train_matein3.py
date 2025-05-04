#!/usr/bin/env python3
"""
Train the chess puzzle solver model on mateIn3 puzzles only
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# Fix import path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.config import TrainingConfig
from src.puzzle_trainer import PuzzleTrainer


def is_mate_in_3(puzzle_data):
    """Check if a puzzle has the mateIn3 tag"""
    themes = puzzle_data.get('Themes', '')
    if pd.isna(themes):
        return False
    
    # Split themes and check for mateIn3
    theme_list = themes.split()
    return 'mateIn3' in theme_list


def filter_mate_in_3_puzzles(puzzles):
    """Filter puzzles to only include mateIn3 puzzles"""
    mate_in_3_puzzles = []
    
    for puzzle in puzzles:
        if is_mate_in_3(puzzle):
            mate_in_3_puzzles.append(puzzle)
    
    return mate_in_3_puzzles


def load_puzzle_dataset(csv_path: str, config: TrainingConfig):
    """Load and preprocess puzzle dataset for mateIn3 puzzles only"""
    logging.info(f"Loading puzzle dataset from {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out puzzles with missing data
    df = df.dropna(subset=['FEN', 'Moves'])
    
    # Convert to list of dictionaries
    all_puzzles = df.to_dict('records')
    
    # Filter for mateIn3 puzzles
    logging.info("Filtering for mateIn3 puzzles...")
    mate_in_3_puzzles = filter_mate_in_3_puzzles(all_puzzles)
    
    logging.info(f"Found {len(mate_in_3_puzzles)} mateIn3 puzzles out of {len(all_puzzles)} total puzzles")
    
    if len(mate_in_3_puzzles) == 0:
        raise ValueError("No mateIn3 puzzles found in the dataset!")
    
    # Shuffle the dataset
    np.random.shuffle(mate_in_3_puzzles)
    
    # Split into train/val/test
    n_puzzles = len(mate_in_3_puzzles)
    train_size = int(n_puzzles * config.train_split)
    val_size = int(n_puzzles * config.val_split)
    
    train_puzzles = mate_in_3_puzzles[:train_size]
    val_puzzles = mate_in_3_puzzles[train_size:train_size + val_size]
    test_puzzles = mate_in_3_puzzles[train_size + val_size:]
    
    logging.info(f"MateIn3 dataset split: Train={len(train_puzzles)}, Val={len(val_puzzles)}, Test={len(test_puzzles)}")
    
    return train_puzzles, val_puzzles, test_puzzles


def main():
    parser = argparse.ArgumentParser(description="Train chess puzzle solver on mateIn3 puzzles")
    parser.add_argument("--data", type=str, required=True, help="Path to puzzle CSV file")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--stockfish-path", type=str, help="Override Stockfish path")
    parser.add_argument("--output-dir", type=str, default="matein3_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = TrainingConfig()
    
    # Override config for faster testing
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.stockfish_depth = 10  # Reduce depth for faster evaluation
    config.save_frequency = 2
    config.eval_frequency = 1
    
    if args.stockfish_path:
        config.stockfish_path = args.stockfish_path
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'matein3_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting mateIn3 puzzle solver training")
    logging.info(f"Configuration: {config}")
    
    try:
        # Load dataset
        train_puzzles, val_puzzles, test_puzzles = load_puzzle_dataset(args.data, config)
        
        # Show some example puzzles
        logging.info("\nExample mateIn3 puzzles:")
        for i, puzzle in enumerate(train_puzzles[:5]):
            logging.info(f"Puzzle {i+1}:")
            logging.info(f"  FEN: {puzzle['FEN']}")
            logging.info(f"  Moves: {puzzle['Moves']}")
            logging.info(f"  Rating: {puzzle.get('Rating', 'N/A')}")
            logging.info(f"  Themes: {puzzle.get('Themes', 'N/A')}")
        
        # Initialize trainer
        trainer = PuzzleTrainer(config)
        
        # Train model
        trainer.train(train_puzzles, val_puzzles)
        
        # Final evaluation on test set
        logging.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_puzzles)
        
        logging.info(f"Test Results:")
        logging.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"  Average Reward: {test_metrics['average_reward']:.4f}")
        logging.info(f"  Stockfish Agreement: {test_metrics['stockfish_agreement']:.4f}")
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, f'matein3_final_model_{timestamp}.pt')
        trainer.save_checkpoint(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        
        # Save test results
        results_path = os.path.join(args.output_dir, f'matein3_test_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump({
                'test_metrics': test_metrics,
                'num_puzzles': {
                    'train': len(train_puzzles),
                    'val': len(val_puzzles),
                    'test': len(test_puzzles)
                },
                'config': {
                    'num_epochs': config.num_epochs,
                    'batch_size': config.batch_size,
                    'stockfish_depth': config.stockfish_depth
                }
            }, f, indent=2)
        
        logging.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()