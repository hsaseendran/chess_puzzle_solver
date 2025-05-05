#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import torch
from torch.utils.data import Subset
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ChessPuzzleNet
from src.dataset import ChessPuzzleDataset
from src.trainer import PuzzleTrainer


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def main():
    parser = argparse.ArgumentParser(description='Train Chess Puzzle Solver')
    parser.add_argument('--data', type=str, required=True, help='Path to puzzle CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-blocks', type=int, default=20, help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=256, help='Number of channels')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test data ratio')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='Device to use')
    parser.add_argument('--themes', type=str, nargs='+', help='Filter puzzles by themes')
    parser.add_argument('--max-puzzles', type=int, help='Maximum number of puzzles to use')
    parser.add_argument('--split-file', type=str, help='Load data splits from file')
    parser.add_argument('--save-splits', action='store_true', help='Save data split indices')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        parser.error("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directories
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(log_dir)
    
    # Device selection
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    logging.info(f"Using device: {device}")
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
    elif device == 'mps':
        logging.info("Using Apple Silicon GPU (MPS)")
    else:
        logging.info("Using CPU")
    
    # Load dataset
    logging.info(f"Loading dataset from {args.data}")
    dataset = ChessPuzzleDataset(args.data, filter_themes=args.themes, max_puzzles=args.max_puzzles)
    logging.info(f"Total puzzles: {len(dataset)}")
    
    if args.themes:
        logging.info(f"Filtered by themes: {args.themes}")
    
    # Create or load data splits
    if args.split_file and os.path.exists(args.split_file):
        logging.info(f"Loading data splits from {args.split_file}")
        split_data = ChessPuzzleDataset.load_split_indices(args.split_file)
        train_dataset = Subset(dataset, split_data['train_indices'])
        val_dataset = Subset(dataset, split_data['val_indices'])
        test_dataset = Subset(dataset, split_data['test_indices'])
    else:
        logging.info("Creating new data splits")
        train_dataset, val_dataset, test_dataset = ChessPuzzleDataset.create_data_splits(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Save splits if requested
        if args.save_splits:
            splits_file = os.path.join(args.output_dir, 'data_splits.pth')
            ChessPuzzleDataset.save_split_indices(
                train_dataset.indices,
                val_dataset.indices,
                test_dataset.indices,
                splits_file
            )
            logging.info(f"Saved data splits to {splits_file}")
    
    logging.info(f"Train size: {len(train_dataset)}")
    logging.info(f"Validation size: {len(val_dataset)}")
    logging.info(f"Test size: {len(test_dataset)} (not used during training)")
    
    # Create model
    model = ChessPuzzleNet(num_blocks=args.num_blocks, channels=args.channels)
    logging.info(f"Model created with {args.num_blocks} residual blocks and {args.channels} channels")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = PuzzleTrainer(model, device=device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Training configuration
    config = {
        'data_file': args.data,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_blocks': args.num_blocks,
        'channels': args.channels,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'device': device,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'themes': args.themes,
        'log_file': log_file,
        'checkpoint_dir': checkpoint_dir
    }
    
    # Save configuration
    config_file = os.path.join(args.output_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("Starting training...")
    logging.info("Note: Test set is reserved for final evaluation only")
    
    try:
        # Train the model (using only train and validation sets)
        metrics = trainer.train(
            train_dataset,
            val_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=checkpoint_dir
        )
        
        # Save final metrics
        metrics_file = os.path.join(args.output_dir, 'final_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info("Training completed successfully!")
        logging.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
        logging.info(f"Checkpoints saved to: {checkpoint_dir}")
        logging.info(f"Logs saved to: {log_dir}")
        logging.info(f"Use evaluate.py with --use-test flag to evaluate on the test set")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        
        # Save interrupted checkpoint
        interrupt_checkpoint = os.path.join(checkpoint_dir, 'interrupted_checkpoint.pth')
        trainer.save_checkpoint(interrupt_checkpoint, start_epoch + 1)
        logging.info(f"Interrupted checkpoint saved to: {interrupt_checkpoint}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()