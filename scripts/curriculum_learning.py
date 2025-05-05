#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ChessPuzzleNet
from src.dataset import ChessPuzzleDataset
from src.trainer import PuzzleTrainer

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'curriculum_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

class CurriculumTrainer(PuzzleTrainer):
    """Extended trainer with support for curriculum learning on mate-in-X puzzles"""
    
    def train_epoch_mixed(self, puzzle_loaders, epoch, weights=None):
        """
        Train one epoch with a mix of puzzle types
        
        Args:
            puzzle_loaders: Dict mapping puzzle types (e.g., 'mateIn1') to DataLoaders
            epoch: Current epoch number
            weights: Dict mapping puzzle types to sampling weights (normalized internally)
        """
        self.model.train()
        total_policy_loss = 0
        total_value_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        # Default to equal weights if not provided
        if weights is None:
            weights = {puzzle_type: 1.0 for puzzle_type in puzzle_loaders.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Get puzzle types and create iterators
        puzzle_types = list(puzzle_loaders.keys())
        iterators = {puzzle_type: iter(loader) for puzzle_type, loader in puzzle_loaders.items()}
        
        # Find minimum number of batches across all loaders
        min_batches = min(len(loader) for loader in puzzle_loaders.values())
        
        # Create progress bar
        progress_bar = tqdm(range(min_batches), desc=f"Mixed Training Epoch {epoch}")
        
        for _ in progress_bar:
            # Choose puzzle type based on weights
            puzzle_type = np.random.choice(
                puzzle_types, 
                p=[weights[ptype] for ptype in puzzle_types]
            )
            
            # Get batch from chosen loader
            try:
                batch = next(iterators[puzzle_type])
            except StopIteration:
                # Restart iterator if exhausted
                iterators[puzzle_type] = iter(puzzle_loaders[puzzle_type])
                batch = next(iterators[puzzle_type])
            
            # Unpack batch
            board_tensor, policy_target, value_target, _ = batch
            
            # Move to device
            board_tensor = board_tensor.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device)
            
            # Forward pass
            policy_output, value_output = self.model(board_tensor)
            
            # Calculate losses
            policy_loss = self.policy_criterion(policy_output, policy_target)
            value_loss = self.value_criterion(value_output, value_target)
            
            # Combined loss (value loss weighted less)
            loss = policy_loss + 0.1 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(policy_output, 1)
            _, target = torch.max(policy_target, 1)
            batch_correct = (predicted == target).sum().item()
            correct_predictions += batch_correct
            total_samples += board_tensor.size(0)
            
            # Update metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'policy_loss': f'{policy_loss.item():.4f}',
                'value_loss': f'{value_loss.item():.4f}',
                'accuracy': f'{batch_correct / board_tensor.size(0):.4f}',
                'type': puzzle_type
            })
        
        epoch_policy_loss = total_policy_loss / min_batches
        epoch_value_loss = total_value_loss / min_batches
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_policy_loss, epoch_value_loss, epoch_accuracy

def curriculum_training(args):
    """
    Curriculum learning approach for chess puzzles of increasing complexity
    Supports arbitrary number of stages with configurable puzzle types
    """
    # Parse curriculum stages
    curriculum = parse_curriculum(args.curriculum)
    
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
    
    # Load all datasets
    puzzle_data = {}
    all_puzzle_types = set()
    for stage in curriculum:
        all_puzzle_types.update(stage['puzzle_types'].keys())
    
    for puzzle_type in all_puzzle_types:
        csv_path = getattr(args, f"{puzzle_type}_data", None)
        if csv_path is None:
            raise ValueError(f"No data file specified for {puzzle_type}. "
                           f"Please provide --{puzzle_type}-data argument.")
        
        logging.info(f"Loading {puzzle_type} dataset from {csv_path}")
        dataset = ChessPuzzleDataset(csv_path, filter_themes=[puzzle_type], max_puzzles=args.max_puzzles)
        logging.info(f"Total {puzzle_type} puzzles: {len(dataset)}")
        
        # Create train/val/test splits
        train_dataset, val_dataset, test_dataset = ChessPuzzleDataset.create_data_splits(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        logging.info(f"{puzzle_type} - Train: {len(train_dataset)}, "
                    f"Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        puzzle_data[puzzle_type] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'csv_path': csv_path
        }
    
    # Create or load model
    if args.resume:
        logging.info(f"Loading existing model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Detect architecture
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Detect parameters
        channels = 256
        num_blocks = 20
        
        conv_input_weight = state_dict.get('conv_input.weight', None)
        if conv_input_weight is not None:
            channels = conv_input_weight.shape[0]
        
        for key in state_dict.keys():
            if key.startswith('residual_blocks.') and key.endswith('.conv1.weight'):
                block_num = int(key.split('.')[1])
                num_blocks = max(num_blocks, block_num + 1)
        
        logging.info(f"Detected architecture: {channels} channels, {num_blocks} blocks")
        
        model = ChessPuzzleNet(num_blocks=num_blocks, channels=channels)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logging.info(f"Creating new model with {args.num_blocks} blocks and {args.channels} channels")
        model = ChessPuzzleNet(num_blocks=args.num_blocks, channels=args.channels)
    
    # Create trainer
    trainer = CurriculumTrainer(model, device=device)
    
    # Track metrics for each stage
    all_metrics = []
    
    # Go through curriculum stages
    for stage_idx, stage in enumerate(curriculum):
        stage_name = stage['name']
        logging.info(f"\n=== STAGE {stage_idx+1}: {stage_name} ===")
        
        # Configure learning rate
        lr = stage.get('lr', 0.001)
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.optimizer, 
            step_size=5, 
            gamma=0.5
        )
        logging.info(f"Learning rate set to {lr}")
        
        # Get puzzle types and weights for this stage
        puzzle_types = stage['puzzle_types']
        logging.info(f"Training on puzzle types: {puzzle_types}")
        
        # Create data loaders
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        
        for puzzle_type, weight in puzzle_types.items():
            train_loaders[puzzle_type] = DataLoader(
                puzzle_data[puzzle_type]['train'],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            val_loaders[puzzle_type] = DataLoader(
                puzzle_data[puzzle_type]['val'],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            test_loaders[puzzle_type] = DataLoader(
                puzzle_data[puzzle_type]['test'],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # For stages with single puzzle type, use regular training
        if len(puzzle_types) == 1:
            puzzle_type = list(puzzle_types.keys())[0]
            logging.info(f"Single puzzle type ({puzzle_type}), using standard training")
            
            # Train for specified epochs
            for epoch in range(stage['epochs']):
                # Train one epoch
                policy_loss, value_loss, train_accuracy = trainer.train_epoch(
                    train_loaders[puzzle_type], 
                    epoch + 1
                )
                
                # Validate
                val_accuracy = trainer.validate(val_loaders[puzzle_type], epoch + 1)
                
                # Logging
                logging.info(f"Epoch {epoch+1}/{stage['epochs']} (Stage {stage_idx+1}: {stage_name})")
                logging.info(f"Train - Policy Loss: {policy_loss:.4f}, "
                           f"Value Loss: {value_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                logging.info(f"Validation - Accuracy: {val_accuracy:.4f}")
                
                # Learning rate scheduling
                trainer.scheduler.step()
                
                # Save checkpoint periodically
                if (epoch + 1) % 5 == 0 or epoch == stage['epochs'] - 1:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 
                        f"stage{stage_idx+1}_{stage_name}_epoch_{epoch+1}.pth"
                    )
                    trainer.save_checkpoint(checkpoint_path, epoch + 1)
        
        else:
            # Multiple puzzle types, use mixed training
            logging.info(f"Multiple puzzle types, using mixed training with weights: {puzzle_types}")
            
            # Create combined validation loader
            all_val_datasets = [data['val'] for data in puzzle_data.values()]
            combined_val_dataset = ConcatDataset(all_val_datasets)
            combined_val_loader = DataLoader(
                combined_val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=4
            )
            
            # Train for specified epochs
            for epoch in range(stage['epochs']):
                # Train with mixed batches
                policy_loss, value_loss, train_accuracy = trainer.train_epoch_mixed(
                    train_loaders,
                    epoch + 1,
                    weights=puzzle_types
                )
                
                # Validate on each puzzle type
                val_accuracies = {}
                for puzzle_type in puzzle_types:
                    val_accuracies[puzzle_type] = trainer.validate(
                        val_loaders[puzzle_type], 
                        epoch + 1
                    )
                
                # Combined validation
                combined_val_accuracy = trainer.validate(combined_val_loader, epoch + 1)
                
                # Logging
                logging.info(f"Epoch {epoch+1}/{stage['epochs']} (Stage {stage_idx+1}: {stage_name})")
                logging.info(f"Train - Policy Loss: {policy_loss:.4f}, "
                           f"Value Loss: {value_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                
                for puzzle_type, accuracy in val_accuracies.items():
                    logging.info(f"Validation - {puzzle_type}: {accuracy:.4f}")
                
                logging.info(f"Validation - Combined: {combined_val_accuracy:.4f}")
                
                # Learning rate scheduling
                trainer.scheduler.step()
                
                # Save checkpoint periodically
                if (epoch + 1) % 5 == 0 or epoch == stage['epochs'] - 1:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 
                        f"stage{stage_idx+1}_{stage_name}_epoch_{epoch+1}.pth"
                    )
                    trainer.save_checkpoint(checkpoint_path, epoch + 1)
        
        # Save stage final model
        stage_model_path = os.path.join(checkpoint_dir, f"stage{stage_idx+1}_{stage_name}_final.pth")
        trainer.save_checkpoint(stage_model_path, stage['epochs'])
        logging.info(f"Stage {stage_idx+1} model saved to {stage_model_path}")
        
        # Evaluate on all puzzle types
        stage_metrics = {
            'stage_name': stage_name,
            'stage_index': stage_idx + 1,
            'accuracies': {}
        }
        
        logging.info(f"Evaluating Stage {stage_idx+1} model on all puzzle types:")
        for puzzle_type in all_puzzle_types:
            test_accuracy = trainer.validate(test_loaders[puzzle_type], 0)
            logging.info(f"  {puzzle_type} Test Accuracy: {test_accuracy:.4f}")
            stage_metrics['accuracies'][puzzle_type] = test_accuracy
        
        all_metrics.append(stage_metrics)
    
    # Save final summary
    summary = {
        'curriculum': curriculum,
        'stage_metrics': all_metrics,
        'final_accuracies': all_metrics[-1]['accuracies']
    }
    
    summary_file = os.path.join(args.output_dir, 'curriculum_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\nFinal Accuracies:")
    for puzzle_type, accuracy in all_metrics[-1]['accuracies'].items():
        logging.info(f"  {puzzle_type}: {accuracy:.4f}")
    
    logging.info(f"\nSummary saved to {summary_file}")
    logging.info("Curriculum training completed!")
    
    return {
        'final_model': stage_model_path,
        'summary': summary
    }

def parse_curriculum(curriculum_str):
    """
    Parse curriculum string into structured format
    Format: stage1_name:puzzle_type1=weight1,puzzle_type2=weight2:epochs:lr;stage2_name:...
    
    Example: 
    "mateIn1:mateIn1=1:20:0.001;mateIn2:mateIn1=0.3,mateIn2=0.7:30:0.0005;all:mateIn1=0.2,mateIn2=0.3,mateIn3=0.5:20:0.0001"
    """
    stages = []
    
    for stage_str in curriculum_str.split(';'):
        parts = stage_str.strip().split(':')
        if len(parts) < 3:
            raise ValueError(f"Invalid stage format: {stage_str}. "
                           f"Expected format: name:puzzle_types:epochs[:lr]")
        
        stage_name = parts[0]
        puzzle_types_str = parts[1]
        epochs = int(parts[2])
        lr = float(parts[3]) if len(parts) > 3 else 0.001
        
        # Parse puzzle types and weights
        puzzle_types = {}
        for type_weight in puzzle_types_str.split(','):
            if '=' in type_weight:
                ptype, weight = type_weight.split('=')
                puzzle_types[ptype] = float(weight)
            else:
                puzzle_types[type_weight] = 1.0
        
        stages.append({
            'name': stage_name,
            'puzzle_types': puzzle_types,
            'epochs': epochs,
            'lr': lr
        })
    
    return stages

def main():
    parser = argparse.ArgumentParser(description='Curriculum Learning for Chess Puzzles')
    parser.add_argument('--curriculum', type=str, required=True, 
                      help=('Curriculum stages in format: '
                            'stage1_name:puzzle_type1=weight1,puzzle_type2=weight2:epochs:lr;'
                            'stage2_name:...'))
    
    # Add data file arguments for each potential puzzle type
    for puzzle_type in ['mateIn1', 'mateIn2', 'mateIn3', 'mateIn4', 'mateIn5']:
        parser.add_argument(f'--{puzzle_type}-data', type=str,
                          help=f'Path to {puzzle_type} puzzles CSV')
    
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-blocks', type=int, default=20, help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=256, help='Number of channels')
    parser.add_argument('--output-dir', type=str, default='curriculum_output', help='Output directory')
    parser.add_argument('--resume', type=str, help='Resume from existing model checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='Device to use')
    parser.add_argument('--max-puzzles', type=int, help='Maximum number of puzzles per category')
    
    args = parser.parse_args()
    
    try:
        curriculum_training(args)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()