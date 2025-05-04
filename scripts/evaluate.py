#!/usr/bin/env python3
"""
Evaluate the chess puzzle solver model
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import TrainingConfig, EvaluationConfig
from src.puzzle_trainer import PuzzleTrainer


def analyze_by_difficulty(results_df: pd.DataFrame, config: EvaluationConfig):
    """Analyze results by puzzle difficulty rating"""
    results_by_rating = defaultdict(list)
    
    # Group results by rating buckets
    for _, row in results_df.iterrows():
        rating = row.get('Rating', 1500)  # Default rating if not present
        for min_rating, max_rating in config.rating_buckets:
            if min_rating <= rating < max_rating:
                results_by_rating[f"{min_rating}-{max_rating}"].append(row)
                break
    
    # Calculate metrics for each bucket
    bucket_metrics = {}
    for bucket, results in results_by_rating.items():
        if results:
            df_bucket = pd.DataFrame(results)
            bucket_metrics[bucket] = {
                'count': len(results),
                'accuracy': df_bucket['is_correct'].mean(),
                'average_reward': df_bucket['reward'].mean(),
                'stockfish_agreement': df_bucket['stockfish_agreement'].mean()
            }
    
    return bucket_metrics


def plot_metrics(metrics_history: dict, save_path: str):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy and stockfish agreement
    plt.subplot(2, 2, 1)
    if 'puzzle_accuracy' in metrics_history:
        plt.plot(metrics_history['puzzle_accuracy'], label='Puzzle Accuracy')
    if 'stockfish_agreement' in metrics_history:
        plt.plot(metrics_history['stockfish_agreement'], label='Stockfish Agreement')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot average reward
    plt.subplot(2, 2, 2)
    if 'average_reward' in metrics_history:
        plt.plot(metrics_history['average_reward'])
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Epoch')
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 2, 3)
    if 'policy_loss' in metrics_history:
        plt.plot(metrics_history['policy_loss'], label='Policy Loss')
    if 'value_loss' in metrics_history:
        plt.plot(metrics_history['value_loss'], label='Value Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot entropy
    plt.subplot(2, 2, 4)
    if 'entropy' in metrics_history:
        plt.plot(metrics_history['entropy'])
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_difficulty_analysis(bucket_metrics: dict, save_path: str):
    """Plot analysis by difficulty rating"""
    buckets = list(bucket_metrics.keys())
    accuracies = [bucket_metrics[b]['accuracy'] for b in buckets]
    stockfish_agreements = [bucket_metrics[b]['stockfish_agreement'] for b in buckets]
    counts = [bucket_metrics[b]['count'] for b in buckets]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot accuracies
    x = range(len(buckets))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], accuracies, width, label='Puzzle Accuracy', alpha=0.8)
    ax1.bar([i + width/2 for i in x], stockfish_agreements, width, label='Stockfish Agreement', alpha=0.8)
    
    ax1.set_xlabel('Rating Bucket')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Puzzle Difficulty')
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot puzzle counts
    ax2.bar(x, counts, alpha=0.8, color='green')
    ax2.set_xlabel('Rating Bucket')
    ax2.set_ylabel('Number of Puzzles')
    ax2.set_title('Puzzle Distribution by Rating')
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess puzzle solver")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to puzzle CSV file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--num-puzzles", type=int, help="Number of puzzles to evaluate")
    parser.add_argument("--by-difficulty", action="store_true", help="Analyze results by difficulty rating")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'evaluation_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting model evaluation")
    
    # Load configurations
    train_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    if args.num_puzzles:
        eval_config.num_puzzles = args.num_puzzles
    
    # Initialize trainer and load model
    trainer = PuzzleTrainer(train_config)
    trainer.load_checkpoint(args.model)
    
    # Load evaluation dataset
    df = pd.read_csv(args.data)
    df = df.dropna(subset=['FEN', 'Moves'])
    
    # Sample puzzles for evaluation
    if len(df) > eval_config.num_puzzles:
        df = df.sample(n=eval_config.num_puzzles, random_state=42)
    
    puzzles = df.to_dict('records')
    logging.info(f"Evaluating on {len(puzzles)} puzzles")
    
    # Evaluate model
    trainer.policy_network.eval()
    trainer.value_network.eval()
    
    results = []
    for i, puzzle in enumerate(puzzles):
        is_correct, reward, puzzle_stats = trainer.train_on_puzzle(puzzle)
        
        result = {
            'puzzle_id': puzzle.get('PuzzleId', i),
            'rating': puzzle.get('Rating', 1500),
            'themes': puzzle.get('Themes', ''),
            'is_correct': is_correct,
            'reward': reward,
            'stockfish_agreement': puzzle_stats.get('stockfish_agreement', False),
            'selected_move': puzzle_stats.get('selected_move', ''),
            'correct_move': puzzle_stats.get('correct_move', ''),
            'stockfish_best': puzzle_stats.get('stockfish_best', '')
        }
        results.append(result)
        
        if (i + 1) % 100 == 0:
            logging.info(f"Evaluated {i + 1}/{len(puzzles)} puzzles")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    overall_metrics = {
        'total_puzzles': len(results_df),
        'accuracy': results_df['is_correct'].mean(),
        'average_reward': results_df['reward'].mean(),
        'stockfish_agreement': results_df['stockfish_agreement'].mean()
    }
    
    # Analyze by difficulty if requested
    difficulty_metrics = None
    if args.by_difficulty and 'Rating' in results_df.columns:
        difficulty_metrics = analyze_by_difficulty(results_df, eval_config)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, f'evaluation_results_{timestamp}.csv'), index=False)
    
    # Save metrics
    metrics_output = {
        'overall': overall_metrics,
        'by_difficulty': difficulty_metrics,
        'timestamp': timestamp,
        'model_path': args.model,
        'num_puzzles': len(puzzles)
    }
    
    with open(os.path.join(args.output_dir, f'evaluation_metrics_{timestamp}.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    # Create visualizations
    if hasattr(trainer, 'metrics') and trainer.metrics:
        plot_metrics(trainer.metrics, os.path.join(args.output_dir, f'training_metrics_{timestamp}.png'))
    
    if difficulty_metrics:
        plot_difficulty_analysis(difficulty_metrics, os.path.join(args.output_dir, f'difficulty_analysis_{timestamp}.png'))
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info("EVALUATION SUMMARY")
    logging.info("="*50)
    logging.info(f"Total Puzzles: {overall_metrics['total_puzzles']}")
    logging.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    logging.info(f"Average Reward: {overall_metrics['average_reward']:.4f}")
    logging.info(f"Stockfish Agreement: {overall_metrics['stockfish_agreement']:.4f}")
    
    if difficulty_metrics:
        logging.info("\nPerformance by Difficulty:")
        for bucket, metrics in difficulty_metrics.items():
            logging.info(f"  {bucket}: Accuracy={metrics['accuracy']:.4f}, Count={metrics['count']}")
    
    logging.info("="*50)
    logging.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()