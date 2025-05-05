#!/usr/bin/env python3

import argparse
import torch
import logging
import pandas as pd
from torch.utils.data import Subset
from tqdm import tqdm
import chess
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ChessPuzzleNet
from src.dataset import ChessPuzzleDataset


def evaluate_puzzle(model, board_tensor, solution_move, legal_moves, device):
    """Evaluate a single puzzle"""
    with torch.no_grad():
        board_tensor = board_tensor.unsqueeze(0).to(device)
        policy_output, value_output = model(board_tensor)
        
        # Get predicted move
        predicted_idx = torch.argmax(policy_output).item()
        predicted_move = ChessPuzzleDataset.index_to_move(predicted_idx)
        
        # Check if move is legal
        if predicted_move in legal_moves:
            is_correct = (predicted_move == solution_move)
        else:
            is_correct = False
            # Find the legal move with highest score
            legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
            legal_move_scores = policy_output[0, legal_move_indices]
            best_legal_idx = legal_move_indices[torch.argmax(legal_move_scores).item()]
            predicted_move = ChessPuzzleDataset.index_to_move(best_legal_idx)
        
        # Get confidence (softmax probability)
        probs = torch.softmax(policy_output[0], dim=0)
        confidence = probs[predicted_idx].item()
        
        # Get value
        position_value = value_output.item()
        
        return is_correct, predicted_move, confidence, position_value


def analyze_by_rating(results_df):
    """Analyze results by rating buckets"""
    buckets = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, 4000)]
    bucket_stats = {}
    
    for min_rating, max_rating in buckets:
        mask = (results_df['rating'] >= min_rating) & (results_df['rating'] < max_rating)
        bucket_df = results_df[mask]
        
        if len(bucket_df) > 0:
            bucket_stats[f"{min_rating}-{max_rating}"] = {
                'count': len(bucket_df),
                'accuracy': bucket_df['is_correct'].mean(),
                'avg_confidence': bucket_df['confidence'].mean(),
                'avg_value': bucket_df['value'].mean()
            }
    
    return bucket_stats


def analyze_by_theme(results_df):
    """Analyze results by puzzle theme"""
    theme_stats = defaultdict(lambda: {'count': 0, 'correct': 0})
    
    for _, row in results_df.iterrows():
        if pd.notna(row['themes']):
            themes = row['themes'].split()
            for theme in themes:
                theme_stats[theme]['count'] += 1
                if row['is_correct']:
                    theme_stats[theme]['correct'] += 1
    
    # Calculate accuracy for each theme
    theme_accuracy = {}
    for theme, stats in theme_stats.items():
        if stats['count'] > 10:  # Only include themes with sufficient puzzles
            theme_accuracy[theme] = stats['correct'] / stats['count']
    
    return theme_accuracy


def plot_results(results_df, bucket_stats, theme_accuracy, output_dir):
    """Create visualization plots"""
    
    # Plot 1: Accuracy by rating
    plt.figure(figsize=(10, 6))
    buckets = list(bucket_stats.keys())
    accuracies = [bucket_stats[b]['accuracy'] for b in buckets]
    counts = [bucket_stats[b]['count'] for b in buckets]
    
    x = range(len(buckets))
    bars = plt.bar(x, accuracies, alpha=0.8, color='skyblue')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Rating Range')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Puzzle Rating')
    plt.xticks(x, buckets, rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_rating.png'), dpi=300)
    plt.close()
    
    # Plot 2: Top themes by accuracy
    if theme_accuracy:
        plt.figure(figsize=(12, 6))
        # Sort themes by accuracy
        sorted_themes = sorted(theme_accuracy.items(), key=lambda x: x[1], reverse=True)
        top_themes = sorted_themes[:15]  # Top 15 themes
        
        themes, accuracies = zip(*top_themes)
        x = range(len(themes))
        
        plt.bar(x, accuracies, alpha=0.8, color='lightgreen')
        plt.xlabel('Puzzle Theme')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy by Puzzle Theme (Top 15)')
        plt.xticks(x, themes, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_theme.png'), dpi=300)
        plt.close()
    
    # Plot 3: Confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['confidence'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Model Confidence')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 4: Confidence vs Accuracy
    plt.figure(figsize=(10, 6))
    confidence_bins = pd.cut(results_df['confidence'], bins=10)
    accuracy_by_confidence = results_df.groupby(confidence_bins)['is_correct'].mean()
    
    bin_centers = [interval.mid for interval in accuracy_by_confidence.index]
    plt.plot(bin_centers, accuracy_by_confidence.values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_confidence.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Chess Puzzle Solver')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to puzzle CSV file')
    parser.add_argument('--split-file', type=str, help='Load data splits from file')
    parser.add_argument('--use-test', action='store_true', help='Evaluate on test set')
    parser.add_argument('--num-puzzles', type=int, help='Limit evaluation to N puzzles')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='Device to use')
    parser.add_argument('--themes', type=str, nargs='+', help='Filter puzzles by themes')
    parser.add_argument('--max-puzzles', type=int, help='Maximum number of puzzles to use from dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
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
    
    # Load model
    logging.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Detect architecture from checkpoint
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
    
    model.to(device)
    model.eval()
    
    # Load dataset
    logging.info(f"Loading dataset from {args.data}")
    dataset = ChessPuzzleDataset(args.data, filter_themes=args.themes, max_puzzles=args.max_puzzles)
    
    if args.themes:
        logging.info(f"Filtered by themes: {args.themes}")
    
    # Determine which dataset to evaluate
    if args.split_file and os.path.exists(args.split_file):
        logging.info(f"Loading data splits from {args.split_file}")
        split_data = ChessPuzzleDataset.load_split_indices(args.split_file)
        
        if args.use_test:
            eval_dataset = Subset(dataset, split_data['test_indices'])
            logging.info(f"Evaluating on test set ({len(eval_dataset)} puzzles)")
        else:
            eval_dataset = Subset(dataset, split_data['val_indices'])
            logging.info(f"Evaluating on validation set ({len(eval_dataset)} puzzles)")
    else:
        eval_dataset = dataset
        logging.info(f"Evaluating on full dataset ({len(eval_dataset)} puzzles)")
        if args.use_test:
            logging.warning("--use-test specified but no split file provided. Evaluating on full dataset.")
    
    # Limit number of puzzles if specified
    num_puzzles = min(args.num_puzzles if args.num_puzzles else len(eval_dataset), len(eval_dataset))
    logging.info(f"Evaluating on {num_puzzles} puzzles")
    
    # Evaluate
    results = []
    
    for i in tqdm(range(num_puzzles), desc="Evaluating"):
        board_tensor, policy_target, value_target, info = eval_dataset[i]
        
        # Get solution move
        solution_idx = torch.argmax(policy_target).item()
        solution_move = ChessPuzzleDataset.index_to_move(solution_idx)
        
        # Get puzzle data for board recreation
        if hasattr(eval_dataset, 'dataset'):  # If it's a Subset
            actual_dataset = eval_dataset.dataset
            actual_idx = eval_dataset.indices[i]
        else:
            actual_dataset = eval_dataset
            actual_idx = i
        
        puzzle = actual_dataset.puzzles.iloc[actual_idx]
        board = chess.Board(puzzle['FEN'])
        moves = puzzle['Moves'].split()
        if len(moves) > 0:
            board.push_uci(moves[0])  # Apply setup move
        
        legal_moves = list(board.legal_moves)
        
        # Evaluate
        is_correct, predicted_move, confidence, value = evaluate_puzzle(
            model, board_tensor, solution_move, legal_moves, device
        )
        
        results.append({
            'puzzle_id': info['puzzle_id'],
            'rating': info['rating'],
            'themes': info['themes'],
            'is_correct': is_correct,
            'solution': solution_move.uci(),
            'predicted': predicted_move.uci(),
            'confidence': confidence,
            'value': value,
            'num_legal_moves': len(legal_moves)
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    overall_accuracy = results_df['is_correct'].mean()
    avg_confidence = results_df['confidence'].mean()
    avg_value = results_df['value'].mean()
    
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logging.info(f"Average Confidence: {avg_confidence:.4f}")
    logging.info(f"Average Value: {avg_value:.4f}")
    
    # Analyze by rating
    bucket_stats = analyze_by_rating(results_df)
    
    logging.info("\nAccuracy by Rating:")
    for bucket, stats in bucket_stats.items():
        logging.info(f"  {bucket}: {stats['accuracy']:.4f} ({stats['count']} puzzles)")
    
    # Analyze by theme
    theme_accuracy = analyze_by_theme(results_df)
    
    if theme_accuracy:
        logging.info("\nTop 10 Themes by Accuracy:")
        sorted_themes = sorted(theme_accuracy.items(), key=lambda x: x[1], reverse=True)
        for theme, accuracy in sorted_themes[:10]:
            logging.info(f"  {theme}: {accuracy:.4f}")
    
    # Create visualizations
    plot_results(results_df, bucket_stats, theme_accuracy, args.output_dir)
    
    # Save detailed results
    results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    
    # Save summary statistics
    summary = {
        'overall_accuracy': overall_accuracy,
        'average_confidence': avg_confidence,
        'average_value': avg_value,
        'num_puzzles_evaluated': num_puzzles,
        'dataset_type': 'test' if args.use_test else 'validation' if args.split_file else 'full',
        'model_path': args.model,
        'data_path': args.data,
        'themes': args.themes
    }
    
    if bucket_stats:
        summary['accuracy_by_rating'] = bucket_stats
    
    if theme_accuracy:
        sorted_themes = sorted(theme_accuracy.items(), key=lambda x: x[1], reverse=True)
        summary['top_10_themes'] = dict(sorted_themes[:10])
    
    with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\nResults saved to {args.output_dir}")
    logging.info("Evaluation completed!")


if __name__ == '__main__':
    main()