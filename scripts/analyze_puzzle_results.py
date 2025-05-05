#!/usr/bin/env python3

import torch
import chess
import sys
import os
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ChessPuzzleNet
from src.dataset import ChessPuzzleDataset
from src.utils import visualize_board_tensor

def analyze_puzzle(model, puzzle_data, device='cpu'):
    """Analyze a single puzzle and return detailed results"""
    fen = puzzle_data['FEN']
    moves = puzzle_data['Moves'].split()
    
    # Create board
    board = chess.Board(fen)
    initial_board = board.copy()
    
    # Apply setup move
    if len(moves) > 0:
        board.push_uci(moves[0])
    
    # Convert to tensor
    dataset = ChessPuzzleDataset('data/lichess_db_puzzle.csv')  # Just to access methods
    board_tensor = dataset.board_to_tensor(board)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        board_tensor = board_tensor.unsqueeze(0).to(device)
        policy_output, value_output = model(board_tensor)
        
        # Get predicted move
        probs = torch.softmax(policy_output[0], dim=0)
        predicted_idx = torch.argmax(policy_output).item()
        predicted_move = ChessPuzzleDataset.index_to_move(predicted_idx)
        
        # Check if legal
        if predicted_move not in board.legal_moves:
            legal_move_indices = [move.from_square * 64 + move.to_square for move in board.legal_moves]
            legal_move_scores = probs[legal_move_indices]
            best_legal_idx = legal_move_indices[torch.argmax(legal_move_scores).item()]
            predicted_move = ChessPuzzleDataset.index_to_move(best_legal_idx)
            predicted_idx = best_legal_idx
        
        confidence = probs[predicted_idx].item()
        
        # Get correct move
        correct_move = None
        if len(moves) > 1:
            correct_move = chess.Move.from_uci(moves[1])
        
        # Check if correct
        is_correct = False
        if correct_move and predicted_move.uci() == correct_move.uci():
            is_correct = True
        
        # Get top 5 moves
        top5_indices = torch.topk(probs, 5).indices
        top5_moves = []
        for idx in top5_indices:
            move = ChessPuzzleDataset.index_to_move(idx.item())
            if move in board.legal_moves:
                move_san = board.san(move)
                prob = probs[idx].item()
                top5_moves.append((move, move_san, prob))
            else:
                prob = probs[idx].item()
                top5_moves.append((move, f"{move.uci()} (illegal)", prob))
        
        return {
            'is_correct': is_correct,
            'predicted_move': predicted_move,
            'correct_move': correct_move,
            'confidence': confidence,
            'value': value_output.item(),
            'initial_board': initial_board,
            'position_board': board,
            'top5_moves': top5_moves,
            'setup_move': moves[0] if moves else None,
            'all_moves': moves
        }

def display_puzzle_analysis(puzzle_data, analysis_result):
    """Display detailed analysis of a puzzle"""
    print("\n" + "="*80)
    print(f"Puzzle Analysis - Rating: {puzzle_data['Rating']}")
    print("="*80)
    
    print(f"Themes: {puzzle_data['Themes']}")
    print(f"FEN: {puzzle_data['FEN']}")
    print(f"Moves: {puzzle_data['Moves']}")
    
    print("\nInitial Position:")
    print(analysis_result['initial_board'])
    
    if analysis_result['setup_move']:
        print(f"\nAfter setup move {analysis_result['setup_move']}:")
        print(analysis_result['position_board'])
    
    print("\nPrediction Results:")
    print(f"Status: {'CORRECT' if analysis_result['is_correct'] else 'INCORRECT'}")
    print(f"Predicted: {analysis_result['predicted_move'].uci()}")
    if analysis_result['predicted_move'] in analysis_result['position_board'].legal_moves:
        print(f"Predicted (SAN): {analysis_result['position_board'].san(analysis_result['predicted_move'])}")
    print(f"Correct: {analysis_result['correct_move'].uci() if analysis_result['correct_move'] else 'None'}")
    if analysis_result['correct_move'] and analysis_result['correct_move'] in analysis_result['position_board'].legal_moves:
        print(f"Correct (SAN): {analysis_result['position_board'].san(analysis_result['correct_move'])}")
    print(f"Confidence: {analysis_result['confidence']:.3f}")
    print(f"Position Value: {analysis_result['value']:.3f}")
    
    print("\nTop 5 Predictions:")
    for i, (move, san, prob) in enumerate(analysis_result['top5_moves']):
        print(f"{i+1}. {san} ({move.uci()}) - {prob:.3f}")
    
    # Show result of predicted move
    if analysis_result['predicted_move'] in analysis_result['position_board'].legal_moves:
        test_board = analysis_result['position_board'].copy()
        test_board.push(analysis_result['predicted_move'])
        print(f"\nPosition after predicted move {analysis_result['predicted_move'].uci()}:")
        print(test_board)
        
        if test_board.is_checkmate():
            print("CHECKMATE!")
        elif test_board.is_check():
            print("CHECK!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Chess Puzzle Results')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--themes', type=str, nargs='+', help='Filter puzzles by themes')
    parser.add_argument('--num-success', type=int, default=3, help='Number of successful puzzles to show')
    parser.add_argument('--num-failures', type=int, default=None, help='Number of failed puzzles to show (default: all)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='Device to use')
    parser.add_argument('--max-puzzles', type=int, default=100, help='Maximum number of puzzles to analyze')
    
    args = parser.parse_args()
    
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
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
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
    
    print(f"Detected architecture: {channels} channels, {num_blocks} blocks")
    
    # Create and load model
    model = ChessPuzzleNet(num_blocks=num_blocks, channels=channels)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading puzzles from {args.data}")
    puzzles_df = pd.read_csv(args.data)
    
    # Filter by themes if specified
    if args.themes:
        mask = puzzles_df['Themes'].fillna('').apply(
            lambda x: any(theme in x.split() for theme in args.themes)
        )
        puzzles_df = puzzles_df[mask]
        print(f"Filtered to {len(puzzles_df)} puzzles with themes: {args.themes}")
    
    # Limit number of puzzles
    if len(puzzles_df) > args.max_puzzles:
        puzzles_df = puzzles_df.sample(n=args.max_puzzles, random_state=42)
        print(f"Limited to {args.max_puzzles} puzzles")
    
    # Analyze puzzles
    results = []
    print("Analyzing puzzles...")
    
    for idx, puzzle in tqdm(puzzles_df.iterrows(), total=len(puzzles_df)):
        analysis = analyze_puzzle(model, puzzle, device)
        analysis['puzzle_data'] = puzzle
        results.append(analysis)
    
    # Separate successes and failures
    successes = [r for r in results if r['is_correct']]
    failures = [r for r in results if not r['is_correct']]
    
    print(f"\n" + "="*80)
    print(f"SUMMARY: {len(successes)} successes, {len(failures)} failures")
    print(f"Accuracy: {len(successes)/len(results):.2%}")
    print("="*80)
    
    # Display successful puzzles
    print(f"\n\nSUCCESSFUL PREDICTIONS ({args.num_success} examples):")
    for i, result in enumerate(successes[:args.num_success]):
        display_puzzle_analysis(result['puzzle_data'], result)
    
    # Display failed puzzles
    num_failures_to_show = args.num_failures if args.num_failures is not None else len(failures)
    print(f"\n\nFAILED PREDICTIONS ({num_failures_to_show} examples):")
    for i, result in enumerate(failures[:num_failures_to_show]):
        display_puzzle_analysis(result['puzzle_data'], result)
    
    # Additional statistics
    if failures:
        print("\n\nFAILURE ANALYSIS:")
        print("="*80)
        
        # Average confidence for failures
        avg_confidence_failures = sum(f['confidence'] for f in failures) / len(failures)
        avg_confidence_successes = sum(s['confidence'] for s in successes) / len(successes) if successes else 0
        
        print(f"Average confidence (failures): {avg_confidence_failures:.3f}")
        print(f"Average confidence (successes): {avg_confidence_successes:.3f}")
        
        # Rating distribution of failures
        rating_failures = [f['puzzle_data']['Rating'] for f in failures]
        if rating_failures:
            print(f"Average rating of failed puzzles: {sum(rating_failures)/len(rating_failures):.0f}")
            print(f"Min rating: {min(rating_failures)}")
            print(f"Max rating: {max(rating_failures)}")

if __name__ == '__main__':
    main()