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

def test_single_puzzle(model, fen, moves, device='cpu'):
    """Test model on a single puzzle"""
    # Create board
    board = chess.Board(fen)
    
    # Apply setup move
    setup_moves = moves.split()
    if len(setup_moves) > 0:
        board.push_uci(setup_moves[0])
        print(f"\nPosition after setup move: {setup_moves[0]}")
    
    # Convert to tensor
    dataset = ChessPuzzleDataset('dummy.csv')  # Just to access methods
    board_tensor = dataset.board_to_tensor(board)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        board_tensor = board_tensor.unsqueeze(0).to(device)
        policy_output, value_output = model(board_tensor)
        
        # Get top 5 moves
        probs = torch.softmax(policy_output[0], dim=0)
        top5_indices = torch.topk(probs, 5).indices
        
        print(f"\nTop 5 predicted moves:")
        for i, idx in enumerate(top5_indices):
            move = ChessPuzzleDataset.index_to_move(idx.item())
            if move in board.legal_moves:
                move_san = board.san(move)
                print(f"{i+1}. {move_san} ({move.uci()}) - {probs[idx].item():.4f}")
            else:
                print(f"{i+1}. {move.uci()} (illegal) - {probs[idx].item():.4f}")
        
        # Show correct answer
        if len(setup_moves) > 1:
            correct_move = setup_moves[1]
            print(f"\nCorrect answer: {correct_move}")
            
            # Check if prediction matches
            predicted_idx = torch.argmax(policy_output).item()
            predicted_move = ChessPuzzleDataset.index_to_move(predicted_idx)
            if predicted_move.uci() == correct_move:
                print("✓ Correct prediction!")
            else:
                print("✗ Incorrect prediction")
        
        print(f"\nPosition evaluation: {value_output.item():.4f}")

def test_model_accuracy(model_path, test_data_path, num_samples=100, device='cpu', filter_themes=None):
    """Test model accuracy on a set of puzzles"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to detect model architecture from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect number of channels from the conv_input layer
    conv_input_weight = state_dict.get('conv_input.weight', None)
    if conv_input_weight is not None:
        channels = conv_input_weight.shape[0]
        print(f"Detected model with {channels} channels")
    else:
        # Try to detect from policy_conv
        policy_conv_weight = state_dict.get('policy_conv.weight', None)
        if policy_conv_weight is not None:
            channels = policy_conv_weight.shape[1]  # Input channels to policy_conv
            print(f"Detected model with {channels} channels from policy layer")
        else:
            print("Warning: Could not detect number of channels, using default 256")
            channels = 256
    
    # Detect number of residual blocks
    num_blocks = 0
    for key in state_dict.keys():
        if key.startswith('residual_blocks.') and key.endswith('.conv1.weight'):
            block_num = int(key.split('.')[1])
            num_blocks = max(num_blocks, block_num + 1)
    
    if num_blocks == 0:
        num_blocks = 20  # Default
        print("Warning: Could not detect number of blocks, using default 20")
    else:
        print(f"Detected model with {num_blocks} residual blocks")
    
    # Create model with correct architecture
    model = ChessPuzzleNet(num_blocks=num_blocks, channels=channels)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load test data with theme filtering
    dataset = ChessPuzzleDataset(test_data_path, filter_themes=filter_themes)
    
    if filter_themes:
        print(f"Filtered to puzzles with themes: {filter_themes}")
    
    correct = 0
    total = 0
    
    # Results by rating
    rating_buckets = {
        '0-1000': {'correct': 0, 'total': 0},
        '1000-1500': {'correct': 0, 'total': 0},
        '1500-2000': {'correct': 0, 'total': 0},
        '2000-2500': {'correct': 0, 'total': 0},
        '2500+': {'correct': 0, 'total': 0}
    }
    
    num_samples = min(num_samples, len(dataset))
    print(f"Testing on {num_samples} puzzles...\n")
    
    for i in tqdm(range(num_samples), desc="Testing"):
        board_tensor, policy_target, value_target, info = dataset[i]
        
        with torch.no_grad():
            board_tensor = board_tensor.unsqueeze(0).to(device)
            policy_output, value_output = model(board_tensor)
            
            # Get predicted move
            predicted_idx = torch.argmax(policy_output).item()
            target_idx = torch.argmax(policy_target).item()
            
            is_correct = (predicted_idx == target_idx)
            if is_correct:
                correct += 1
            total += 1
            
            # Track accuracy by rating
            rating = info['rating']
            if rating < 1000:
                bucket = '0-1000'
            elif rating < 1500:
                bucket = '1000-1500'
            elif rating < 2000:
                bucket = '1500-2000'
            elif rating < 2500:
                bucket = '2000-2500'
            else:
                bucket = '2500+'
            
            rating_buckets[bucket]['total'] += 1
            if is_correct:
                rating_buckets[bucket]['correct'] += 1
    
    # Print results
    print(f"\n=== Test Results ===")
    print(f"Overall Accuracy: {correct/total:.4f} ({correct}/{total})")
    
    print("\nAccuracy by Rating:")
    for bucket, stats in rating_buckets.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"  {bucket}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    return correct / total

def display_sample_puzzles(model_path, test_data_path, num_samples=5, device='cpu', filter_themes=None):
    """Display predictions for a few sample puzzles"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect channels and blocks
    channels = 256
    num_blocks = 20
    
    conv_input_weight = state_dict.get('conv_input.weight', None)
    if conv_input_weight is not None:
        channels = conv_input_weight.shape[0]
    
    for key in state_dict.keys():
        if key.startswith('residual_blocks.') and key.endswith('.conv1.weight'):
            block_num = int(key.split('.')[1])
            num_blocks = max(num_blocks, block_num + 1)
    
    # Create and load model
    model = ChessPuzzleNet(num_blocks=num_blocks, channels=channels)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = ChessPuzzleDataset(test_data_path, filter_themes=filter_themes)
    
    num_samples = min(num_samples, len(dataset))
    
    for i in range(num_samples):
        print(f"\n{'='*50}")
        print(f"Puzzle {i+1}/{num_samples}")
        print('='*50)
        
        board_tensor, policy_target, value_target, info = dataset[i]
        puzzle = dataset.puzzles.iloc[i]
        
        print(f"Rating: {info['rating']}")
        print(f"Themes: {info['themes']}")
        print(f"FEN: {puzzle['FEN']}")
        print(f"Moves: {puzzle['Moves']}")
        
        test_single_puzzle(model, puzzle['FEN'], puzzle['Moves'], device)
        print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Chess Puzzle Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--mode', choices=['accuracy', 'display', 'single'], default='accuracy', 
                        help='Test mode: accuracy test, display samples, or single puzzle')
    parser.add_argument('--themes', type=str, nargs='+', help='Filter puzzles by themes (e.g., mate mateIn2 fork)')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples for accuracy test')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='Device to use')
    
    # Arguments for single puzzle mode
    parser.add_argument('--fen', type=str, help='FEN for single puzzle test')
    parser.add_argument('--moves', type=str, help='Moves for single puzzle test')
    
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
    
    if args.mode == 'single':
        if not args.fen or not args.moves:
            print("Error: --fen and --moves required for single puzzle test")
            return
        
        # Load model for single puzzle
        checkpoint = torch.load(args.model, map_location=device)
        
        # Detect architecture
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        channels = 256
        num_blocks = 20
        
        conv_input_weight = state_dict.get('conv_input.weight', None)
        if conv_input_weight is not None:
            channels = conv_input_weight.shape[0]
        
        for key in state_dict.keys():
            if key.startswith('residual_blocks.') and key.endswith('.conv1.weight'):
                block_num = int(key.split('.')[1])
                num_blocks = max(num_blocks, block_num + 1)
        
        model = ChessPuzzleNet(num_blocks=num_blocks, channels=channels)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        
        test_single_puzzle(model, args.fen, args.moves, device)
    
    elif args.mode == 'display':
        display_sample_puzzles(args.model, args.data, args.num_samples, device, args.themes)
    
    else:  # accuracy mode
        test_model_accuracy(args.model, args.data, args.num_samples, device, args.themes)

if __name__ == '__main__':
    main()