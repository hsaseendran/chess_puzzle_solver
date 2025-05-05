#!/usr/bin/env python3

import torch
import chess
import chess.svg
import sys
import os
import pandas as pd
from IPython.display import SVG, display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ChessPuzzleNet
from src.dataset import ChessPuzzleDataset

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
    dataset = ChessPuzzleDataset('dummy.csv')  # Just to access methods
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

def visualize_puzzle(puzzle_data, analysis_result, output_file=None):
    """Create a visualization of the puzzle"""
    fig = plt.figure(figsize=(12, 8))
    
    # Create layout
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Initial position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Initial Position")
    initial_board = analysis_result['initial_board']
    if analysis_result['setup_move']:
        setup_move = chess.Move.from_uci(analysis_result['setup_move'])
        svg_data = chess.svg.board(initial_board, arrows=[(setup_move.from_square, setup_move.to_square)])
    else:
        svg_data = chess.svg.board(initial_board)
    
    # Convert SVG to image for matplotlib
    import io
    from PIL import Image
    import cairosvg
    
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    img = Image.open(io.BytesIO(png_data))
    ax1.imshow(img)
    ax1.axis('off')
    
    # Position after setup move
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Position to Solve")
    position_board = analysis_result['position_board']
    
    # Show predicted vs correct move
    arrows = []
    if analysis_result['predicted_move']:
        pred_move = analysis_result['predicted_move']
        arrows.append((pred_move.from_square, pred_move.to_square, {'color': 'red'}))
    
    if analysis_result['correct_move'] and not analysis_result['is_correct']:
        corr_move = analysis_result['correct_move']
        arrows.append((corr_move.from_square, corr_move.to_square, {'color': 'green'}))
    
    svg_data = chess.svg.board(position_board, arrows=arrows)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    img = Image.open(io.BytesIO(png_data))
    ax2.imshow(img)
    ax2.axis('off')
    
    # Info panel
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    info_text = f"Puzzle Rating: {puzzle_data['Rating']}\n"
    info_text += f"Themes: {puzzle_data['Themes']}\n"
    info_text += f"FEN: {puzzle_data['FEN']}\n"
    info_text += f"Moves: {puzzle_data['Moves']}\n\n"
    
    info_text += f"Prediction: {'CORRECT' if analysis_result['is_correct'] else 'INCORRECT'}\n"
    info_text += f"Predicted: {analysis_result['predicted_move'].uci() if analysis_result['predicted_move'] else 'None'}\n"
    info_text += f"Correct: {analysis_result['correct_move'].uci() if analysis_result['correct_move'] else 'None'}\n"
    info_text += f"Confidence: {analysis_result['confidence']:.3f}\n"
    info_text += f"Position Value: {analysis_result['value']:.3f}\n\n"
    
    info_text += "Top 5 Predictions:\n"
    for i, (move, san, prob) in enumerate(analysis_result['top5_moves']):
        info_text += f"{i+1}. {san} - {prob:.3f}\n"
    
    ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Chess Puzzle Results')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--themes', type=str, nargs='+', help='Filter puzzles by themes')
    parser.add_argument('--num-success', type=int, default=5, help='Number of successful puzzles to show')
    parser.add_argument('--show-all-failures', action='store_true', help='Show all failed puzzles')
    parser.add_argument('--output-dir', type=str, default='puzzle_visualizations', help='Output directory for visualizations')
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    print(f"\nResults: {len(successes)} successes, {len(failures)} failures")
    
    # Visualize successful puzzles
    print(f"\nVisualizing {args.num_success} successful puzzles...")
    for i, result in enumerate(successes[:args.num_success]):
        output_file = os.path.join(args.output_dir, f'success_{i+1}.png')
        visualize_puzzle(result['puzzle_data'], result, output_file)
        print(f"Saved: {output_file}")
    
    # Visualize failed puzzles
    failures_to_show = failures if args.show_all_failures else failures[:5]
    print(f"\nVisualizing {len(failures_to_show)} failed puzzles...")
    for i, result in enumerate(failures_to_show):
        output_file = os.path.join(args.output_dir, f'failure_{i+1}.png')
        visualize_puzzle(result['puzzle_data'], result, output_file)
        print(f"Saved: {output_file}")
    
    # Create summary report
    summary_file = os.path.join(args.output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Chess Puzzle Analysis Summary\n")
        f.write("============================\n\n")
        f.write(f"Total puzzles analyzed: {len(results)}\n")
        f.write(f"Successful predictions: {len(successes)}\n")
        f.write(f"Failed predictions: {len(failures)}\n")
        f.write(f"Accuracy: {len(successes)/len(results):.2%}\n\n")
        
        f.write("Failed Puzzles Details:\n")
        f.write("----------------------\n")
        for i, result in enumerate(failures):
            puzzle = result['puzzle_data']
            f.write(f"\nFailure #{i+1}:\n")
            f.write(f"  Rating: {puzzle['Rating']}\n")
            f.write(f"  Themes: {puzzle['Themes']}\n")
            f.write(f"  FEN: {puzzle['FEN']}\n")
            f.write(f"  Moves: {puzzle['Moves']}\n")
            f.write(f"  Predicted: {result['predicted_move'].uci()}\n")
            f.write(f"  Correct: {result['correct_move'].uci()}\n")
            f.write(f"  Confidence: {result['confidence']:.3f}\n")
            f.write(f"  Top 5 predictions:\n")
            for j, (move, san, prob) in enumerate(result['top5_moves']):
                f.write(f"    {j+1}. {san} - {prob:.3f}\n")
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Summary saved to {summary_file}")

if __name__ == '__main__':
    main()