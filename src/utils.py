import chess
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging


def board_to_fen(board_tensor: torch.Tensor) -> str:
    """Convert board tensor back to FEN string"""
    piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    fen_rows = []
    for row in range(8):
        fen_row = ''
        empty_count = 0
        
        for col in range(8):
            piece_found = False
            for piece_idx in range(12):
                if board_tensor[piece_idx, row, col] > 0:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece_symbols[piece_idx]
                    piece_found = True
                    break
            
            if not piece_found:
                empty_count += 1
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    return '/'.join(fen_rows) + ' w - - 0 1'  # Assume white to move


def visualize_board_tensor(board_tensor: torch.Tensor) -> str:
    """Create ASCII representation of board tensor"""
    piece_symbols = {
        0: '♙', 1: '♘', 2: '♗', 3: '♖', 4: '♕', 5: '♔',
        6: '♟', 7: '♞', 8: '♝', 9: '♜', 10: '♛', 11: '♚'
    }
    
    board_str = "  a b c d e f g h\n"
    for row in range(8):
        board_str += str(8 - row) + " "
        for col in range(8):
            piece_found = False
            for piece_idx in range(12):
                if board_tensor[piece_idx, row, col] > 0:
                    board_str += piece_symbols[piece_idx] + " "
                    piece_found = True
                    break
            
            if not piece_found:
                board_str += ". "
        
        board_str += str(8 - row) + "\n"
    
    board_str += "  a b c d e f g h"
    return board_str


def get_move_san(board: chess.Board, move: chess.Move) -> str:
    """Get move in Standard Algebraic Notation"""
    try:
        return board.san(move)
    except:
        return move.uci()


def calculate_material_balance(board: chess.Board) -> int:
    """Calculate material balance (positive for white advantage)"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    
    balance = 0
    for piece_type in piece_values:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        balance += (white_pieces - black_pieces) * piece_values[piece_type]
    
    return balance


def is_tactical_position(board: chess.Board) -> Tuple[bool, List[str]]:
    """Check if position contains tactical motifs"""
    tactics = []
    
    # Check for checks
    if board.is_check():
        tactics.append("check")
    
    # Check for attacks on queen
    for square in board.pieces(chess.QUEEN, not board.turn):
        if board.is_attacked_by(board.turn, square):
            tactics.append("queen_attack")
            break
    
    # Check for undefended pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color != board.turn:
            if board.is_attacked_by(board.turn, square):
                attackers = board.attackers(board.turn, square)
                defenders = board.attackers(not board.turn, square)
                
                if len(attackers) > len(defenders):
                    tactics.append("hanging_piece")
                    break
    
    # Check for forks
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for square in board.pieces(piece_type, board.turn):
            attacks = board.attacks(square)
            valuable_targets = 0
            
            for target_square in attacks:
                target = board.piece_at(target_square)
                if target and target.color != board.turn:
                    if target.piece_type in [chess.QUEEN, chess.ROOK] or target.piece_type > piece_type:
                        valuable_targets += 1
            
            if valuable_targets >= 2:
                tactics.append(f"{chess.piece_name(piece_type)}_fork")
                break
    
    return len(tactics) > 0, tactics


def analyze_puzzle_difficulty(fen: str, moves: List[str]) -> Dict[str, int]:
    """Analyze factors that contribute to puzzle difficulty"""
    board = chess.Board(fen)
    
    # Apply setup move
    if moves and len(moves) > 0:
        board.push_uci(moves[0])
    
    difficulty_factors = {
        'material_imbalance': abs(calculate_material_balance(board)),
        'num_legal_moves': len(list(board.legal_moves)),
        'num_captures': sum(1 for m in board.legal_moves if board.is_capture(m)),
        'num_checks': sum(1 for m in board.legal_moves if board.gives_check(m)),
        'pieces_count': len(board.piece_map()),
        'is_endgame': len(board.piece_map()) <= 12,
        'has_promotion': any(m.promotion for m in board.legal_moves),
        'tactical_complexity': len(is_tactical_position(board)[1])
    }
    
    return difficulty_factors


def create_augmented_positions(board: chess.Board) -> List[Tuple[chess.Board, bool]]:
    """Create augmented positions for training data augmentation"""
    augmented = []
    
    # Original position
    augmented.append((board.copy(), False))
    
    # Mirrored position (flip colors)
    mirrored = board.mirror()
    augmented.append((mirrored, True))
    
    return augmented


def puzzle_complexity_score(puzzle_data: Dict) -> float:
    """Calculate a complexity score for a puzzle"""
    rating = puzzle_data.get('Rating', 1500)
    moves = puzzle_data.get('Moves', '').split()
    themes = puzzle_data.get('Themes', '').split()
    
    # Base score from rating
    score = rating / 3000.0
    
    # Adjust for number of moves
    if len(moves) > 4:
        score += 0.1
    
    # Adjust for tactical themes
    complex_themes = ['sacrifice', 'clearance', 'interference', 'zugzwang']
    for theme in themes:
        if theme in complex_themes:
            score += 0.05
    
    return min(score, 1.0)


def create_curriculum_batches(puzzles_df: pd.DataFrame, num_batches: int = 10) -> List[pd.DataFrame]:
    """Create curriculum learning batches sorted by difficulty"""
    # Add complexity scores
    puzzles_df['complexity'] = puzzles_df.apply(puzzle_complexity_score, axis=1)
    
    # Sort by complexity
    puzzles_df = puzzles_df.sort_values('complexity')
    
    # Create batches
    batch_size = len(puzzles_df) // num_batches
    batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_batches - 1 else len(puzzles_df)
        batch = puzzles_df.iloc[start_idx:end_idx].copy()
        batches.append(batch)
    
    return batches


def validate_puzzle_data(puzzle_data: Dict) -> bool:
    """Validate puzzle data for training"""
    required_fields = ['FEN', 'Moves']
    
    # Check required fields
    for field in required_fields:
        if field not in puzzle_data or not puzzle_data[field]:
            return False
    
    # Validate FEN
    try:
        board = chess.Board(puzzle_data['FEN'])
    except:
        return False
    
    # Validate moves
    moves = puzzle_data['Moves'].split()
    if len(moves) < 2:  # Need at least setup move and solution
        return False
    
    # Validate move sequence
    try:
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return False
            board.push(move)
    except:
        return False
    
    return True


def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logging.info(f"CUDA GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory reporting like CUDA
        logging.info("Using MPS (Apple Silicon GPU) - Memory reporting not available")


def setup_deterministic_training(seed: int = 42):
    """Setup deterministic training for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS doesn't have the same level of deterministic control as CUDA
        logging.info("Note: MPS (Apple Silicon) may not be fully deterministic")
        torch.mps.manual_seed(seed)