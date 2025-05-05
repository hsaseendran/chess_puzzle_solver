import torch
from torch.utils.data import Dataset, Subset
import chess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ChessPuzzleDataset(Dataset):
    """Dataset for chess puzzles with proper train/val/test splitting"""
    
    def __init__(self, csv_path, filter_themes=None, max_puzzles=None):
        self.puzzles = pd.read_csv(csv_path)
        self.puzzles = self.puzzles.dropna(subset=['FEN', 'Moves'])
        
        # Filter by themes if specified
        if filter_themes:
            if isinstance(filter_themes, str):
                filter_themes = [filter_themes]
            
            # Create a mask for puzzles containing any of the specified themes
            mask = self.puzzles['Themes'].fillna('').apply(
                lambda x: any(theme in x.split() for theme in filter_themes)
            )
            self.puzzles = self.puzzles[mask]
            
            print(f"Filtered to {len(self.puzzles)} puzzles with themes: {filter_themes}")
        
        # Limit number of puzzles if specified
        if max_puzzles and len(self.puzzles) > max_puzzles:
            self.puzzles = self.puzzles.sample(n=max_puzzles, random_state=42)
            print(f"Limited to {max_puzzles} puzzles")
        
        # Piece to index mapping
        self.piece_to_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        puzzle = self.puzzles.iloc[idx]
        fen = puzzle['FEN']
        moves = puzzle['Moves'].split()
        
        # Create board from FEN
        board = chess.Board(fen)
        
        # Apply setup move (opponent's move)
        if len(moves) > 0:
            board.push_uci(moves[0])
        
        # The solution move is our target
        solution_move = chess.Move.from_uci(moves[1]) if len(moves) > 1 else None
        
        # Convert board to tensor
        board_tensor = self.board_to_tensor(board)
        
        # Convert move to policy index
        policy_target = torch.zeros(4096)
        if solution_move:
            move_idx = self.move_to_index(solution_move)
            policy_target[move_idx] = 1.0
        
        # Value target (1.0 for winning puzzle position)
        value_target = torch.tensor([1.0])
        
        # Additional info for evaluation
        info = {
            'rating': puzzle.get('Rating', 1500),
            'themes': puzzle.get('Themes', ''),
            'puzzle_id': puzzle.get('PuzzleId', str(idx))
        }
        
        return board_tensor, policy_target, value_target, info
    
    def board_to_tensor(self, board):
        """Convert board to 12x8x8 tensor (piece-centric representation)"""
        tensor = torch.zeros(12, 8, 8)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = self.piece_to_idx[piece.symbol()]
                row = 7 - (square // 8)  # Flip to match board view
                col = square % 8
                tensor[idx, row, col] = 1.0
        
        return tensor
    
    def move_to_index(self, move):
        """Convert move to policy index (0-4095)"""
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square
    
    @staticmethod
    def index_to_move(index):
        """Convert policy index back to move"""
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    @staticmethod
    def create_data_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
        """Create train/val/test splits with stratification by rating if possible"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Get indices
        indices = list(range(len(dataset)))
        
        # Try to stratify by rating buckets
        ratings = []
        for i in indices:
            _, _, _, info = dataset[i]
            rating = info['rating']
            # Create rating buckets (every 500 rating points)
            bucket = int(rating // 500) * 500
            ratings.append(bucket)
        
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=random_state,
            stratify=ratings
        )
        
        # Second split: separate train and validation
        train_val_ratings = [ratings[i] for i in train_val_idx]
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_ratings
        )
        
        # Create subset datasets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def save_split_indices(train_idx, val_idx, test_idx, filepath):
        """Save dataset split indices for reproducibility"""
        split_data = {
            'train_indices': train_idx,
            'val_indices': val_idx,
            'test_indices': test_idx
        }
        torch.save(split_data, filepath)
    
    @staticmethod
    def load_split_indices(filepath):
        """Load dataset split indices"""
        return torch.load(filepath)
    
    @staticmethod
    def mirror_board_tensor(tensor):
        """Mirror board tensor for data augmentation"""
        # Swap white and black pieces
        mirrored = torch.zeros_like(tensor)
        mirrored[0:6] = tensor[6:12]  # Black pieces become white
        mirrored[6:12] = tensor[0:6]  # White pieces become black
        
        # Flip vertically
        mirrored = torch.flip(mirrored, dims=[1])
        
        return mirrored
    
    @staticmethod
    def mirror_move_index(index):
        """Mirror move index for data augmentation"""
        from_square = index // 64
        to_square = index % 64
        
        # Mirror squares
        from_rank = 7 - (from_square // 8)
        from_file = from_square % 8
        to_rank = 7 - (to_square // 8)
        to_file = to_square % 8
        
        mirrored_from = from_rank * 8 + from_file
        mirrored_to = to_rank * 8 + to_file
        
        return mirrored_from * 64 + mirrored_to