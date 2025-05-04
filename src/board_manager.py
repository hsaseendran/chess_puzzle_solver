import chess
from typing import List


class BoardManager:
    """Handles chess board operations using python-chess"""
    
    def __init__(self):
        self.board = chess.Board()
    
    def set_position(self, fen: str) -> None:
        """Set board to specific position"""
        self.board.set_fen(fen)
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in current position"""
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> None:
        """Execute a move on the board"""
        self.board.push(move)
    
    def unmake_move(self) -> None:
        """Undo the last move"""
        self.board.pop()
    
    def copy(self) -> 'BoardManager':
        """Create a copy of the current board state"""
        new_manager = BoardManager()
        new_manager.board = self.board.copy()
        return new_manager
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.board.is_game_over()
    
    def get_fen(self) -> str:
        """Get current FEN position"""
        return self.board.fen()
    
    def get_board_state(self) -> chess.Board:
        """Get the underlying chess.Board object"""
        return self.board