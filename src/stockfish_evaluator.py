import chess
import chess.engine
import numpy as np
import logging
from typing import Optional, Tuple, Union


class StockfishEvaluator:
    """Wrapper for Stockfish chess engine"""
    
    def __init__(self, stockfish_path: str, depth: int = 15):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the Stockfish engine"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            logging.info(f"Stockfish engine initialized successfully at {self.stockfish_path}")
        except Exception as e:
            logging.error(f"Failed to initialize Stockfish engine: {e}")
            raise
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Get Stockfish evaluation of position"""
        if self.engine is None:
            raise RuntimeError("Stockfish engine not initialized")
        
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            
            # Extract score
            score = info["score"].relative
            
            # Handle mate scores
            if score.is_mate():
                mate_score = score.mate()
                if mate_score > 0:
                    return 1.0  # Winning
                else:
                    return -1.0  # Losing
            
            # Convert centipawn score to [-1, 1] range
            cp_score = score.score()
            if cp_score is None:
                return 0.0
            
            # Use tanh to normalize score
            normalized_score = np.tanh(cp_score / 1000.0)
            return normalized_score
            
        except Exception as e:
            logging.error(f"Error evaluating position: {e}")
            return 0.0
    
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get Stockfish's recommended move"""
        if self.engine is None:
            raise RuntimeError("Stockfish engine not initialized")
        
        try:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            return result.move
        except Exception as e:
            logging.error(f"Error getting best move: {e}")
            return None
    
    def evaluate_move(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate a specific move by comparing positions"""
        if self.engine is None:
            raise RuntimeError("Stockfish engine not initialized")
        
        try:
            # Current position evaluation
            current_eval = self.evaluate_position(board)
            
            # Position after move
            board.push(move)
            new_eval = self.evaluate_position(board)
            board.pop()
            
            # Return the improvement
            return new_eval - current_eval
            
        except Exception as e:
            logging.error(f"Error evaluating move: {e}")
            return 0.0
    
    def get_move_analysis(self, board: chess.Board, move: chess.Move) -> Tuple[float, Optional[str]]:
        """Get detailed analysis of a move"""
        if self.engine is None:
            raise RuntimeError("Stockfish engine not initialized")
        
        try:
            # Analyze position after move
            board.push(move)
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            board.pop()
            
            # Extract score and principal variation
            score = info["score"].relative
            pv = info.get("pv", [])
            
            # Convert score
            if score.is_mate():
                eval_score = 1.0 if score.mate() > 0 else -1.0
            else:
                cp_score = score.score()
                eval_score = np.tanh(cp_score / 1000.0) if cp_score is not None else 0.0
            
            # Format principal variation
            pv_string = " ".join([m.uci() for m in pv[:5]]) if pv else None
            
            return eval_score, pv_string
            
        except Exception as e:
            logging.error(f"Error analyzing move: {e}")
            return 0.0, None
    
    def close(self):
        """Clean up engine resources"""
        if self.engine is not None:
            try:
                self.engine.quit()
                logging.info("Stockfish engine closed successfully")
            except Exception as e:
                logging.error(f"Error closing Stockfish engine: {e}")
            finally:
                self.engine = None
    
    def __del__(self):
        """Ensure engine is closed on object destruction"""
        self.close()