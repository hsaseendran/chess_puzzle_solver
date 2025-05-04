import chess
import numpy as np
import torch
from typing import List, Tuple, Dict


class FeatureExtractor:
    """Extracts features from chess positions and moves"""
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
    
    def extract_board_features(self, board: chess.Board) -> torch.Tensor:
        """Extract features from the entire board position"""
        features = []
        
        # 1. Piece placement matrix (8x8x12)
        piece_matrix = self._get_piece_placement_matrix(board)
        features.append(piece_matrix.flatten())
        
        # 2. Attack and defense maps
        attack_map = self._get_attack_map(board)
        features.append(attack_map.flatten())
        
        # 3. Material count
        material = self._get_material_count(board)
        features.append(material)
        
        # 4. Castling rights
        castling = self._get_castling_rights(board)
        features.append(castling)
        
        # 5. King safety features
        king_safety = self._evaluate_king_safety(board)
        features.append(king_safety)
        
        # 6. Pawn structure
        pawn_structure = self._evaluate_pawn_structure(board)
        features.append(pawn_structure)
        
        # 7. Piece mobility
        mobility = self._calculate_mobility(board)
        features.append(mobility)
        
        # 8. Center control
        center_control = self._evaluate_center_control(board)
        features.append(center_control)
        
        return torch.cat([torch.tensor(f).float() for f in features])
    
    def extract_move_features(self, board: chess.Board, move: chess.Move) -> torch.Tensor:
        """Extract features specific to a move"""
        features = []
        
        # 1. Basic move properties
        features.extend([
            float(board.is_capture(move)),
            float(board.gives_check(move)),
            float(board.is_castling(move)),
            float(board.is_en_passant(move)),
            float(move.promotion is not None)
        ])
        
        # 2. Piece values involved
        moving_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        features.extend([
            self.piece_values.get(moving_piece.piece_type, 0) if moving_piece else 0,
            self.piece_values.get(captured_piece.piece_type, 0) if captured_piece else 0
        ])
        
        # 3. Square properties
        from_rank, from_file = chess.square_rank(move.from_square), chess.square_file(move.from_square)
        to_rank, to_file = chess.square_rank(move.to_square), chess.square_file(move.to_square)
        features.extend([from_rank, from_file, to_rank, to_file])
        
        # 4. Move creates tactical opportunities
        test_board = board.copy()
        test_board.push(move)
        
        features.extend([
            float(self._creates_fork(board, move)),
            float(self._creates_pin(board, move)),
            float(self._creates_discovered_attack(board, move)),
            float(self._threatens_mate_in_n(test_board, n=1)),
            float(self._threatens_mate_in_n(test_board, n=2))
        ])
        
        return torch.tensor(features).float()
    
    def _get_piece_placement_matrix(self, board: chess.Board) -> np.ndarray:
        """Create 8x8x12 tensor representing piece positions"""
        matrix = np.zeros((8, 8, 12))
        piece_idx = {
            (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                idx = piece_idx[(piece.piece_type, piece.color)]
                matrix[rank][file][idx] = 1
        
        return matrix
    
    def _get_attack_map(self, board: chess.Board) -> np.ndarray:
        """Create 8x8 matrix of attack/defense values"""
        attack_map = np.zeros((8, 8))
        
        for square in chess.SQUARES:
            # Count attackers and defenders
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            
            rank, file = chess.square_rank(square), chess.square_file(square)
            attack_map[rank][file] = white_attackers - black_attackers
        
        return attack_map
    
    def _get_material_count(self, board: chess.Board) -> np.ndarray:
        """Count material for both sides"""
        material = np.zeros(12)  # 6 piece types * 2 colors
        
        for piece_type in chess.PIECE_TYPES:
            for color in chess.COLORS:
                count = len(board.pieces(piece_type, color))
                idx = piece_type - 1 + (6 if color == chess.BLACK else 0)
                material[idx] = count
        
        return material
    
    def _get_castling_rights(self, board: chess.Board) -> np.ndarray:
        """Get castling rights as binary features"""
        rights = np.zeros(4)
        rights[0] = float(board.has_kingside_castling_rights(chess.WHITE))
        rights[1] = float(board.has_queenside_castling_rights(chess.WHITE))
        rights[2] = float(board.has_kingside_castling_rights(chess.BLACK))
        rights[3] = float(board.has_queenside_castling_rights(chess.BLACK))
        return rights
    
    def _evaluate_king_safety(self, board: chess.Board) -> np.ndarray:
        """Evaluate king safety for both sides"""
        safety = np.zeros(2)  # White, Black
        
        for color in chess.COLORS:
            king_square = board.king(color)
            if king_square is not None:
                # Count pawn shield
                pawn_shield = self._count_pawn_shield(board, king_square, color)
                # Count attackers near king
                attackers = self._count_king_attackers(board, king_square, not color)
                safety[color] = pawn_shield - attackers
        
        return safety
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> np.ndarray:
        """Evaluate pawn structure features"""
        structure = np.zeros(6)  # doubled, isolated, passed for each color
        
        for color in chess.COLORS:
            pawns = board.pieces(chess.PAWN, color)
            files_with_pawns = [chess.square_file(sq) for sq in pawns]
            
            # Doubled pawns
            structure[color * 3] = len(files_with_pawns) - len(set(files_with_pawns))
            
            # Isolated pawns
            isolated = 0
            for file in set(files_with_pawns):
                if file > 0 and (file - 1) not in files_with_pawns:
                    if file < 7 and (file + 1) not in files_with_pawns:
                        isolated += 1
            structure[color * 3 + 1] = isolated
            
            # Passed pawns (simplified)
            passed = 0
            for pawn_sq in pawns:
                if self._is_passed_pawn(board, pawn_sq, color):
                    passed += 1
            structure[color * 3 + 2] = passed
        
        return structure
    
    def _calculate_mobility(self, board: chess.Board) -> np.ndarray:
        """Calculate piece mobility for both sides"""
        mobility = np.zeros(2)  # White, Black
        
        for color in chess.COLORS:
            board_copy = board.copy()
            board_copy.turn = color
            mobility[color] = len(list(board_copy.legal_moves))
        
        return mobility
    
    def _evaluate_center_control(self, board: chess.Board) -> np.ndarray:
        """Evaluate control of central squares"""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        control = np.zeros(2)  # White, Black
        
        for square in center_squares:
            white_control = len(board.attackers(chess.WHITE, square))
            black_control = len(board.attackers(chess.BLACK, square))
            control[0] += white_control
            control[1] += black_control
        
        return control
    
    # Helper methods for tactical features
    def _creates_fork(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates a fork"""
        test_board = board.copy()
        test_board.push(move)
        
        piece = test_board.piece_at(move.to_square)
        if not piece:
            return False
        
        attacks = board.attacks(move.to_square)
        valuable_targets = 0
        
        for square in attacks:
            target = test_board.piece_at(square)
            if target and target.color != piece.color:
                if target.piece_type in [chess.QUEEN, chess.ROOK] or target.piece_type == chess.KING:
                    valuable_targets += 1
        
        return valuable_targets >= 2
    
    def _creates_pin(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates a pin"""
        test_board = board.copy()
        test_board.push(move)
        
        piece = test_board.piece_at(move.to_square)
        if not piece:
            return False
        
        # Simplified pin detection
        if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            return any(test_board.is_pinned(not piece.color, sq) 
                      for sq in chess.SQUARES 
                      if test_board.piece_at(sq) and test_board.piece_at(sq).color != piece.color)
        
        return False
    
    def _creates_discovered_attack(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates a discovered attack"""
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return False
        
        # Check if moving piece was blocking an attack
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == moving_piece.color:
                if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    # Check if the piece's attack goes through the moving piece's square
                    attacks_before = board.attacks(square)
                    test_board = board.copy()
                    test_board.push(move)
                    attacks_after = test_board.attacks(square)
                    
                    if len(attacks_after) > len(attacks_before):
                        return True
        
        return False
    
    def _threatens_mate_in_n(self, board: chess.Board, n: int) -> bool:
        """Simplified mate detection"""
        if n == 1:
            return any(board.is_checkmate() for move in board.legal_moves 
                      if (test_board := board.copy()) or test_board.push(move) or True)
        
        # For n > 1, this would require more complex analysis
        # Simplified version just checks if we're in check
        return board.is_check()
    
    # Helper methods for king safety
    def _count_pawn_shield(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        """Count pawns protecting the king"""
        shield_count = 0
        king_rank, king_file = chess.square_rank(king_square), chess.square_file(king_square)
        
        # Check squares in front of king
        direction = 1 if color == chess.WHITE else -1
        for file_offset in [-1, 0, 1]:
            file = king_file + file_offset
            rank = king_rank + direction
            
            if 0 <= file <= 7 and 0 <= rank <= 7:
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_count += 1
        
        return shield_count
    
    def _count_king_attackers(self, board: chess.Board, king_square: chess.Square, attacking_color: chess.Color) -> int:
        """Count pieces attacking squares near the king"""
        attacker_count = 0
        
        # Check 3x3 grid around king
        king_rank, king_file = chess.square_rank(king_square), chess.square_file(king_square)
        
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                rank = king_rank + rank_offset
                file = king_file + file_offset
                
                if 0 <= rank <= 7 and 0 <= file <= 7:
                    square = chess.square(file, rank)
                    attacker_count += len(board.attackers(attacking_color, square))
        
        return attacker_count
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn is passed (simplified)"""
        pawn_rank, pawn_file = chess.square_rank(pawn_square), chess.square_file(pawn_square)
        direction = 1 if color == chess.WHITE else -1
        
        # Check files to left, center, and right
        for file_offset in [-1, 0, 1]:
            file = pawn_file + file_offset
            if 0 <= file <= 7:
                # Check all ranks ahead of the pawn
                rank = pawn_rank + direction
                while 0 <= rank <= 7:
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color != color:
                        return False
                    rank += direction
        
        return True