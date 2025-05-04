import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PolicyNetwork(nn.Module):
    """Policy network for selecting moves"""
    
    def __init__(self, config):
        super().__init__()
        
        # Board state encoder - replacing BatchNorm with LayerNorm for single-sample support
        self.board_encoder = nn.Sequential(
            nn.Linear(config.board_feature_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),  # Changed from BatchNorm1d
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),   # Changed from BatchNorm1d
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Move encoder - replacing BatchNorm with LayerNorm
        self.move_encoder = nn.Sequential(
            nn.Linear(config.move_feature_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),   # Changed from BatchNorm1d
            
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Attention mechanism for move selection
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True  # Added batch_first=True for consistency
        )
        
        # Final policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, board_features: torch.Tensor, move_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            board_features: Features of current board state
            move_features_list: List of features for each legal move
        Returns:
            Probability distribution over moves
        """
        # Handle batch dimension
        if board_features.dim() == 1:
            board_features = board_features.unsqueeze(0)
        
        # Handle edge case: if no moves, return empty tensor
        if not move_features_list:
            return torch.tensor([], device=board_features.device)
        
        # Encode board state
        board_embedding = self.board_encoder(board_features)
        
        # Encode each move
        move_embeddings = []
        for move_features in move_features_list:
            if move_features.dim() == 1:
                move_features = move_features.unsqueeze(0)
            move_embedding = self.move_encoder(move_features)
            move_embeddings.append(move_embedding)
        
        # Stack move embeddings
        move_embeddings = torch.stack(move_embeddings)  # [num_moves, batch_size, embed_dim]
        
        # Transpose for batch_first attention
        move_embeddings = move_embeddings.transpose(0, 1)  # [batch_size, num_moves, embed_dim]
        
        # Use attention to score moves based on board state
        # board_embedding as query, move_embeddings as key/value
        attended_moves, attention_weights = self.attention(
            board_embedding.unsqueeze(1),   # [batch_size, 1, embed_dim]
            move_embeddings,                 # [batch_size, num_moves, embed_dim]
            move_embeddings                  # [batch_size, num_moves, embed_dim]
        )
        
        # Get scores for each move
        move_scores = []
        for i in range(attended_moves.size(1)):
            score = self.policy_head(attended_moves[:, i])
            move_scores.append(score)
        
        if move_scores:
            move_scores = torch.cat(move_scores, dim=0)
        else:
            move_scores = torch.tensor([], device=board_features.device)
        
        # Handle single move case - ensure tensor is at least 1D
        if move_scores.dim() == 0:
            move_scores = move_scores.unsqueeze(0)
        
        # Return probability distribution
        return torch.softmax(move_scores, dim=0)


class ValueNetwork(nn.Module):
    """Value network for position evaluation"""
    
    def __init__(self, config):
        super().__init__()
        
        # Replace BatchNorm with LayerNorm for single-sample support
        self.network = nn.Sequential(
            nn.Linear(config.board_feature_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),  # Changed from BatchNorm1d
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),   # Changed from BatchNorm1d
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),   # Changed from BatchNorm1d
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, board_features: torch.Tensor) -> torch.Tensor:
        """Estimate value of current position"""
        if board_features.dim() == 1:
            board_features = board_features.unsqueeze(0)
        return self.network(board_features)