import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PolicyNetwork(nn.Module):
    """Policy network for selecting moves"""
    
    def __init__(self, config):
        super().__init__()
        
        # Board state encoder
        self.board_encoder = nn.Sequential(
            nn.Linear(config.board_feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Move encoder
        self.move_encoder = nn.Sequential(
            nn.Linear(config.move_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Attention mechanism for move selection
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
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
        
        # Use attention to score moves based on board state
        # board_embedding as query, move_embeddings as key/value
        attended_moves, attention_weights = self.attention(
            board_embedding.unsqueeze(0),  # [1, batch_size, embed_dim]
            move_embeddings,                # [num_moves, batch_size, embed_dim]
            move_embeddings                 # [num_moves, batch_size, embed_dim]
        )
        
        # Get scores for each move
        move_scores = []
        for i in range(attended_moves.size(0)):
            score = self.policy_head(attended_moves[i])
            move_scores.append(score)
        
        move_scores = torch.cat(move_scores, dim=0)
        
        # Return probability distribution
        return torch.softmax(move_scores.squeeze(), dim=0)


class ValueNetwork(nn.Module):
    """Value network for position evaluation"""
    
    def __init__(self, config):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(config.board_feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, board_features: torch.Tensor) -> torch.Tensor:
        """Estimate value of current position"""
        if board_features.dim() == 1:
            board_features = board_features.unsqueeze(0)
        return self.network(board_features)