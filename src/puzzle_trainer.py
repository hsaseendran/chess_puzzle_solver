import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import chess
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import os
import json
from datetime import datetime

from src.board_manager import BoardManager
from src.feature_extractor import FeatureExtractor
from src.networks import PolicyNetwork, ValueNetwork
from src.stockfish_evaluator import StockfishEvaluator
from src.rl_components import PPOTrainer, ReplayBuffer
from src.config import TrainingConfig


class PuzzleTrainer:
    """Main training pipeline for chess puzzles"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.board_manager = BoardManager()
        self.feature_extractor = FeatureExtractor()
        self.policy_network = PolicyNetwork(config).to(self.device)
        self.value_network = ValueNetwork(config).to(self.device)
        self.stockfish = StockfishEvaluator(config.stockfish_path, config.stockfish_depth)
        self.trainer = PPOTrainer(self.policy_network, self.value_network, config)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity)
        
        # Metrics tracking
        self.metrics = {
            'puzzle_accuracy': [],
            'average_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'stockfish_agreement': []
        }
        
        self.episode_count = 0
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(
            self.config.log_dir, 
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train_on_puzzle(self, puzzle_data: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Train on a single puzzle"""
        try:
            fen = puzzle_data['FEN']
            moves = puzzle_data['Moves'].split()
            
            # Set up position (after opponent's move)
            self.board_manager.set_position(fen)
            
            # The first move in the puzzle is the opponent's move (setting up the puzzle)
            if len(moves) > 0:
                opponent_move = chess.Move.from_uci(moves[0])
                self.board_manager.make_move(opponent_move)
            
            # The correct answer is the next move
            correct_move = chess.Move.from_uci(moves[1]) if len(moves) > 1 else None
            
            # Get legal moves
            legal_moves = self.board_manager.get_legal_moves()
            
            if not legal_moves:
                logging.warning(f"No legal moves for puzzle {puzzle_data.get('PuzzleId', 'unknown')}")
                return False, 0.0, {}
            
            # Extract features
            board_features = self.feature_extractor.extract_board_features(self.board_manager.board)
            move_features = []
            for move in legal_moves:
                move_feature = self.feature_extractor.extract_move_features(self.board_manager.board, move)
                move_features.append(move_feature)
            
            # Move to device
            board_features = board_features.to(self.device)
            move_features_device = [mf.to(self.device) for mf in move_features]
            
            # Get policy prediction
            with torch.no_grad():
                move_probs = self.policy_network(board_features, move_features_device)
            
            # Sample action
            action_dist = torch.distributions.Categorical(move_probs)
            action_idx = action_dist.sample()
            log_prob = action_dist.log_prob(action_idx)
            
            selected_move = legal_moves[action_idx]
            
            # Get Stockfish evaluation
            stockfish_best = self.stockfish.get_best_move(self.board_manager.board)
            stockfish_eval = self.stockfish.evaluate_move(self.board_manager.board, selected_move)
            
            # Calculate reward
            reward = self._calculate_reward(
                selected_move, 
                correct_move, 
                stockfish_best, 
                stockfish_eval,
                self.board_manager.board
            )
            
            # Store experience
            experience = (
                board_features.cpu(),
                action_idx.item(),
                reward,
                board_features.cpu(),  # Next state (simplified)
                False,  # Not done (simplified)
                log_prob.item()
            )
            self.replay_buffer.push(experience)
            
            # Train if buffer is sufficient
            train_stats = {}
            if len(self.replay_buffer) >= self.config.min_buffer_size:
                if self.episode_count % self.config.batch_size == 0:
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    train_stats = self.trainer.update(batch)
            
            # Update metrics
            is_correct = selected_move == correct_move
            stockfish_agreement = selected_move == stockfish_best
            
            puzzle_stats = {
                'is_correct': is_correct,
                'stockfish_agreement': stockfish_agreement,
                'reward': reward,
                'selected_move': selected_move.uci(),
                'correct_move': correct_move.uci() if correct_move else None,
                'stockfish_best': stockfish_best.uci() if stockfish_best else None,
                **train_stats
            }
            
            self.episode_count += 1
            
            return is_correct, reward, puzzle_stats
            
        except Exception as e:
            logging.error(f"Error training on puzzle: {e}")
            return False, 0.0, {}
    
    def _calculate_reward(self, selected_move: chess.Move, correct_move: chess.Move, 
                         stockfish_best: chess.Move, stockfish_eval: float,
                         board: chess.Board) -> float:
        """Calculate reward for a move"""
        reward = 0.0
        
        # Base reward from Stockfish evaluation
        reward += stockfish_eval * self.config.stockfish_weight
        
        # Bonus for correct puzzle move
        if selected_move == correct_move:
            reward += self.config.correct_move_bonus
        
        # Bonus for agreeing with Stockfish
        if selected_move == stockfish_best:
            reward += 0.5
        
        # Check for checkmate
        test_board = board.copy()
        test_board.push(selected_move)
        if test_board.is_checkmate():
            reward += self.config.mate_bonus
        
        # Penalty for blunders (significantly worse than best move)
        if stockfish_eval < -0.5:
            reward -= 0.5
        
        return reward
    
    def train_epoch(self, puzzle_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train on full dataset epoch"""
        correct_predictions = 0
        total_rewards = []
        stockfish_agreements = 0
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        for puzzle in tqdm(puzzle_dataset, desc="Training"):
            is_correct, reward, puzzle_stats = self.train_on_puzzle(puzzle)
            
            correct_predictions += is_correct
            total_rewards.append(reward)
            
            if puzzle_stats.get('stockfish_agreement', False):
                stockfish_agreements += 1
            
            # Accumulate training statistics
            for key in ['policy_loss', 'value_loss', 'entropy']:
                if key in puzzle_stats:
                    epoch_stats[key].append(puzzle_stats[key])
        
        # Calculate epoch metrics
        epoch_metrics = {
            'accuracy': correct_predictions / len(puzzle_dataset),
            'average_reward': np.mean(total_rewards),
            'stockfish_agreement': stockfish_agreements / len(puzzle_dataset),
            'policy_loss': np.mean(epoch_stats['policy_loss']) if epoch_stats['policy_loss'] else 0,
            'value_loss': np.mean(epoch_stats['value_loss']) if epoch_stats['value_loss'] else 0,
            'entropy': np.mean(epoch_stats['entropy']) if epoch_stats['entropy'] else 0
        }
        
        # Update metrics history
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        return epoch_metrics
    
    def evaluate(self, eval_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        self.policy_network.eval()
        self.value_network.eval()
        
        correct_predictions = 0
        stockfish_agreements = 0
        total_rewards = []
        
        with torch.no_grad():
            for puzzle in tqdm(eval_dataset, desc="Evaluating"):
                is_correct, reward, puzzle_stats = self.train_on_puzzle(puzzle)
                
                correct_predictions += is_correct
                total_rewards.append(reward)
                
                if puzzle_stats.get('stockfish_agreement', False):
                    stockfish_agreements += 1
        
        self.policy_network.train()
        self.value_network.train()
        
        return {
            'accuracy': correct_predictions / len(eval_dataset),
            'average_reward': np.mean(total_rewards),
            'stockfish_agreement': stockfish_agreements / len(eval_dataset)
        }
    
    def train(self, train_dataset: List[Dict[str, Any]], 
              val_dataset: List[Dict[str, Any]]) -> None:
        """Full training loop"""
        logging.info(f"Starting training with {len(train_dataset)} training puzzles")
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training epoch
            epoch_metrics = self.train_epoch(train_dataset)
            
            # Log metrics
            logging.info(
                f"Train - Accuracy: {epoch_metrics['accuracy']:.4f}, "
                f"Reward: {epoch_metrics['average_reward']:.4f}, "
                f"Stockfish Agreement: {epoch_metrics['stockfish_agreement']:.4f}"
            )
            
            # Validation
            if epoch % self.config.eval_frequency == 0:
                val_metrics = self.evaluate(val_dataset)
                logging.info(
                    f"Val - Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"Reward: {val_metrics['average_reward']:.4f}, "
                    f"Stockfish Agreement: {val_metrics['stockfish_agreement']:.4f}"
                )
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    self.save_checkpoint(os.path.join(self.config.model_dir, 'best_model.pt'))
                    logging.info("New best model saved!")
            
            # Regular checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(
                    os.path.join(self.config.model_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                )
        
        logging.info("Training completed!")
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.episode_count // self.config.batch_size,
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
        
        # Also save metrics to JSON for easy visualization
        metrics_path = filepath.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.metrics = checkpoint['metrics']
        
        if 'epoch' in checkpoint:
            self.episode_count = checkpoint['epoch'] * self.config.batch_size
        
        logging.info(f"Checkpoint loaded from {filepath}")