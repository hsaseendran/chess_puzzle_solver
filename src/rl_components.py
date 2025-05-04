import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Any
import logging


class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Tuple[Any, ...]) -> None:
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size ({len(self.buffer)}) is smaller than batch size ({batch_size})")
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Unpack and organize experiences
        states, actions, rewards, next_states, dones, old_log_probs = zip(
            *[self.buffer[idx] for idx in indices]
        )
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(old_log_probs, dtype=torch.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all experiences from buffer"""
        self.buffer.clear()


class PPOTrainer:
    """Proximal Policy Optimization trainer"""
    
    def __init__(self, policy_network: nn.Module, value_network: nn.Module, config: Any):
        self.policy_network = policy_network
        self.value_network = value_network
        self.config = config
        
        self.policy_optimizer = torch.optim.Adam(
            policy_network.parameters(), 
            lr=config.policy_lr
        )
        self.value_optimizer = torch.optim.Adam(
            value_network.parameters(), 
            lr=config.value_lr
        )
        
        self.clip_epsilon = config.clip_epsilon
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        
        # Training statistics
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, 
                    gamma: float = 0.99, lambda_: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # Calculate TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # Calculate advantage
            advantages[t] = last_advantage = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
            
            # Calculate return (for value function training)
            returns[t] = rewards[t] + gamma * (1 - dones[t]) * last_value
            last_value = returns[t]
        
        return advantages, returns
    
    def update(self, experiences: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Perform PPO update"""
        states, actions, rewards, next_states, dones, old_log_probs = experiences
        
        # Move to device
        device = next(self.policy_network.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        old_log_probs = old_log_probs.to(device)
        
        # Get current value estimates
        with torch.no_grad():
            old_values = self.value_network(states).squeeze()
            next_values = self.value_network(next_states).squeeze()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.ppo_epochs):
            # Get current policy predictions
            # Note: This is simplified. In practice, you'd need to handle variable number of moves per state
            action_probs = []
            for i in range(len(states)):
                # Here we would need the actual move features for each state
                # This is a simplified version
                state_probs = self.policy_network(states[i], [states[i]])  # Placeholder
                action_probs.append(state_probs)
            
            action_probs = torch.stack(action_probs)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            
            # Compute ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value loss
            values = self.value_network(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            
            # Combined loss
            total_loss = (
                policy_loss + 
                self.value_loss_coef * value_loss - 
                self.entropy_coef * entropy
            )
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Accumulate statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Average statistics over PPO epochs
        stats = {
            'policy_loss': total_policy_loss / self.config.ppo_epochs,
            'value_loss': total_value_loss / self.config.ppo_epochs,
            'entropy': total_entropy / self.config.ppo_epochs,
            'total_loss': (total_policy_loss + self.value_loss_coef * total_value_loss - 
                          self.entropy_coef * total_entropy) / self.config.ppo_epochs
        }
        
        # Update running statistics
        for key, value in stats.items():
            self.stats[key].append(value)
        
        return stats
    
    def get_action(self, policy_network: nn.Module, state_features: torch.Tensor, 
                   move_features_list: List[torch.Tensor]) -> Tuple[int, float]:
        """Get action from policy network"""
        with torch.no_grad():
            action_probs = policy_network(state_features, move_features_list)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action.item(), log_prob.item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'stats': self.stats
        }
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.stats = checkpoint['stats']
        logging.info(f"Checkpoint loaded from {filepath}")