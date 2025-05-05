import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
from collections import defaultdict


class PuzzleTrainer:
    """Training pipeline for chess puzzle solver"""
    
    def __init__(self, model, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Metrics
        self.metrics = {
            'train_policy_loss': [],
            'train_value_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_policy_loss': [],
            'val_value_loss': []
        }
        
        self.best_val_accuracy = 0.0
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_policy_loss = 0
        total_value_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (board_tensor, policy_target, value_target, info) in enumerate(progress_bar):
            board_tensor = board_tensor.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device)
            
            # Forward pass
            policy_output, value_output = self.model(board_tensor)
            
            # Calculate losses
            policy_loss = self.policy_criterion(policy_output, policy_target)
            value_loss = self.value_criterion(value_output, value_target)
            
            # Combined loss (value loss weighted less)
            loss = policy_loss + 0.1 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(policy_output, 1)
            _, target = torch.max(policy_target, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += board_tensor.size(0)
            
            # Update metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'policy_loss': f'{policy_loss.item():.4f}',
                'value_loss': f'{value_loss.item():.4f}',
                'accuracy': f'{correct_predictions / total_samples:.4f}'
            })
        
        epoch_policy_loss = total_policy_loss / len(train_loader)
        epoch_value_loss = total_value_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        self.metrics['train_policy_loss'].append(epoch_policy_loss)
        self.metrics['train_value_loss'].append(epoch_value_loss)
        self.metrics['train_accuracy'].append(epoch_accuracy)
        
        return epoch_policy_loss, epoch_value_loss, epoch_accuracy
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_policy_loss = 0
        total_value_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        # Accuracy by rating
        rating_buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for board_tensor, policy_target, value_target, info in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                board_tensor = board_tensor.to(self.device)
                policy_target = policy_target.to(self.device)
                value_target = value_target.to(self.device)
                
                # Forward pass
                policy_output, value_output = self.model(board_tensor)
                
                # Calculate losses
                policy_loss = self.policy_criterion(policy_output, policy_target)
                value_loss = self.value_criterion(value_output, value_target)
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(policy_output, 1)
                _, target = torch.max(policy_target, 1)
                
                correct_mask = (predicted == target)
                correct_predictions += correct_mask.sum().item()
                total_samples += board_tensor.size(0)
                
                # Track accuracy by rating
                for i in range(len(board_tensor)):
                    rating = info['rating'][i].item()
                    bucket = (rating // 500) * 500  # Round to nearest 500
                    rating_buckets[bucket]['total'] += 1
                    if correct_mask[i]:
                        rating_buckets[bucket]['correct'] += 1
        
        epoch_policy_loss = total_policy_loss / len(val_loader)
        epoch_value_loss = total_value_loss / len(val_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        self.metrics['val_policy_loss'].append(epoch_policy_loss)
        self.metrics['val_value_loss'].append(epoch_value_loss)
        self.metrics['val_accuracy'].append(epoch_accuracy)
        
        # Log accuracy by rating
        logging.info("Accuracy by rating:")
        for bucket in sorted(rating_buckets.keys()):
            bucket_data = rating_buckets[bucket]
            if bucket_data['total'] > 0:
                bucket_accuracy = bucket_data['correct'] / bucket_data['total']
                logging.info(f"  {bucket}-{bucket+500}: {bucket_accuracy:.4f} ({bucket_data['total']} puzzles)")
        
        return epoch_accuracy
    
    def train(self, train_dataset, val_dataset, num_epochs=50, batch_size=64, save_dir='checkpoints'):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        for epoch in range(num_epochs):
            # Train
            policy_loss, value_loss, train_accuracy = self.train_epoch(train_loader, epoch + 1)
            
            # Validate
            val_accuracy = self.validate(val_loader, epoch + 1)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            logging.info(f"Train - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            logging.info(f"Val - Accuracy: {val_accuracy:.4f}")
            
            # Save checkpoint
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_checkpoint(f'{save_dir}/best_model.pth', epoch + 1)
                logging.info("New best model saved!")
            
            # Regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'{save_dir}/checkpoint_epoch_{epoch+1}.pth', epoch + 1)
        
        # Save final model
        self.save_checkpoint(f'{save_dir}/final_model.pth', num_epochs)
        
        # Plot training curves
        self.plot_metrics(save_dir)
        
        return self.metrics
    
    def save_checkpoint(self, filepath, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'best_val_accuracy': self.best_val_accuracy
        }
        torch.save(checkpoint, filepath)
        
        # Also save metrics as JSON
        metrics_path = filepath.replace('.pth', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        return checkpoint['epoch']
    
    def plot_metrics(self, save_dir):
        """Plot training metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['train_policy_loss'], label='Train Policy Loss')
        plt.plot(self.metrics['val_policy_loss'], label='Val Policy Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Policy Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['train_value_loss'], label='Train Value Loss')
        plt.plot(self.metrics['val_value_loss'], label='Val Value Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Value Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['train_accuracy'], label='Train Accuracy')
        plt.plot(self.metrics['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 4)
        lrs = [self.optimizer.param_groups[0]['lr']] * len(self.metrics['train_accuracy'])
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()