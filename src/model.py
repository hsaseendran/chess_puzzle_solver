import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block as used in AlphaZero/LeelaChessZero"""
    def __init__(self, channels=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessPuzzleNet(nn.Module):
    """
    Neural network for chess puzzle solving
    Architecture based on AlphaZero with Maia-style training
    """
    def __init__(self, num_blocks=20, channels=256):
        super(ChessPuzzleNet, self).__init__()
        
        # Input convolution
        self.conv_input = nn.Conv2d(12, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        # Policy head (move prediction)
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # 4096 = 64*64 possible moves
        
        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value