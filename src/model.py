#!/usr/bin/env python
"""
Model architecture implementations for AGVR experiments.
Supports: ResNet-18/50 (CIFAR-10/100), 3-layer MLP (MNIST), 2-layer LSTM (Penn Treebank).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class MLPNet(nn.Module):
    """Simple 3-layer MLP for MNIST."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 num_classes: int = 10):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50."""
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, 
                              bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture."""
    
    def __init__(self, block, num_blocks: list, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LSTMLanguageModel(nn.Module):
    """2-layer LSTM for Penn Treebank language modeling."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 400, 
                 hidden_dim: int = 1150, num_layers: int = 2, dropout: float = 0.5):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits


def build_model(model_cfg: DictConfig) -> nn.Module:
    """Factory function to build model from config."""
    
    model_name = model_cfg.name.lower()
    
    if model_name == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], 
                      num_classes=model_cfg.get("num_classes", 10))
    elif model_name == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3], 
                      num_classes=model_cfg.get("num_classes", 100))
    elif model_name == "mlp":
        model = MLPNet(
            input_size=model_cfg.get("input_size", 784),
            hidden_size=model_cfg.get("hidden_size", 256),
            num_classes=model_cfg.get("num_classes", 10)
        )
    elif model_name == "lstm":
        model = LSTMLanguageModel(
            vocab_size=model_cfg.get("vocab_size", 10000),
            embedding_dim=model_cfg.get("embedding_dim", 400),
            hidden_dim=model_cfg.get("hidden_dim", 1150),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.5)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model
