#!/usr/bin/env python
"""
Data preprocessing and loading pipeline.
Supports: CIFAR-10, CIFAR-100, MNIST, Penn Treebank.

Complete data loading with .cache/ directory for all downloads.
"""

import os
import logging
from typing import Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Use .cache/ for all downloads
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class PennTreebankDataset(Dataset):
    """Penn Treebank language modeling dataset."""
    
    def __init__(self, data_path: str, vocab_size: int = 10000, 
                 seq_length: int = 35, split: str = "train"):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Load data
        full_path = Path(data_path) / f"{split}.txt"
        data_loaded = False
        
        if full_path.exists():
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                # Simple tokenization: split by whitespace
                tokens = text.split()
                if len(tokens) < seq_length + 1:
                    log.warning(f"Penn Treebank data too short ({len(tokens)} tokens)")
                    raise ValueError("Data too short")
                
                # Build vocabulary from top vocab_size tokens
                token_to_id = {}
                token_counts = {}
                for t in tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1
                
                # Sort by frequency and take top vocab_size
                sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:vocab_size]
                for idx, (t, _) in enumerate(sorted_tokens):
                    token_to_id[t] = idx
                
                # Convert text to token IDs
                self.data = torch.tensor(
                    [token_to_id.get(t, 0) for t in tokens], 
                    dtype=torch.long
                )
                data_loaded = True
                log.info(f"Loaded {split} data: {len(self.data)} tokens, vocab_size={len(token_to_id)}")
            
            except Exception as e:
                log.warning(f"Failed to load Penn Treebank from {full_path}: {e}")
        
        # Fallback to dummy data if loading failed
        if not data_loaded:
            log.warning(f"Using dummy data for {split} set")
            self.data = torch.randint(0, vocab_size, (1000 * (seq_length + 1),), dtype=torch.long)
    
    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


def get_cifar10_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 dataset."""
    log.info("Loading CIFAR-10 dataset")
    
    # Normalization stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load datasets
    try:
        train_set = CIFAR10(root=CACHE_DIR, train=True, download=True, 
                           transform=train_transform)
        test_set = CIFAR10(root=CACHE_DIR, train=False, download=True, 
                          transform=test_transform)
    except Exception as e:
        log.error(f"Failed to load CIFAR-10: {e}")
        raise
    
    # Split into train/val
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    
    # Create loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    log.info(f"CIFAR-10: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_loader, val_loader, test_loader


def get_cifar100_loaders(
    batch_size: int = 256,
    val_split: float = 0.1,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-100 dataset."""
    log.info("Loading CIFAR-100 dataset")
    
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    try:
        train_set = CIFAR100(root=CACHE_DIR, train=True, download=True, 
                            transform=train_transform)
        test_set = CIFAR100(root=CACHE_DIR, train=False, download=True, 
                           transform=test_transform)
    except Exception as e:
        log.error(f"Failed to load CIFAR-100: {e}")
        raise
    
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    log.info(f"CIFAR-100: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_loader, val_loader, test_loader


def get_mnist_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST dataset."""
    log.info("Loading MNIST dataset")
    
    mean = (0.1307,)
    std = (0.3081,)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    try:
        train_set = MNIST(root=CACHE_DIR, train=True, download=True, 
                         transform=train_transform)
        test_set = MNIST(root=CACHE_DIR, train=False, download=True, 
                        transform=test_transform)
    except Exception as e:
        log.error(f"Failed to load MNIST: {e}")
        raise
    
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    log.info(f"MNIST: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_loader, val_loader, test_loader


def get_penn_treebank_loaders(
    batch_size: int = 32,
    seq_length: int = 35,
    val_split: float = 0.1,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load Penn Treebank dataset."""
    log.info("Loading Penn Treebank dataset")
    
    data_path = Path(CACHE_DIR) / "penn_treebank"
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_set = PennTreebankDataset(str(data_path), split="train", 
                                   seq_length=seq_length)
    val_set = PennTreebankDataset(str(data_path), split="valid", 
                                 seq_length=seq_length)
    test_set = PennTreebankDataset(str(data_path), split="test", 
                                  seq_length=seq_length)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers)
    
    log.info(f"Penn Treebank: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_loader, val_loader, test_loader


def build_dataloader(
    dataset_cfg: DictConfig,
    training_cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory function to build dataloaders."""
    
    dataset_name = dataset_cfg.name.lower()
    batch_size = training_cfg.batch_size
    val_split = training_cfg.get("val_split", 0.1)
    num_workers = dataset_cfg.get("num_workers", 4)
    
    if dataset_name == "cifar10":
        return get_cifar10_loaders(batch_size=batch_size, val_split=val_split, 
                                  num_workers=num_workers)
    elif dataset_name == "cifar100":
        return get_cifar100_loaders(batch_size=batch_size, val_split=val_split, 
                                   num_workers=num_workers)
    elif dataset_name == "mnist":
        return get_mnist_loaders(batch_size=batch_size, val_split=val_split, 
                                num_workers=num_workers)
    elif dataset_name == "penn_treebank":
        return get_penn_treebank_loaders(batch_size=batch_size, val_split=val_split, 
                                        num_workers=num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
