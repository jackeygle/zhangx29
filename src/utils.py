"""
Utility functions for data loading, training helpers, and logging.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple
import yaml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, test_loader
    """
    return get_dataloader('cifar10', data_dir, batch_size, num_workers)


def get_dataloader(
    dataset: str = 'cifar10',
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test data loaders for CIFAR-10 or CIFAR-100.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, test_loader
    """
    # CIFAR-10 and CIFAR-100 have the same normalization stats
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Select dataset
    if dataset.lower() == 'cifar10':
        DatasetClass = torchvision.datasets.CIFAR10
    elif dataset.lower() == 'cifar100':
        DatasetClass = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'cifar10' or 'cifar100'")
    
    train_dataset = DatasetClass(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = DatasetClass(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_num_classes(dataset: str) -> int:
    """Get number of classes for a dataset."""
    if dataset.lower() == 'cifar10':
        return 10
    elif dataset.lower() == 'cifar100':
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    path: str
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, path)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_module_by_name(model: nn.Module, module_path: str) -> nn.Module:
    """
    Get a nested module from a model by a dot-separated path.

    Supports integer indices for Sequential/ModuleList.
    Example: "features.18" or "layer4.1.conv2"
    """
    current = model
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


class FeatureHook:
    """Capture forward outputs from a module."""

    def __init__(self, module: nn.Module):
        self.features = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _input, output):
        self.features = output

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


if __name__ == "__main__":
    # Test data loading
    print("Testing CIFAR-10 data loading...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
