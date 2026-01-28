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
    Get train and test data loaders for CIFAR-10, CIFAR-100, or Tiny-ImageNet.
    
    Args:
        dataset: 'cifar10', 'cifar100', or 'tinyimagenet'
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, test_loader
    """
    dataset_key = dataset.lower().replace("-", "").replace("_", "")

    if dataset_key in {'cifar10', 'cifar100'}:
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
        if dataset_key == 'cifar10':
            DatasetClass = torchvision.datasets.CIFAR10
        else:
            DatasetClass = torchvision.datasets.CIFAR100

        train_dataset = DatasetClass(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = DatasetClass(
            root=data_dir, train=False, download=True, transform=test_transform
        )
    elif dataset_key == 'tinyimagenet':
        # Tiny-ImageNet (200 classes, 64x64)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        tiny_root = os.path.join(data_dir, 'tiny-imagenet-200')
        train_dir = os.path.join(tiny_root, 'train')
        val_dir = os.path.join(tiny_root, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                "Tiny-ImageNet not found. Run scripts/prepare_tinyimagenet.sh "
                "or place the dataset at data/tiny-imagenet-200."
            )

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Use 'cifar10', 'cifar100', or 'tinyimagenet'"
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
    dataset_key = dataset.lower().replace("-", "").replace("_", "")
    if dataset_key == 'cifar10':
        return 10
    elif dataset_key == 'cifar100':
        return 100
    elif dataset_key == 'tinyimagenet':
        return 200
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


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """
    Get a module from model by its name (e.g., 'features.0').
    
    Args:
        model: PyTorch model
        name: Module name, can be dot-separated (e.g., 'features.0')
    
    Returns:
        The requested module
    """
    names = name.split('.')
    module = model
    for n in names:
        if hasattr(module, n):
            module = getattr(module, n)
        elif hasattr(module, '__getitem__'):
            module = module[int(n)]
        else:
            raise AttributeError(f"Module {model.__class__.__name__} has no attribute '{n}'")
    return module


class FeatureHook:
    """
    Hook to capture intermediate feature maps from a model.
    Used for feature-based distillation methods (FitNets, Attention Transfer).
    """
    
    def __init__(self, module: nn.Module):
        self.module = module
        self.features = None
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Store the output feature map."""
        self.features = output
    
    def remove(self):
        """Remove the hook."""
        self.hook.remove()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


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
