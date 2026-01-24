"""
Train teacher model (ResNet-34) on CIFAR-10.

Usage:
    python src/train_teacher.py --epochs 100 --lr 0.1
"""

import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import get_teacher, count_parameters
from utils import (
    set_seed, get_dataloader, get_num_classes, save_checkpoint,
    get_lr_scheduler, AverageMeter, accuracy
)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = accuracy(outputs, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}%'})
    
    return losses.avg, top1.avg


def validate(model, test_loader, criterion, device):
    """Validate model on test set."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc = accuracy(outputs, labels)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))
    
    return losses.avg, top1.avg


def main(args):
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Data
    num_classes = get_num_classes(args.dataset)
    train_loader, test_loader = get_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Dataset: {args.dataset.upper()} ({num_classes} classes)")
    
    # Model
    model = get_teacher(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print(f"Teacher model: {count_parameters(model):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Tensorboard
    exp_name = f'teacher_{args.dataset}'
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name))
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(args.checkpoint_dir, f'teacher_{args.dataset}_best.pth')
            )
            print(f"New best accuracy: {best_acc:.2f}%")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(args.checkpoint_dir, f'teacher_{args.dataset}_epoch{epoch}.pth')
            )
    
    writer.close()
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--save-freq', type=int, default=20)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
