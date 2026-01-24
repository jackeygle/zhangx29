"""
Train student model with or without Knowledge Distillation.

Usage:
    # Train with KD
    python src/train_student.py --teacher-ckpt checkpoints/teacher_best.pth
    
    # Train without KD (baseline)
    python src/train_student.py --no-kd
"""

import argparse
import copy
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import get_teacher, get_student, count_parameters, load_checkpoint
from distillation import (
    DistillationLoss, StudentLoss, FeatureDistillationLoss,
    AttentionTransferLoss, ContrastiveDistillationLoss
)
from utils import (
    set_seed, get_dataloader, get_num_classes, save_checkpoint,
    get_lr_scheduler, AverageMeter, accuracy, get_module_by_name, FeatureHook
)


def _match_spatial_size(student_feat, teacher_feat):
    if student_feat.dim() != 4 or teacher_feat.dim() != 4:
        return student_feat, teacher_feat
    target_h = min(student_feat.size(2), teacher_feat.size(2))
    target_w = min(student_feat.size(3), teacher_feat.size(3))
    if (student_feat.size(2), student_feat.size(3)) != (target_h, target_w):
        student_feat = F.adaptive_avg_pool2d(student_feat, (target_h, target_w))
    if (teacher_feat.size(2), teacher_feat.size(3)) != (target_h, target_w):
        teacher_feat = F.adaptive_avg_pool2d(teacher_feat, (target_h, target_w))
    return student_feat, teacher_feat


def update_ema_teacher(ema_teacher, student, decay):
    with torch.no_grad():
        for ema_param, student_param in zip(ema_teacher.parameters(), student.parameters()):
            ema_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)


def train_epoch_distill(
    student,
    teacher,
    train_loader,
    hard_criterion,
    distill_criterion,
    optimizer,
    device,
    distill_method,
    alpha,
    hooks=None,
    ema_teacher=None,
    ema_decay=0.999,
    max_batches=None
):
    """Train student with selected distillation method for one epoch."""
    student.train()
    if teacher is not None:
        teacher.eval()
    if ema_teacher is not None:
        ema_teacher.eval()
    
    losses = AverageMeter()
    hard_losses = AverageMeter()
    distill_losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Training ({distill_method.upper()})')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        if distill_method in {'kd', 'self'}:
            with torch.no_grad():
                teacher_logits = ema_teacher(images) if distill_method == 'self' else teacher(images)
            student_logits = student(images)
            loss, loss_dict = distill_criterion(student_logits, teacher_logits, labels)
            hard_loss = loss_dict['hard_loss']
            distill_loss = loss_dict['soft_loss']
        else:
            with torch.no_grad():
                _ = teacher(images)
                teacher_feat = hooks['teacher'].features
            student_logits = student(images)
            student_feat = hooks['student'].features
            student_feat, teacher_feat = _match_spatial_size(student_feat, teacher_feat)

            hard_loss = hard_criterion(student_logits, labels)
            distill_loss = distill_criterion(student_feat, teacher_feat)
            loss = alpha * hard_loss + (1 - alpha) * distill_loss

            loss_dict = {
                'total_loss': loss.item(),
                'hard_loss': hard_loss.item(),
                'soft_loss': distill_loss.item()
            }

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if distill_method == 'self' and ema_teacher is not None:
            update_ema_teacher(ema_teacher, student, decay=ema_decay)
        
        # Metrics
        acc = accuracy(student_logits, labels)[0]
        losses.update(loss_dict['total_loss'], images.size(0))
        hard_losses.update(loss_dict['hard_loss'], images.size(0))
        distill_losses.update(loss_dict['soft_loss'], images.size(0))
        top1.update(acc, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%'
        })
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    
    return {
        'loss': losses.avg,
        'hard_loss': hard_losses.avg,
        'soft_loss': distill_losses.avg,
        'accuracy': top1.avg
    }


def train_epoch_baseline(student, train_loader, criterion, optimizer, device, max_batches=None):
    """Train student without distillation for one epoch."""
    student.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training (Baseline)')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = student(images)
        loss, loss_dict = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = accuracy(outputs, labels)[0]
        losses.update(loss_dict['total_loss'], images.size(0))
        top1.update(acc, images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}%'})
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    
    return {'loss': losses.avg, 'accuracy': top1.avg}


def validate(model, test_loader, device, max_batches=None):
    """Validate model on test set."""
    model.eval()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc='Validating')):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc = accuracy(outputs, labels)[0]
            top1.update(acc, images.size(0))
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    
    return top1.avg


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
    
    # Student model
    student = get_student(num_classes=num_classes, width_mult=args.width_mult)
    student = student.to(device)
    print(f"Student model: {count_parameters(student):,} parameters")
    
    # Teacher model (if using KD)
    teacher = None
    ema_teacher = None
    if not args.no_kd and args.distill_method != 'self':
        teacher = get_teacher(num_classes=num_classes, pretrained=False)
        teacher = load_checkpoint(teacher, args.teacher_ckpt)
        teacher = teacher.to(device)
        teacher.eval()
        print(f"Teacher model loaded from: {args.teacher_ckpt}")
        print(f"Teacher parameters: {count_parameters(teacher):,}")

    # Loss function and experiment name
    if args.no_kd:
        hard_criterion = None
        distill_criterion = StudentLoss()
        exp_name = f'student_baseline_{args.dataset}_w{args.width_mult}'
    else:
        hard_criterion = nn.CrossEntropyLoss()
        if args.distill_method in {'kd', 'self'}:
            distill_criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
        elif args.distill_method == 'fitnets':
            distill_criterion = None
        elif args.distill_method == 'attention':
            distill_criterion = AttentionTransferLoss(weight=args.feature_weight)
        elif args.distill_method == 'contrastive':
            distill_criterion = None
        else:
            raise ValueError(f"Unknown distill method: {args.distill_method}")

        exp_name = (
            f'student_{args.distill_method}_{args.dataset}_'
            f'T{args.temperature}_a{args.alpha}_w{args.width_mult}'
        )

    # Feature hooks for feature-based distillation
    hooks = None
    if not args.no_kd and args.distill_method in {'fitnets', 'attention', 'contrastive'}:
        teacher_module = get_module_by_name(teacher, args.teacher_layer)
        student_module = get_module_by_name(student, args.student_layer)
        hooks = {
            'teacher': FeatureHook(teacher_module),
            'student': FeatureHook(student_module)
        }

        # Warmup to infer channel dims for projector
        with torch.no_grad():
            images, _ = next(iter(train_loader))
            images = images.to(device)
            _ = teacher(images)
            _ = student(images)
        teacher_feat = hooks['teacher'].features
        student_feat = hooks['student'].features
        if teacher_feat is None or student_feat is None:
            raise RuntimeError("Feature hooks did not capture outputs. Check layer names.")

        if args.distill_method == 'fitnets':
            distill_criterion = FeatureDistillationLoss(
                student_channels=student_feat.size(1),
                teacher_channels=teacher_feat.size(1),
                weight=args.feature_weight
            )
        elif args.distill_method == 'contrastive':
            student_dim = student_feat.size(1)
            teacher_dim = teacher_feat.size(1)
            distill_criterion = ContrastiveDistillationLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                temperature=args.contrastive_temp,
                weight=args.feature_weight
            )

    # Self-distillation uses EMA teacher
    if not args.no_kd and args.distill_method == 'self':
        ema_teacher = copy.deepcopy(student).to(device)
        for param in ema_teacher.parameters():
            param.requires_grad = False
    
    # Optimizer
    optimizer_params = list(student.parameters())
    if distill_criterion is not None:
        distill_criterion = distill_criterion.to(device)
        extra_params = [p for p in distill_criterion.parameters() if p.requires_grad]
        if extra_params:
            optimizer_params += extra_params
    optimizer = optim.SGD(
        optimizer_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name))
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        if args.no_kd:
            train_metrics = train_epoch_baseline(
                student, train_loader, distill_criterion, optimizer, device,
                max_batches=args.max_train_batches
            )
        else:
            train_metrics = train_epoch_distill(
                student, teacher, train_loader, hard_criterion, distill_criterion,
                optimizer, device, args.distill_method, args.alpha,
                hooks=hooks, ema_teacher=ema_teacher, ema_decay=args.ema_decay,
                max_batches=args.max_train_batches
            )
        
        # Validate
        val_acc = validate(student, test_loader, device, max_batches=args.max_val_batches)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if not args.no_kd:
            writer.add_scalar('Loss/hard', train_metrics['hard_loss'], epoch)
            writer.add_scalar('Loss/soft', train_metrics['soft_loss'], epoch)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        
        # Save best checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(
                student, optimizer, epoch, val_acc,
                os.path.join(args.checkpoint_dir, f'{exp_name}_best.pth')
            )
            print(f"New best accuracy: {best_acc:.2f}%")
    
    writer.close()
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    
    # Save final results
    results = {
        'experiment': exp_name,
        'best_accuracy': best_acc,
        'parameters': count_parameters(student),
        'dataset': args.dataset,
        'use_kd': not args.no_kd,
        'distill_method': None if args.no_kd else args.distill_method,
        'temperature': args.temperature if not args.no_kd else None,
        'alpha': args.alpha if not args.no_kd else None,
        'teacher_layer': args.teacher_layer if hooks is not None else None,
        'student_layer': args.student_layer if hooks is not None else None,
        'feature_weight': args.feature_weight if not args.no_kd else None,
        'contrastive_temp': args.contrastive_temp if args.distill_method == 'contrastive' else None,
    }
    
    results_file = os.path.join(args.checkpoint_dir, f'{exp_name}_results.txt')
    with open(results_file, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Student Model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model
    parser.add_argument('--width-mult', type=float, default=1.0,
                        help='Width multiplier for MobileNetV2')
    
    # Teacher (for KD)
    parser.add_argument('--teacher-ckpt', type=str, default='./checkpoints/teacher_best.pth')
    parser.add_argument('--no-kd', action='store_true', help='Train without KD (baseline)')
    
    # Distillation
    parser.add_argument('--distill-method', type=str, default='kd',
                        choices=['kd', 'fitnets', 'attention', 'self', 'contrastive'])
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--feature-weight', type=float, default=1.0)
    parser.add_argument('--contrastive-temp', type=float, default=0.1)
    parser.add_argument('--teacher-layer', type=str, default='layer4')
    parser.add_argument('--student-layer', type=str, default='features.18')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    
    args = parser.parse_args()
    main(args)
