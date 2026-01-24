"""
Evaluate trained models and compare results.

Usage:
    python src/evaluate.py --checkpoint checkpoints/teacher_best.pth --model teacher
    python src/evaluate.py --checkpoint checkpoints/student_kd_best.pth --model student
"""

import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from models import get_teacher, get_student, count_parameters, load_checkpoint
from utils import set_seed, get_dataloader, get_num_classes, accuracy, AverageMeter


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy and inference speed."""
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Accuracy
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
    
    # Inference speed (warmup + measure)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs * 1000  # ms
    
    return {
        'top1': top1.avg,
        'top5': top5.avg,
        'inference_time_ms': avg_time
    }


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    num_classes = get_num_classes(args.dataset)
    _, test_loader = get_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model
    if args.model == 'teacher':
        model = get_teacher(num_classes=num_classes, pretrained=False)
    else:
        model = get_student(num_classes=num_classes, width_mult=args.width_mult)
    
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    
    # Get metrics
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*50)
    print(f"Parameters: {num_params:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Top-1 Accuracy: {results['top1']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5']:.2f}%")
    print(f"Inference Time: {results['inference_time_ms']:.3f} ms")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['teacher', 'student'], required=True)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
