"""
Models for Knowledge Distillation experiments.
Teacher: ResNet-34 (pretrained on ImageNet, fine-tuned on CIFAR-10)
Student: MobileNetV2 (trained from scratch or with KD)
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_teacher(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Get teacher model (ResNet-34).
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        ResNet-34 model adapted for CIFAR-10
    """
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
    else:
        model = models.resnet34(weights=None)
    
    # Adapt for CIFAR-10 (32x32 images)
    # Replace first conv layer: 7x7 -> 3x3, stride 2 -> 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def get_student(num_classes: int = 10, width_mult: float = 1.0) -> nn.Module:
    """
    Get student model (MobileNetV2).
    
    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier for MobileNetV2
    
    Returns:
        MobileNetV2 model adapted for CIFAR-10
    """
    model = models.mobilenet_v2(weights=None, width_mult=width_mult)
    
    # Adapt for CIFAR-10 (32x32 images)
    # Modify first conv layer for smaller input
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_student_small(num_classes: int = 10) -> nn.Module:
    """
    Get a smaller student model (MobileNetV2 with width_mult=0.5).
    Even more compression, good for ablation studies.
    """
    return get_student(num_classes=num_classes, width_mult=0.5)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


if __name__ == "__main__":
    # Test models
    teacher = get_teacher(num_classes=10)
    student = get_student(num_classes=10)
    student_small = get_student_small(num_classes=10)
    
    print(f"Teacher (ResNet-34): {count_parameters(teacher):,} parameters")
    print(f"Student (MobileNetV2): {count_parameters(student):,} parameters")
    print(f"Student Small (MobileNetV2 x0.5): {count_parameters(student_small):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    print(f"\nTeacher output shape: {teacher(x).shape}")
    print(f"Student output shape: {student(x).shape}")
    print(f"Student Small output shape: {student_small(x).shape}")
