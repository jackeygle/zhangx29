# Knowledge Distillation for Neural Network Compression

A PyTorch implementation of Knowledge Distillation on CIFAR-10, demonstrating how smaller models can learn from larger models.

## Project Structure
```
knowledge-distillation/
├── src/
│   ├── models.py           # Teacher & Student models
│   ├── distillation.py     # KD loss
│   ├── train_teacher.py    # Train teacher
│   ├── train_student.py    # Train student with/without KD
│   └── evaluate.py         # Evaluation
├── scripts/                # SLURM job scripts
└── results/                # Experiment outputs
```

## Quick Start

```bash
# Train teacher model
sbatch scripts/train_teacher.sh

# Train student without KD (baseline)
sbatch scripts/train_student.sh --no-kd

# Train student with KD
sbatch scripts/train_student.sh
```

## Results

| Model | Accuracy | Parameters | Compression |
|-------|----------|------------|-------------|
| Teacher (ResNet-34) | ~94% | 21M | 1.0x |
| Student baseline | ~85% | 3.5M | 6x |
| Student + KD | ~90% | 3.5M | 6x |

## Knowledge Distillation

The core idea: train a small "student" model to mimic a large "teacher" model's soft predictions.

```
L_total = α × L_CE(student, labels) + (1-α) × T² × L_KL(soft_student, soft_teacher)
```

- **Temperature (T)**: Softens probability distributions
- **Alpha (α)**: Balance between hard and soft labels

## Author
Xinle Zhang - Aalto University
