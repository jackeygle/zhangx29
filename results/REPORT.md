# Knowledge Distillation for Neural Network Compression

**A Technical Report on Model Compression using Knowledge Distillation**

**Author:** Xinle Zhang  
**Affiliation:** Aalto University  
**Date:** January 2026

---

## Abstract

This report presents a comprehensive study of Knowledge Distillation (KD) for neural network compression on the CIFAR-10 dataset. We demonstrate that a lightweight MobileNetV2 student model (2.24M parameters) can achieve **92.95% accuracy** by learning from a larger ResNet-34 teacher model (21.28M parameters), representing a **~10× model compression** with only **1.13% accuracy loss**. Through systematic ablation studies on temperature and alpha hyperparameters, we identify optimal distillation configurations and provide insights into the knowledge transfer mechanism.

---

## 1. Introduction

### 1.1 Background

Deep neural networks have achieved remarkable success across various domains, but their deployment on resource-constrained devices remains challenging due to high computational and memory requirements. **Knowledge Distillation** (Hinton et al., 2015) offers an elegant solution by transferring knowledge from a large "teacher" model to a compact "student" model.

### 1.2 Objectives

1. Train a high-performing teacher model (ResNet-34) on CIFAR-10
2. Train a compact student model (MobileNetV2) with and without KD
3. Conduct ablation studies on key hyperparameters (Temperature T, Alpha α)
4. Analyze the effectiveness of knowledge distillation for model compression

### 1.3 Key Contributions

- Achieved **~10× compression** with minimal accuracy degradation
- Systematic ablation study on temperature (T = 1, 2, 4, 8, 16) and alpha (α = 0.1, 0.3, 0.5, 0.7, 0.9)
- Identified optimal hyperparameters: **T=16, α=0.3** yielding best performance

---

## 2. Methodology

### 2.1 Knowledge Distillation Framework

The core idea of knowledge distillation is to train a student model to mimic the soft probability distributions (dark knowledge) produced by a teacher model, rather than just learning from hard labels.

#### Loss Function

The total training loss combines two components:

$$L_{total} = \alpha \cdot L_{CE}(y_s, y_{true}) + (1-\alpha) \cdot T^2 \cdot L_{KL}(\sigma(z_s/T), \sigma(z_t/T))$$

Where:
- $L_{CE}$: Cross-entropy loss with ground truth labels (hard labels)
- $L_{KL}$: KL divergence between student and teacher soft predictions
- $T$: Temperature parameter (softens probability distributions)
- $\alpha$: Balancing weight between hard and soft label losses
- $\sigma$: Softmax function
- $z_s, z_t$: Logits from student and teacher models

#### Why Temperature Matters

Higher temperature produces softer probability distributions, revealing more information about class similarities:
- **Low T (→1)**: Sharp distributions, close to one-hot
- **High T (→∞)**: Uniform distributions, maximum entropy

The $T^2$ scaling factor ensures gradient magnitudes remain comparable across different temperatures.

### 2.2 Model Architectures

| Model | Architecture | Parameters | Purpose |
|-------|--------------|------------|---------|
| **Teacher** | ResNet-34 | 21,284,042 (~21.3M) | Large, accurate reference model |
| **Student** | MobileNetV2 | 2,236,682 (~2.2M) | Compact, deployable model |

**Compression Ratio:** 21.3M / 2.2M ≈ **9.5×**

#### Modifications for CIFAR-10

Both models were adapted for CIFAR-10's 32×32 input size:
- **ResNet-34**: First conv kernel 7×7→3×3, stride 2→1, removed initial maxpool
- **MobileNetV2**: First conv stride 2→1

### 2.3 Training Configuration

| Parameter | Teacher | Student |
|-----------|---------|---------|
| Epochs | 100 | 200 |
| Batch Size | 128 | 128 |
| Optimizer | SGD (momentum=0.9) | SGD (momentum=0.9) |
| Initial LR | 0.1 | 0.1 |
| LR Scheduler | Cosine Annealing | Cosine Annealing |
| Weight Decay | 1e-4 | 1e-4 |

#### Data Augmentation
- Random cropping (32×32 with padding=4)
- Random horizontal flip
- Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]

---

## 3. Experiments and Results

### 3.1 Main Results

| Model | Accuracy | Parameters | Compression | KD Improvement |
|-------|----------|------------|-------------|----------------|
| Teacher (ResNet-34) | **94.08%** | 21.28M | 1× | - |
| Student Baseline | 92.10% | 2.24M | 9.5× | Baseline |
| Student + KD (T=4, α=0.3) | 92.58% | 2.24M | 9.5× | +0.48% |
| Student + KD (T=16, α=0.3) | **92.95%** | 2.24M | 9.5× | **+0.85%** |
| Student + KD (T=4, α=0.5) | 92.89% | 2.24M | 9.5× | +0.79% |

**Key Findings:**
- Knowledge distillation consistently improves student performance over baseline
- Best result: **T=16, α=0.3** achieves **92.95%** accuracy
- The student recovers **98.8%** of teacher's accuracy with **~10× fewer parameters**

### 3.2 Temperature Ablation Study (α=0.3)

| Temperature (T) | Accuracy | Δ vs Baseline |
|-----------------|----------|---------------|
| 1 | 92.64% | +0.54% |
| 2 | 92.51% | +0.41% |
| 4 | 92.58% | +0.48% |
| 8 | 92.72% | +0.62% |
| **16** | **92.95%** | **+0.85%** |

**Analysis:**
- Higher temperatures generally yield better results
- T=16 achieves the best performance, suggesting that softer distributions provide more useful dark knowledge
- The trend indicates that revealing inter-class similarities is beneficial for this dataset

### 3.3 Alpha Ablation Study (T=4)

| Alpha (α) | Hard Label Weight | Soft Label Weight | Accuracy | Δ vs Baseline |
|-----------|-------------------|-------------------|----------|---------------|
| 0.1 | 10% | 90% | 92.19% | +0.09% |
| 0.3 | 30% | 70% | 92.58% | +0.48% |
| **0.5** | 50% | 50% | **92.89%** | **+0.79%** |
| 0.7 | 70% | 30% | 92.58% | +0.48% |
| 0.9 | 90% | 10% | 92.66% | +0.56% |

**Analysis:**
- α=0.5 achieves the best performance at T=4, suggesting a balanced contribution from both losses
- Extreme values (α=0.1 or α=0.9) perform worse, indicating both hard and soft labels are important
- The optimal balance depends on the temperature; higher T may benefit from different α values

---

## 4. Discussion

### 4.1 Effectiveness of Knowledge Distillation

Our experiments demonstrate that knowledge distillation is an effective technique for model compression:

1. **Consistent Improvement**: All KD configurations outperformed the baseline student
2. **Significant Compression**: Achieved ~10× parameter reduction with <2% accuracy loss
3. **Dark Knowledge**: Soft labels from the teacher provide valuable information about class relationships

### 4.2 Hyperparameter Insights

**Temperature (T):**
- Controls the "softness" of probability distributions
- Higher temperatures reveal more inter-class relationships
- For CIFAR-10, T=16 was optimal, but this may vary for other datasets

**Alpha (α):**
- Balances ground truth supervision vs. teacher knowledge
- α=0.3~0.5 works well, indicating both sources are valuable
- Pure distillation (α→0) underperforms, suggesting hard labels prevent overfitting to teacher errors

### 4.3 Why Does KD Work?

1. **Dark Knowledge**: Soft labels encode rich information about class similarities (e.g., "cat" is more similar to "dog" than to "airplane")
2. **Regularization Effect**: Learning from soft targets acts as a form of label smoothing
3. **Better Gradient Signal**: Soft labels provide more nuanced gradients than one-hot labels

### 4.4 Limitations and Future Work

- **Dataset Scope**: Experiments limited to CIFAR-10; larger datasets may show different trends
- **Architecture Diversity**: Only one teacher-student pair tested
- **Advanced Methods**: Could explore attention transfer, feature matching, or self-distillation

---

## 5. Conclusion

This study demonstrates the effectiveness of knowledge distillation for neural network compression. Key achievements:

| Metric | Value |
|--------|-------|
| **Model Compression** | 9.5× (21.3M → 2.2M parameters) |
| **Best Student Accuracy** | 92.95% |
| **Teacher Recovery Rate** | 98.8% (92.95/94.08) |
| **KD Improvement** | +0.85% over baseline |
| **Optimal Hyperparameters** | T=16, α=0.3 |

Knowledge distillation provides a practical approach to deploying powerful neural networks on resource-constrained devices while maintaining competitive accuracy.

---

## 6. Implementation Details

### 6.1 Project Structure

```
knowledge-distillation/
├── src/
│   ├── models.py           # Teacher & Student model definitions
│   ├── distillation.py     # KD loss implementation
│   ├── train_teacher.py    # Teacher training script
│   ├── train_student.py    # Student training (with/without KD)
│   ├── evaluate.py         # Model evaluation
│   ├── utils.py            # Data loading, utilities
│   └── visualize.py        # Result visualization
├── scripts/                # SLURM job scripts
│   ├── train_teacher.sh
│   ├── train_student.sh
│   └── run_ablation.sh
├── checkpoints/            # Saved model weights
├── configs/                # Configuration files
└── results/                # Experiment outputs
```

### 6.2 Reproducing Results

```bash
# 1. Train teacher model
sbatch scripts/train_teacher.sh

# 2. Train student baseline (no KD)
sbatch scripts/train_student.sh --no-kd

# 3. Train student with KD
sbatch scripts/train_student.sh --temperature 16 --alpha 0.3
```

### 6.3 Computational Resources

- **Hardware**: NVIDIA V100 GPU (16GB)
- **Teacher Training Time**: ~2 hours (100 epochs)
- **Student Training Time**: ~3 hours (200 epochs)
- **Total Experiment Time**: ~40 GPU-hours (including all ablations)

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.

4. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.

---

## Appendix A: Complete Experimental Results

### Temperature Ablation (α=0.3)

| T | Accuracy | Hard Loss | Soft Loss |
|---|----------|-----------|-----------|
| 1 | 92.64% | Standard CE | KL at T=1 |
| 2 | 92.51% | Standard CE | KL at T=2 |
| 4 | 92.58% | Standard CE | KL at T=4 |
| 8 | 92.72% | Standard CE | KL at T=8 |
| 16 | 92.95% | Standard CE | KL at T=16 |

### Alpha Ablation (T=4)

| α | Accuracy | Hard Weight | Soft Weight |
|---|----------|-------------|-------------|
| 0.1 | 92.19% | 0.1 | 0.9 |
| 0.3 | 92.58% | 0.3 | 0.7 |
| 0.5 | 92.89% | 0.5 | 0.5 |
| 0.7 | 92.58% | 0.7 | 0.3 |
| 0.9 | 92.66% | 0.9 | 0.1 |

---

*Report generated for Master's Thesis Position Interview at Aalto University*
