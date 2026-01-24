# Knowledge Distillation on CIFAR-100

**A Technical Report on Model Compression for Fine-Grained Classification**

**Author:** Xinle Zhang  
**Affiliation:** Aalto University  
**Date:** January 2026

---

## Abstract

This report presents experiments on Knowledge Distillation (KD) using the CIFAR-100 dataset, a more challenging benchmark with 100 fine-grained classes. Our results demonstrate that KD provides significantly larger improvements on harder tasks: the student model achieves **73.07% accuracy** with KD versus **69.26% baseline**, representing a **+3.81% improvement**—4.5× greater than the improvement observed on CIFAR-10.

---

## 1. Motivation

Our initial CIFAR-10 experiments showed only +0.85% improvement from KD. This raised the question: *Is Knowledge Distillation really worth the effort?*

To answer this, we extended our experiments to CIFAR-100, which is significantly more challenging:

| Dataset | Classes | Images/Class | Difficulty |
|---------|---------|--------------|------------|
| CIFAR-10 | 10 | 6,000 | Easy |
| CIFAR-100 | 100 | 600 | **Hard** |

---

## 2. Experimental Results

### 2.1 Main Results

| Model | Accuracy | Parameters | Compression |
|-------|----------|------------|-------------|
| Teacher (ResNet-34) | **79.16%** | 21.3M | 1× |
| Student Baseline | 69.26% | 2.4M | ~9× |
| Student + KD (T=4, α=0.3) | **73.07%** | 2.4M | ~9× |

### 2.2 Knowledge Distillation Improvement

```
KD Improvement = 73.07% - 69.26% = +3.81%
Error Reduction = (30.74 - 26.93) / 30.74 = 12.4%
```

### 2.3 Comparison: CIFAR-10 vs CIFAR-100

| Metric | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| Teacher Accuracy | 94.08% | 79.16% |
| Student Baseline | 92.10% | 69.26% |
| Student + KD | 92.95% | 73.07% |
| **KD Improvement** | +0.85% | **+3.81%** |
| **Improvement Ratio** | 1× | **4.5×** |

---

## 3. Key Findings

### 3.1 KD Benefits Scale with Task Difficulty

```
CIFAR-10 (Easy):    +0.85% improvement
CIFAR-100 (Hard):   +3.81% improvement  ← 4.5× larger!
```

**Why?** On harder tasks:
- The teacher has more "dark knowledge" to transfer
- The student has more room for improvement
- Soft labels provide more valuable inter-class relationship information

### 3.2 Model Compression Analysis

| Metric | Value |
|--------|-------|
| Parameter Reduction | 21.3M → 2.4M (8.9×) |
| Accuracy Loss (vs Teacher) | 79.16% → 73.07% (-6.09%) |
| Accuracy Loss (Baseline) | 79.16% → 69.26% (-9.90%) |
| **Performance Recovered by KD** | 3.81% / 9.90% = **38.5%** |

### 3.3 Training Dynamics

The student with KD showed steady improvement throughout training:
- Epoch 100: ~69% (same as baseline)
- Epoch 150: ~72%
- Epoch 200: **73.07%** (final)

---

## 4. Training Configuration

| Parameter | Teacher | Student |
|-----------|---------|---------|
| Dataset | CIFAR-100 | CIFAR-100 |
| Epochs | 100 | 200 |
| Batch Size | 128 | 128 |
| Learning Rate | 0.1 (cosine) | 0.1 (cosine) |
| Temperature (T) | - | 4.0 |
| Alpha (α) | - | 0.3 |

---

## 5. Discussion

### 5.1 Why KD Works Better on Harder Tasks

**CIFAR-10 (10 classes):**
- Categories are distinct (airplane vs cat)
- Simple classification, limited inter-class information
- Student can learn most patterns independently

**CIFAR-100 (100 classes):**
- Many similar categories (e.g., "maple tree" vs "oak tree")
- Rich inter-class relationships
- Student benefits greatly from teacher's soft predictions

### 5.2 Practical Implications

For real-world deployment:
- If your task is simple → KD provides modest gains
- If your task is hard → **KD is essential!**

---

## 6. Conclusion

| CIFAR-100 Results | Value |
|-------------------|-------|
| **Model Compression** | 8.9× |
| **Baseline Accuracy** | 69.26% |
| **KD Accuracy** | 73.07% |
| **KD Improvement** | +3.81% |

**Key Takeaway:** Knowledge Distillation provides significantly larger benefits on challenging tasks. The +3.81% improvement on CIFAR-100 (vs +0.85% on CIFAR-10) validates KD as an essential technique for deploying compact models on complex classification problems.

---

## 7. Commands to Reproduce

```bash
# Train teacher on CIFAR-100
sbatch scripts/train_teacher_cifar100.sh

# Train student baseline
sbatch scripts/train_student_cifar100.sh --no-kd

# Train student with KD
sbatch scripts/train_student_cifar100.sh
```

---

*This report demonstrates the effectiveness of Knowledge Distillation on fine-grained classification tasks.*
