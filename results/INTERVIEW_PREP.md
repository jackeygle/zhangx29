# Knowledge Distillation Interview Notes (English)

Master's Thesis Interview Preparation

---

## Elevator Pitch (30 seconds)

"I implemented a complete knowledge distillation pipeline that compresses a 21M-parameter ResNet-34 teacher into a 2.5M MobileNetV2 student (~9x compression). I compared six distillation methods across CIFAR-10, CIFAR-100, and Tiny-ImageNet. On Tiny-ImageNet, vanilla KD improves the student from 55.30% to 60.55% (+5.25%), showing that distillation gains grow with task difficulty."

---

## Key Numbers (must memorize)

### Quick Comparison Table (all datasets)

| Dataset | Teacher | Baseline | Best Method | Best Acc | Gain |
|--------|---------|----------|-------------|----------|------|
| CIFAR-10 | 94.08% | 92.10% | KD | 92.95% | +0.85% |
| CIFAR-100 | 79.16% | 69.26% | FitNets | 73.80% | +4.54% |
| Tiny-ImageNet | 71.42% | 55.30% | KD | 60.55% | +5.25% |

Core insight: the harder the task, the larger the distillation gain.

### Tiny-ImageNet (200 classes, 64x64)

| Model | Accuracy | Params | Gain |
|------|----------|--------|------|
| Teacher (ResNet-34) | 71.42% | 21.4M | - |
| Student Baseline | 55.30% | 2.5M | Baseline |
| Student + KD | 60.55% | 2.5M | +5.25% |
| Student + Contrastive | 59.05% | 2.5M | +3.75% |
| Student + FitNets | 57.05% | 2.5M | +1.75% |
| Student + Attention | 53.47% | 2.5M | -1.83% |
| Student + Self | 52.88% | 2.5M | -2.42% |

Summary points:
- Compression ratio: ~9x
- CIFAR-100 KD gain is 4.5x larger than CIFAR-10
- Tiny-ImageNet KD gain is the largest (+5.25%)
- Best hyperparams in this setup: T=4~16, alpha=0.3~0.5

---

## Method Comparison (Tiny-ImageNet ranking)

| Rank | Method | Accuracy | Gain vs Baseline |
|------|--------|----------|------------------|
| 1 | KD (logits) | 60.55% | +5.25% |
| 2 | Contrastive | 59.05% | +3.75% |
| 3 | FitNets | 57.05% | +1.75% |
| 4 | Baseline | 55.30% | - |
| 5 | Attention | 53.47% | -1.83% |
| 6 | Self | 52.88% | -2.42% |

Takeaway: KD and Contrastive are most stable for harder tasks.

---

## Common Interview Questions (with answers)

### Q1: What is the core idea of knowledge distillation?
Answer: Knowledge distillation trains a small student to match a large teacher’s output distribution, not just the hard labels. The teacher’s soft probabilities encode class similarity (dark knowledge), e.g., an image of “cat” may assign non‑trivial probability to “dog.” This gives the student a richer training signal than one‑hot labels and acts as a regularizer, improving generalization.

### Q2: What is the loss function?
Answer:
L = alpha * L_CE + (1 - alpha) * T^2 * L_KL
- L_CE: cross‑entropy with ground‑truth labels (forces correctness)
- L_KL: KL divergence between teacher and student soft distributions (transfers dark knowledge)
- T: temperature to smooth probabilities (reveals inter‑class similarity)
- alpha: balance between hard and soft supervision
In practice I use T in [4, 16] and alpha in [0.3, 0.5] as a stable range.

### Q3: Why multiply by T^2?
Answer: Dividing logits by T makes softmax outputs smoother but also shrinks gradients (approximately by 1/T^2). Without compensation, large T would make the distillation loss too weak. Multiplying by T^2 restores the gradient scale so different temperatures remain comparable and training stays stable.

### Q4: What does temperature do?
Answer: Temperature controls the “softness” of the probability distribution. With T=1, the teacher outputs are sharp (close to one‑hot). With larger T, probabilities spread across similar classes, exposing fine‑grained relationships that guide the student (e.g., “oak” vs “maple” in Tiny‑ImageNet).

### Q5: Why forward KL instead of reverse KL?
Answer: Forward KL, D_KL(P_teacher || Q_student), penalizes the student if it assigns near‑zero probability to classes the teacher considers plausible. This encourages coverage of the teacher’s full distribution. Reverse KL is mode‑seeking and tends to ignore secondary classes, which is undesirable for distillation because it discards dark knowledge.

### Q6: Why does KD work better on harder datasets?
Answer: Harder datasets have more confusing classes and richer inter‑class structure, so the teacher’s soft labels contain more useful information. The student also has more room to improve. That’s why the gain grows from CIFAR‑10 (+0.85%) to CIFAR‑100 (+3.81%) to Tiny‑ImageNet (+5.25%).

### Q7: Why MobileNetV2 as student?
Answer: MobileNetV2 is designed for efficient inference with depthwise separable convolutions, giving a strong accuracy‑to‑compute tradeoff. It has ~2.5M parameters vs 21M in ResNet‑34, providing ~9× compression while still being competitive for distillation.

### Q8: Why did Attention Transfer and Self-distillation underperform on Tiny-ImageNet?
Answer:
- Attention transfer relies on spatial maps; at 64x64 resolution and 200 fine‑grained classes, attention can be too coarse and noisy. It may emphasize regions that are not discriminative enough.
- Self‑distillation uses an EMA teacher derived from the student. On hard datasets, the student is not strong enough early on, so the EMA teacher may provide weak supervision compared to a strong external teacher.

### Q9: Which methods did you implement?
Answer: I implemented six methods: baseline (no KD), KD (logits), FitNets (feature MSE with a 1x1 projection), Attention Transfer (attention map alignment), Self‑distillation (EMA teacher), and Contrastive Distillation (InfoNCE on pooled features). I also built scripts to run all methods and generate comparison plots automatically.

### Q10: What would you do next?
Answer: I would scale to ImageNet‑1K, experiment with hybrid losses (KD + feature‑based), and explore pruning/quantization for further compression. I would also measure actual inference latency/energy on target hardware, not just parameter count.

### Q11: How did you choose teacher and student architectures?
Answer: I chose ResNet‑34 as a strong but manageable teacher and MobileNetV2 as a lightweight student designed for mobile/edge deployment. This pairing gives a clear compression gap (~9×) while keeping the student competitive enough to benefit from distillation.

### Q12: Why not train the student from scratch only?
Answer: A baseline trained only with hard labels misses the teacher’s dark knowledge. Distillation adds soft supervision that encodes class similarities and acts as regularization, which consistently improves accuracy on harder datasets.

### Q13: How did you select the distillation layer for feature methods?
Answer: I use mid‑to‑late feature layers where semantic information is richer. In code, I specify layers explicitly (e.g., ResNet layer4 and MobileNetV2 features.18) and align spatial sizes before computing loss.

### Q14: What happens if teacher and student feature maps have different sizes?
Answer: I align spatial dimensions using adaptive average pooling to the smaller size. This makes feature losses well‑defined without introducing extra trainable parameters.

### Q15: How did you ensure reproducibility?
Answer: I set random seeds, fixed data preprocessing, logged metrics to TensorBoard, and saved checkpoints. Scripts are deterministic and can be run on SLURM with controlled configs.

### Q16: What metrics did you track besides accuracy?
Answer: I tracked training/validation loss, best accuracy, parameter count, and (when needed) inference speed using a separate evaluation script.

### Q17: How do you interpret the KD gains across datasets?
Answer: Gains increase with task difficulty (CIFAR‑10 < CIFAR‑100 < Tiny‑ImageNet). Harder tasks have richer class relationships, so soft labels are more informative.

### Q18: Why do some methods underperform on Tiny‑ImageNet?
Answer: Attention Transfer can be too coarse at 64×64 resolution, and EMA self‑distillation relies on a strong student, which is less reliable on difficult tasks. KD and Contrastive use a strong teacher signal, so they remain effective.

### Q19: How would you combine methods?
Answer: I would use a weighted sum of KD loss and feature loss (e.g., KD + FitNets) or KD + Contrastive. A small ablation would determine the best mixing weights.

### Q20: How would you deploy the student model?
Answer: I would export the student to ONNX/TensorRT, measure latency on target hardware, and consider quantization for further speedup while monitoring accuracy drops.

### Q21: What are the limitations of your current study?
Answer: Only one teacher‑student pair was tested, datasets were limited to CIFAR/Tiny‑ImageNet, and I did not measure real hardware latency. These are natural extensions.

### Q22: What practical insights does this project offer?
Answer: Distillation is most beneficial for harder tasks, and simple logits‑based KD can outperform more complex methods. This suggests a strong baseline before investing in sophisticated techniques.

---

## Implementation Highlights (talking points)

- Teacher: ResNet-34 adapted for CIFAR/Tiny-ImageNet
- Student: MobileNetV2, width multiplier = 1.0
- Distillation methods: logits, feature, attention, self, contrastive
- SLURM scripts with reproducible checkpoints
- TensorBoard logging for training curves

---

## Questions to Ask Interviewers

1. What is the project's main research focus in model compression?
2. How much emphasis is placed on theory vs. engineering?
3. What datasets or benchmarks do you expect students to use?
4. What computing resources are available for large-scale training?

---

Good luck!

