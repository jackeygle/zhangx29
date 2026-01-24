# 🎯 Knowledge Distillation 面试准备指南

**Aalto University 硕士论文职位面试**

---

## 📋 项目概述（30秒电梯演讲）

> "我实现了一个完整的知识蒸馏项目，将 21M 参数的 ResNet-34 压缩到 2.4M 的 MobileNetV2，实现约 9 倍压缩。在 CIFAR-100 上，知识蒸馏将学生模型的准确率从 69.26% 提升到 73.07%，提升了 3.81 个百分点。这个项目展示了知识蒸馏在模型压缩中的实际价值。"

---

## 📊 关键数据（必须记住）

### CIFAR-10 结果
| 模型 | 准确率 | 参数量 |
|------|--------|--------|
| Teacher (ResNet-34) | 94.08% | 21.3M |
| Student Baseline | 92.10% | 2.2M |
| Student + KD | 92.95% | 2.2M |
| **KD 提升** | **+0.85%** | - |

### CIFAR-100 结果
| 模型 | 准确率 | 参数量 |
|------|--------|--------|
| Teacher (ResNet-34) | 79.16% | 21.3M |
| Student Baseline | 69.26% | 2.4M |
| Student + KD | 73.07% | 2.4M |
| **KD 提升** | **+3.81%** | - |

### 核心发现
- 模型压缩: **~9x**
- CIFAR-100 上 KD 提升是 CIFAR-10 的 **4.5 倍**
- 最优超参数: T=4~16, α=0.3~0.5

---

## 🧠 技术问题与答案

### Q1: 知识蒸馏的核心原理是什么？

**答案**:
知识蒸馏让小模型（Student）从大模型（Teacher）的软标签中学习。软标签（soft labels）包含类别之间的相似性信息，这被称为"暗知识"（dark knowledge）。

例如，对于一张猫的图片：
- 硬标签: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] — 只有"猫"是1
- 软标签: [0.01, 0.02, 0.05, 0.60, 0.08, **0.15**, 0.03, 0.02, 0.02, 0.02]

软标签告诉我们"这张图虽然是猫，但看起来有点像狗"。

---

### Q2: 损失函数是什么？

**答案**:
$$L_{total} = \alpha \cdot L_{CE}(student, labels) + (1-\alpha) \cdot T^2 \cdot L_{KL}(soft_{student}, soft_{teacher})$$

- **第一项**: 交叉熵损失，让 Student 学习正确答案
- **第二项**: KL 散度，让 Student 模仿 Teacher 的概率分布
- **α**: 平衡两个损失的权重 (默认 0.3)
- **T**: 温度，控制概率分布的软硬程度

---

### Q3: 为什么要乘以 T²？

**答案**:
当我们对 logits 除以 T 时，softmax 的梯度会缩小约 1/T²。为了保持梯度量级与温度无关，需要乘以 T² 来补偿。这确保不同温度下的训练稳定性。

---

### Q4: Temperature 的作用是什么？

**答案**:
Temperature 控制概率分布的"软硬程度"：
- T=1: 分布尖锐，接近 one-hot
- T=4~16: 分布平滑，暴露类别间的相似性
- T→∞: 接近均匀分布

较高的温度能更好地揭示类别之间的关系，这正是"暗知识"的来源。

---

### Q5: 为什么用 Forward KL 而不是 Reverse KL？

**答案**:
- **Forward KL** $D_{KL}(P_{teacher} \| Q_{student})$: 让 Student 覆盖 Teacher 所有认为可能的类别
- **Reverse KL**: 让 Student 只抓住一个模式

我们希望 Student 学习 Teacher 的所有知识，所以用 Forward KL。

---

### Q6: 为什么 CIFAR-100 上 KD 效果更好？

**答案**:
CIFAR-100 有 100 个细粒度类别（如"橡树" vs "枫树"），类别之间有更丰富的相似性关系。Teacher 的软标签提供了更多有价值的信息。

在简单任务（CIFAR-10）上，Student 本身就能学得很好，KD 的额外信息价值有限。

---

### Q7: 你选择 MobileNetV2 作为 Student 的原因？

**答案**:
MobileNetV2 是为移动设备设计的轻量级网络，使用深度可分离卷积（Depthwise Separable Convolution）大幅减少计算量。它只有 2.4M 参数，是 ResNet-34 的约 1/9，非常适合作为压缩目标。

---

### Q8: 还有哪些其他的知识蒸馏方法？

**答案**:
1. **Feature-based (FitNets)**: 让 Student 模仿 Teacher 的中间层特征
2. **Attention Transfer**: 匹配注意力图
3. **Self-distillation**: 模型蒸馏自己
4. **Contrastive Distillation**: 使用对比学习

---

## 💡 准备解释的代码片段

### 蒸馏损失函数
```python
# distillation.py
def forward(self, student_logits, teacher_logits, labels):
    # 硬标签损失
    hard_loss = self.ce_loss(student_logits, labels)
    
    # 软标签损失
    student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
    soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
    
    # 组合损失
    total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
    return total_loss
```

### 蒸馏训练循环
```python
# train_student.py
for images, labels in train_loader:
    # Teacher 不计算梯度（冻结）
    with torch.no_grad():
        teacher_logits = teacher(images)
    
    # Student 前向传播
    student_logits = student(images)
    
    # 蒸馏损失
    loss = criterion(student_logits, teacher_logits, labels)
    
    # 只更新 Student
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🎤 可能的开放性问题

### "这个项目对你的研究兴趣有什么启发？"

> "这个项目让我对高效深度学习产生了浓厚兴趣。我发现模型压缩不仅是工程问题，更涉及深刻的学习理论问题——比如知识如何在神经网络间传递。我希望在硕士期间深入研究模型压缩、神经架构搜索或高效推理等方向。"

### "如果有更多时间，你会做什么改进？"

> "我会：
> 1. 在更大数据集（如 ImageNet）上验证
> 2. 尝试 Feature-based 蒸馏方法
> 3. 结合量化/剪枝进一步压缩
> 4. 测量实际推理延迟，而不仅仅是参数量"

---

## ✅ 面试前检查清单

- [ ] 能流利说出项目概述（30秒版本）
- [ ] 记住关键数据（准确率、参数量、提升幅度）
- [ ] 理解损失函数每一项的含义
- [ ] 能解释 Temperature 和 Alpha 的作用
- [ ] 能解释为什么 CIFAR-100 效果更好
- [ ] 准备好代码演示（如需要）
- [ ] 准备 2-3 个你想问面试官的问题

---

## 🗣️ 提问面试官的建议问题

1. "这个硕士项目的主要研究方向是什么？"
2. "团队目前在模型压缩/高效学习方面有哪些正在进行的工作？"
3. "作为硕士生，我会有哪些合作机会？"

---

**祝面试顺利！Good luck! 🍀**
