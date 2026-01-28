#!/bin/bash
#SBATCH --job-name=kd_teacher
#SBATCH --output=logs/teacher_%j.out
#SBATCH --error=logs/teacher_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Config
DATASET=${DATASET:-cifar10}

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints data

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Train teacher model
python src/train_teacher.py \
    --dataset "$DATASET" \
    --epochs 100 \
    --lr 0.1 \
    --batch-size 128 \
    --scheduler cosine \
    --pretrained \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs

echo "Teacher training complete!"
