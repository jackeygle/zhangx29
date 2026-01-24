#!/bin/bash
#SBATCH --job-name=kd_methods
#SBATCH --output=logs/distill_methods_%j.out
#SBATCH --error=logs/distill_methods_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-16g

# Run multiple distillation methods on Triton GPU.

module load mamba
source activate kd

PROJECT_DIR="/scratch/work/zhangx29/knowledge-distillation"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results/figures

EPOCHS=${EPOCHS:-200}
DATASET=${DATASET:-cifar10}
TEACHER_CKPT=${TEACHER_CKPT:-./checkpoints/teacher_best.pth}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-}

COMMON_ARGS="--epochs $EPOCHS --dataset $DATASET --teacher-ckpt $TEACHER_CKPT \
  --checkpoint-dir ./checkpoints --log-dir ./logs"

if [ -n "$MAX_TRAIN_BATCHES" ]; then
  COMMON_ARGS="$COMMON_ARGS --max-train-batches $MAX_TRAIN_BATCHES"
fi
if [ -n "$MAX_VAL_BATCHES" ]; then
  COMMON_ARGS="$COMMON_ARGS --max-val-batches $MAX_VAL_BATCHES"
fi

echo "Running baseline..."
python src/train_student.py $COMMON_ARGS --no-kd

echo "Running KD (logits)..."
python src/train_student.py $COMMON_ARGS --distill-method kd --temperature 4 --alpha 0.3

echo "Running FitNets (feature-based)..."
python src/train_student.py $COMMON_ARGS --distill-method fitnets --alpha 0.3

echo "Running Attention Transfer..."
python src/train_student.py $COMMON_ARGS --distill-method attention --alpha 0.3

echo "Running Self-Distillation (EMA)..."
python src/train_student.py $COMMON_ARGS --distill-method self --alpha 0.3

echo "Running Contrastive Distillation..."
python src/train_student.py $COMMON_ARGS --distill-method contrastive --alpha 0.3 --contrastive-temp 0.1

echo "Generating comparison plot..."
python src/compare_methods.py --dataset $DATASET

echo "Done."

