#!/bin/bash
#SBATCH --job-name=kd_student
#SBATCH --output=logs/student_%j.out
#SBATCH --error=logs/student_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-16g

# Parse arguments
USE_KD=true
TEMPERATURE=4.0
ALPHA=0.3
WIDTH=1.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-kd)
            USE_KD=false
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints data

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Build command
if [ "$USE_KD" = true ]; then
    echo "Training student WITH Knowledge Distillation"
    echo "Temperature: $TEMPERATURE, Alpha: $ALPHA, Width: $WIDTH"
    
    python src/train_student.py \
        --epochs 200 \
        --lr 0.1 \
        --batch-size 128 \
        --scheduler cosine \
        --teacher-ckpt ./checkpoints/teacher_best.pth \
        --temperature $TEMPERATURE \
        --alpha $ALPHA \
        --width-mult $WIDTH \
        --data-dir ./data \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs
else
    echo "Training student WITHOUT Knowledge Distillation (baseline)"
    
    python src/train_student.py \
        --epochs 200 \
        --lr 0.1 \
        --batch-size 128 \
        --scheduler cosine \
        --no-kd \
        --width-mult $WIDTH \
        --data-dir ./data \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs
fi

echo "Student training complete!"
