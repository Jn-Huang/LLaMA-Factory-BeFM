#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen3_4b_sft_2gpu
#SBATCH --output=/home/huangjin/logs/qwen3_4b_2gpu_%x-%A-%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

# Project directory - UPDATE THIS PATH
PROJECT_DIR=/home/huangjin/LLaMA-Factory-BeFM

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Display GPU information
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Change to project directory
cd $PROJECT_DIR || exit 1

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate conda environment - UPDATE THIS if you have a different environment name
conda activate llamafactory  # Change to your environment name

# Verify environment
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "Number of GPUs:"
python -c "import torch; print(torch.cuda.device_count())"
echo "=========================================="

# Run training with YAML config
# FORCE_TORCHRUN=1 is required for multi-GPU training
echo "Starting LoRA training with 2 GPUs..."
FORCE_TORCHRUN=1 llamafactory-cli train slurm_scripts/qwen3_4b_lora_sft.yaml

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Job finished at: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi
