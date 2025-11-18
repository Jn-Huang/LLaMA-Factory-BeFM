#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=llama3_8b_lora_sft
#SBATCH --output=/home/huangjin/logs/llama3_8b_lora_%x-%A-%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

# Project directory - UPDATE THIS PATH
PROJECT_DIR=/home/huangjin/LLaMA-Factory-BeFM

echo "=========================================="
echo "LLaMA-3-8B-Instruct LoRA Fine-tuning"
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

# Step 1: Run LoRA fine-tuning
echo "Step 1: Starting LoRA fine-tuning..."
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

if [ $? -ne 0 ]; then
    echo "=========================================="
    echo "Training failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="

# Step 2: Skip chat (interactive command, not suitable for batch jobs)
# To run chat interactively after training, use:
# llamafactory-cli chat examples/inference/llama3_lora_sft.yaml

# Step 3: Merge LoRA adapters with base model
echo "Step 3: Merging LoRA adapters..."
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

if [ $? -ne 0 ]; then
    echo "=========================================="
    echo "Merge failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "All steps completed successfully!"
echo "Training output: saves/llama3-8b/lora/sft"
echo "Merged model: output/llama3_lora_sft"
echo "Job finished at: $(date)"
echo "=========================================="
