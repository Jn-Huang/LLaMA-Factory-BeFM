# SLURM Training Scripts for LLaMA-Factory

This directory contains SLURM scripts for training models using LLaMA-Factory on HPC clusters.

## Files

### Training Scripts
- `train_qwen3_4b.sh` - SLURM script for LoRA fine-tuning (1 GPU, ~12-16GB memory)
- `train_qwen3_4b_full.sh` - SLURM script for full parameter fine-tuning (4 GPUs, DeepSpeed ZeRO-3)

### Training Configurations
- `qwen3_4b_lora_sft.yaml` - LoRA fine-tuning configuration
- `qwen3_4b_full_sft.yaml` - Full parameter fine-tuning configuration

### Inference & Export
- `qwen3_4b_inference.yaml` - Inference configuration (chat/API)
- `qwen3_4b_merge.yaml` - LoRA adapter merge configuration

## Setup Instructions

### 1. Update the SLURM Script

Edit `train_qwen3_4b.sh` and update the following variables:

```bash
# Line 12: Update with your actual project directory path
PROJECT_DIR=/home/huangjin/LLaMA-Factory-BeFM  # Change this!

# Line 10: Update with your log directory path
#SBATCH --output=/home/huangjin/logs/qwen3_4b_%x-%A-%j.log

# Line 27: Update with your conda environment name
conda activate llamafactory  # Change to your environment name
```

### 2. Update the Training Configuration

Edit `qwen3_4b_lora_sft.yaml` to customize your training:

**Dataset:**
```yaml
dataset: identity,alpaca_en_demo  # Change to your dataset name
max_samples: 1000  # Remove this line to use full dataset
```

**Training Parameters:**
```yaml
per_device_train_batch_size: 2  # Adjust based on GPU memory
gradient_accumulation_steps: 4   # Effective batch size = 2 * 4 = 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
```

**Output Directory:**
```yaml
output_dir: saves/qwen3-4b/lora/sft  # Change if needed
```

**Experiment Tracking (optional):**
```yaml
report_to: wandb  # Options: none, wandb, tensorboard, swanlab, mlflow
```

### 3. Prepare Your Dataset

#### Option A: Use Built-in Datasets
LLaMA-Factory includes many datasets in `data/dataset_info.json`. See available datasets:
```bash
cat data/dataset_info.json
```

Common datasets:
- `alpaca_en_demo` - English instruction dataset
- `alpaca_zh_demo` - Chinese instruction dataset
- `identity` - Identity/system prompt dataset

#### Option B: Add Your Own Dataset

1. Create your dataset in JSON/JSONL format (see `data/README.md` for format details)
2. Add dataset info to `data/dataset_info.json`:
```json
"your_dataset_name": {
  "file_name": "your_data.json",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```
3. Update `qwen3_4b_lora_sft.yaml`:
```yaml
dataset: your_dataset_name
```

### 4. Submit the Job

```bash
# Navigate to the project directory
cd /home/huangjin/LLaMA-Factory-BeFM

# Submit the job
sbatch slurm_scripts/train_qwen3_4b.sh
```

### 5. Monitor the Job

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f /home/huangjin/logs/qwen3_4b_*.log

# Cancel a job if needed
scancel <job_id>
```

## Training Configurations

### Quick Comparison

| Feature | LoRA Fine-Tuning | Full Parameter Fine-Tuning |
|---------|------------------|----------------------------|
| **GPUs Required** | 1 GPU | 4+ GPUs |
| **GPU Memory** | ~12-16GB | ~80-120GB (distributed) |
| **Training Speed** | Fast | Slower |
| **Parameters Updated** | ~0.1-1% (adapters) | 100% (all weights) |
| **Checkpoint Size** | ~10-50MB | ~8GB |
| **Performance** | Good | Best |
| **Use Case** | General fine-tuning | Domain adaptation, max performance |
| **Recommended For** | Most users | Advanced users, specific domains |

### LoRA Fine-Tuning (Recommended)
- **Memory**: ~12-16GB GPU memory per device
- **GPUs**: 1 GPU
- **Speed**: Fast, efficient
- **Use case**: Most common scenario, parameter-efficient
- **Script**: `train_qwen3_4b.sh`
- **Config**: `qwen3_4b_lora_sft.yaml`

**Submit:**
```bash
sbatch slurm_scripts/train_qwen3_4b.sh
```

**Pros:**
- Lower memory requirements
- Faster training
- Easy to merge or switch adapters
- Good performance for most tasks

**Cons:**
- Slightly lower performance than full fine-tuning
- Limited to adapter parameters (~0.1-1% of model)

### Full Parameter Fine-Tuning
- **Memory**: ~80-120GB total (distributed across GPUs)
- **GPUs**: 4 GPUs minimum (uses DeepSpeed ZeRO-3)
- **Speed**: Slower than LoRA
- **Use case**: Maximum performance, domain adaptation
- **Script**: `train_qwen3_4b_full.sh`
- **Config**: `qwen3_4b_full_sft.yaml`

**Submit:**
```bash
sbatch slurm_scripts/train_qwen3_4b_full.sh
```

**Pros:**
- Maximum performance
- All parameters updated
- Better for significant domain shift

**Cons:**
- Higher memory requirements
- Longer training time
- Requires multiple GPUs
- Larger checkpoint size

**Important Notes:**
- Uses DeepSpeed ZeRO-3 for memory efficiency
- Requires `FORCE_TORCHRUN=1` for distributed training
- Lower learning rate (1e-5 vs 5e-5 for LoRA)
- Ensure DeepSpeed is installed: `pip install deepspeed`

### Multi-GPU Training

For multiple GPUs, update the SLURM script:
```bash
#SBATCH --gres=gpu:4  # Use 4 GPUs
```

And run with torchrun:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train slurm_scripts/qwen3_4b_lora_sft.yaml
```

### Advanced Options

#### QLoRA (Quantized LoRA)
Add to YAML config:
```yaml
quantization_bit: 4  # 4-bit quantization
quantization_type: otfq  # Options: otfq, bnb, gptq, awq
```

#### DeepSpeed ZeRO-3
```bash
#SBATCH --gres=gpu:4
```
In YAML:
```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

## Output Files

### LoRA Fine-Tuning Output
After LoRA training, you'll find:

```
saves/qwen3-4b/lora/sft/
├── adapter_config.json      # LoRA adapter configuration
├── adapter_model.safetensors # Trained LoRA weights (~10-50MB)
├── trainer_state.json        # Training state
├── training_args.bin         # Training arguments
├── training_loss.png         # Loss plot (if plot_loss: true)
└── all_results.json         # Final metrics
```

### Full Fine-Tuning Output
After full parameter training, you'll find:

```
saves/qwen3-4b/full/sft/
├── model-00001-of-00002.safetensors  # Model weights (sharded)
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json     # Shard index
├── config.json                       # Model configuration
├── generation_config.json
├── trainer_state.json                # Training state
├── training_args.bin                 # Training arguments
├── training_loss.png                 # Loss plot (if plot_loss: true)
└── all_results.json                 # Final metrics
```

## Inference After Training

### Using LoRA Fine-Tuned Model

**Option 1: CLI Chat**
```bash
llamafactory-cli chat slurm_scripts/qwen3_4b_inference.yaml
```

**Option 2: Web UI**
```bash
llamafactory-cli webchat slurm_scripts/qwen3_4b_inference.yaml
```

**Option 3: Merge LoRA and Export**
```bash
llamafactory-cli export slurm_scripts/qwen3_4b_merge.yaml
```

### Using Full Fine-Tuned Model

Create an inference config file:
```yaml
# qwen3_4b_full_inference.yaml
model_name_or_path: saves/qwen3-4b/full/sft  # Path to full fine-tuned model
trust_remote_code: true
template: qwen3_nothink
infer_backend: huggingface  # or vllm for faster inference
```

Then run:
```bash
# CLI Chat
llamafactory-cli chat qwen3_4b_full_inference.yaml

# Web UI
llamafactory-cli webchat qwen3_4b_full_inference.yaml

# API Server
llamafactory-cli api qwen3_4b_full_inference.yaml
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use quantization: `quantization_bit: 4`
- Reduce `cutoff_len` (sequence length)

### Slow Training
- Increase `per_device_train_batch_size` if memory allows
- Reduce `preprocessing_num_workers` if CPU-bound
- Use multiple GPUs

### Dataset Not Found
- Check dataset name in `data/dataset_info.json`
- Verify file path is correct
- Ensure dataset format matches specification

## Additional Resources

- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [Examples Directory](../examples/)
- [Dataset Format Guide](../data/README.md)
