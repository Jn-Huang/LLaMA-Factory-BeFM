# SLURM Training Scripts for LLaMA-Factory

Training scripts for fine-tuning Qwen3-4B-Instruct-2507 on HPC clusters.

## Files

| File | Description |
|------|-------------|
| `train_qwen3_4b.sh` | LoRA fine-tuning (1 GPU) |
| `train_qwen3_4b_full.sh` | Full parameter fine-tuning (4 GPUs) |
| `qwen3_4b_lora_sft.yaml` | LoRA training config |
| `qwen3_4b_full_sft.yaml` | Full training config |
| `qwen3_4b_inference.yaml` | LoRA inference config |
| `qwen3_4b_full_inference.yaml` | Full model inference config |
| `qwen3_4b_merge.yaml` | LoRA merge config |

## Quick Comparison

| Feature | LoRA | Full Parameter |
|---------|------|----------------|
| GPUs | 1 | 4+ |
| Memory | ~12-16GB | ~80-120GB (distributed) |
| Parameters | ~0.1-1% | 100% |
| Speed | Fast | Slower |
| Performance | Good | Best |

## Setup

### 1. Update SLURM Scripts

Edit the `.sh` files and update:
- `PROJECT_DIR` - your project path
- `conda activate` - your environment name
- `--output` - your log directory

### 2. Update Training Config

Edit the `.yaml` files:
```yaml
dataset: your_dataset_name  # Change from identity,alpaca_en_demo
max_samples: 1000           # Remove for full dataset
```

### 3. Submit Jobs

**LoRA Fine-Tuning:**
```bash
sbatch slurm_scripts/train_qwen3_4b.sh
```

**Full Parameter Fine-Tuning:**
```bash
sbatch slurm_scripts/train_qwen3_4b_full.sh
```

### 4. Monitor

```bash
squeue -u $USER                              # Check status
tail -f /home/huangjin/logs/qwen3_4b_*.log   # View logs
scancel <job_id>                             # Cancel job
```

## Inference

**LoRA model:**
```bash
llamafactory-cli chat slurm_scripts/qwen3_4b_inference.yaml
llamafactory-cli export slurm_scripts/qwen3_4b_merge.yaml  # Merge adapter
```

**Full model:**
```bash
llamafactory-cli chat slurm_scripts/qwen3_4b_full_inference.yaml
```

## Troubleshooting

**Out of Memory:**
- Reduce `per_device_train_batch_size`
- Add `quantization_bit: 4` for QLoRA
- Reduce `cutoff_len`

**Dataset Not Found:**
- Check dataset name in `data/dataset_info.json`
- See `data/README.md` for format details

## Resources

- [LLaMA-Factory Docs](https://github.com/hiyouga/LLaMA-Factory)
- [Examples](../examples/)
- [Dataset Guide](../data/README.md)
