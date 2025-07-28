# Linear Weight Merging for Vision-Language Models

This module provides linear weight merging functionality for Qwen2.5-VL and LLaVA-1.5 models. It supports both command-line arguments and YAML configuration files.

## Features

- **Supports multiple model types**: Qwen2.5-VL and LLaVA-1.5
- **Automatic model detection**: Reads model configuration to determine type
- **Linear interpolation**: Merges weights using the formula `λ × model1 + (1-λ) × model2`
- **Flexible configuration**: Command-line arguments or YAML files
- **Complete model saving**: Saves merged model, tokenizer, and processor

## Usage

### 1. Command Line Interface

```bash
# Basic usage with local checkpoint paths
python Merging/weight_merging.py \
    --checkpoint1 saves/qwen2_5vl-3b/full-sft-imagenet-vqa/checkpoint-50000 \
    --checkpoint2 saves/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa/checkpoint-50000 \
    --lambda_weight 0.5 \
    --output_path saves/qwen2_5vl-3b/merged-model \
    --device cpu

# Mixed usage: HuggingFace base model + local fine-tuned model
python Merging/weight_merging.py \
    --checkpoint1 "Qwen/Qwen2.5-VL-3B-Instruct" \
    --checkpoint2 saves/qwen2_5vl-3b/full-sft-imagenet-vqa/checkpoint-50000 \
    --lambda_weight 0.8 \
    --output_path saves/qwen2_5vl-3b/merged-base-finetuned \
    --device cuda
```

### 2. YAML Configuration

```bash
# Using YAML configuration file
python Merging/weight_merging.py --config Merging/config_qwen2_5vl_example.yaml

# Override specific parameters from YAML
python Merging/weight_merging.py \
    --config Merging/config_qwen2_5vl_example.yaml \
    --lambda_weight 0.7 \
    --device cuda
```

### 3. Python API

```python
from Merging.weight_merging import linear_weight_merging

# Local checkpoints
merged_model = linear_weight_merging(
    checkpoint1_path="saves/qwen2_5vl-3b/full-sft-imagenet-vqa/checkpoint-50000",
    checkpoint2_path="saves/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa/checkpoint-50000", 
    lambda_weight=0.5,
    output_path="saves/qwen2_5vl-3b/merged-model",
    device='cpu'
)

# Mixed: HuggingFace + local checkpoint
merged_model = linear_weight_merging(
    checkpoint1_path="Qwen/Qwen2.5-VL-3B-Instruct",  # HuggingFace
    checkpoint2_path="saves/qwen2_5vl-3b/full-sft-imagenet-vqa/checkpoint-50000",  # Local
    lambda_weight=0.8,
    output_path="saves/qwen2_5vl-3b/merged-base-finetuned",
    device='cuda'
)
```

## YAML Configuration Format

Create a YAML file with the following structure:

```yaml
# Model checkpoints (supports both local paths and HuggingFace identifiers)
checkpoint1: "path/to/first/checkpoint"  # or "Qwen/Qwen2.5-VL-3B-Instruct"
checkpoint2: "path/to/second/checkpoint" # or "liuhaotian/llava-v1.5-7b"

# Interpolation weight (0.0 to 1.0)
lambda_weight: 0.5

# Output path
output_path: "path/to/save/merged/model"

# Device ("cpu" or "cuda")
device: "cpu"
```

### Supported Path Types

- **Local paths**: `saves/model/checkpoint-1000`, `/absolute/path/to/model`
- **HuggingFace identifiers**: `Qwen/Qwen2.5-VL-3B-Instruct`, `liuhaotian/llava-v1.5-7b`
- **Mixed usage**: You can mix local and HuggingFace paths in the same configuration

## Example YAML Files

- **`config_qwen2_5vl_example.yaml`**: Basic Qwen2.5-VL merging (local paths)
- **`config_llava_example.yaml`**: Basic LLaVA merging (local paths)
- **`config_huggingface_example.yaml`**: Mixed local + HuggingFace paths
- **`config_advanced_example.yaml`**: Advanced merging scenarios

## Lambda Weight Guidelines

| Lambda | Effect |
|--------|--------|
| 0.0 | Use only model2 |
| 0.1 | 10% model1 + 90% model2 |
| 0.5 | Equal mixing (50/50) |
| 0.7 | 70% model1 + 30% model2 |
| 0.9 | 90% model1 + 10% model2 |
| 1.0 | Use only model1 |

## Requirements

- PyTorch
- Transformers library
- PyYAML
- Compatible model checkpoints

## Notes

- Both checkpoints must be from the same model architecture
- Ensure sufficient disk space for the merged model
- Use GPU (`device: "cuda"`) for faster merging of large models
- The merged model includes tokenizer and processor from checkpoint1
- **HuggingFace + Local mixing**: Useful for merging base models from HuggingFace with your fine-tuned local checkpoints
- **Automatic downloads**: HuggingFace models are automatically downloaded when first used

## Error Handling

The script provides comprehensive error checking for:
- Missing checkpoint files
- Incompatible model architectures
- Invalid lambda weights
- Configuration validation 