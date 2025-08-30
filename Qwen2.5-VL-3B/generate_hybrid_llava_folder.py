#!/usr/bin/env python3
"""
Script to generate a hybrid-llava folder structure with configurable parameters.
Usage: python generate_hybrid_llava_folder.py <number>
Example: python generate_hybrid_llava_folder.py 0.2
"""

import os
import sys
import argparse
from pathlib import Path


def create_full_config_yaml(hybrid_llava_number: float, one_minus_hybrid_llava_number: float, folder_name: str) -> str:
    """Generate the main training configuration YAML content."""
    return f"""### model
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: cache/ds_z2_config.json

### dataset
dataset: imagenet_vqa_train_json,llava_665k_instruct
mix_strategy: interleave_over
interleave_probs: [{one_minus_hybrid_llava_number}, {hybrid_llava_number}]
template: qwen2_vl
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
cache_dir: /pasteur/u/lihe50hz/.cache/huggingface/llama-factory

### output
output_dir: working/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa-{folder_name}
logging_steps: 10
save_steps: 10000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: qwen2_5vl-3b-full-lowlr-sft-imagenet-vqa-{folder_name}

### train
per_device_train_batch_size: 1 # using 4 L40s
gradient_accumulation_steps: 4 # using 4 L40s, 4*4
learning_rate: 1.0e-6
# num_train_epochs: 1.0
max_steps: 80000
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
flash_attn: fa2
"""


def create_imagenet_eval_yaml(folder_name: str, checkpoint_step: int) -> str:
    """Generate ImageNet evaluation configuration YAML content."""
    return f"""### model
model_name_or_path: working/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_predict: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false

### dataset
eval_dataset: imagenet_vqa_val_json
template: qwen2_vl
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
cache_dir: /pasteur/u/lihe50hz/.cache/huggingface/llama-factory

### output
output_dir: working/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}/eval
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: qwen2_5vl-3b-full-lowlr-sft-imagenet-vqa-{folder_name}-checkpoint-{checkpoint_step}-eval

### eval
per_device_eval_batch_size: 32
predict_with_generate: true
ddp_timeout: 180000000
flash_attn: fa2
"""


def create_imagewikiqa_eval_yaml(folder_name: str, checkpoint_step: int) -> str:
    """Generate ImageWikiQA evaluation configuration YAML content."""
    return f"""### model
model_name_or_path: working/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_predict: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false

### dataset
eval_dataset: imagewikiqa_vqa
template: qwen2_vl
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
cache_dir: /pasteur/u/lihe50hz/.cache/huggingface/llama-factory

### output
output_dir: working/qwen2_5vl-3b/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}/imagewikiqa
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: qwen2_5vl-3b-full-lowlr-sft-imagenet-vqa-{folder_name}-checkpoint-{checkpoint_step}-imagewikiqa

### eval
per_device_eval_batch_size: 32
predict_with_generate: true
ddp_timeout: 180000000
flash_attn: fa2
"""


def create_train_eval_script(folder_path: str, folder_name: str) -> str:
    """Generate the training and evaluation shell script content."""
    script_content = 'export WANDB_PROJECT="VLM-Transfer"\n\n'
    
    # Add main training command
    script_content += f"llamafactory-cli train {folder_path}/full-lowlr-{folder_name}.yaml\n\n"
    
    # Add imagenet evaluation commands
    for step in range(10000, 90000, 10000):
        script_content += f"llamafactory-cli train {folder_path}/imagenet-{step}.yaml\n"
    
    script_content += "\n"
    
    # Add imagewikiqa evaluation commands
    for step in range(10000, 90000, 10000):
        script_content += f"llamafactory-cli train {folder_path}/imagewikiqa-{step}.yaml\n"
    
    return script_content


def generate_hybrid_llava_folder(input_number: float):
    """Generate the complete hybrid-llava folder structure."""
    # Create folder name
    folder_name = f"hybrid-llava-{input_number}"
    
    # Get the current directory (should be Qwen2.5-VL-3B)
    base_dir = Path("/pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B")
    folder_path = base_dir / folder_name
    
    # Create the folder
    folder_path.mkdir(exist_ok=True)
    print(f"Created folder: {folder_path}")
    
    # Generate main training config
    full_config_path = folder_path / f"full-lowlr-{folder_name}.yaml"
    with open(full_config_path, 'w') as f:
        f.write(create_full_config_yaml(input_number, 1 - input_number, folder_name))
    print(f"Created: {full_config_path}")
    
    # Generate evaluation configs for each checkpoint step
    checkpoint_steps = range(10000, 90000, 10000)
    
    for step in checkpoint_steps:
        # ImageNet evaluation config
        imagenet_config_path = folder_path / f"imagenet-{step}.yaml"
        with open(imagenet_config_path, 'w') as f:
            f.write(create_imagenet_eval_yaml(folder_name, step))
        print(f"Created: {imagenet_config_path}")
        
        # ImageWikiQA evaluation config
        imagewikiqa_config_path = folder_path / f"imagewikiqa-{step}.yaml"
        with open(imagewikiqa_config_path, 'w') as f:
            f.write(create_imagewikiqa_eval_yaml(folder_name, step))
        print(f"Created: {imagewikiqa_config_path}")
    
    # Generate training/evaluation script
    train_eval_script_path = folder_path / "train-eval.sh"
    with open(train_eval_script_path, 'w') as f:
        f.write(create_train_eval_script(str(folder_path), folder_name))
    
    # Make the script executable
    os.chmod(train_eval_script_path, 0o755)
    print(f"Created: {train_eval_script_path}")
    
    print(f"\nGenerated complete folder structure for {folder_name}")
    print(f"Total files created: {len(list(folder_path.iterdir()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid-llava folder structure with configurable input number"
    )
    parser.add_argument(
        "input_number", 
        type=float, 
        help="Input number between 0 and 1 (e.g., 0.1, 0.2, 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not 0 <= args.input_number <= 1:
        print("Error: Input number must be between 0 and 1")
        sys.exit(1)
    
    try:
        generate_hybrid_llava_folder(args.input_number)
        print(f"\nSuccess! Generated hybrid-llava-{args.input_number} folder structure.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 