import os
import sys
import argparse
from pathlib import Path


def create_full_config_yaml(hybrid_ocrvqa_number: float, one_minus_hybrid_ocrvqa_number: float, folder_name: str) -> str:
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
dataset: imagenet_vqa_train_json,ocrvqa_train
mix_strategy: interleave_over
interleave_probs: [{one_minus_hybrid_ocrvqa_number}, {hybrid_ocrvqa_number}]
template: qwen2_vl
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: outputs/mixing-ocrvqa/full-lowlr-sft-imagenet-vqa-{folder_name}
logging_steps: 10
save_steps: 80000
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
model_name_or_path: outputs/mixing-ocrvqa/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}
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
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: outputs/mixing-ocrvqa/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}/eval
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb
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
model_name_or_path: outputs/mixing-ocrvqa/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}
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

### output
output_dir: outputs/mixing-ocrvqa/full-lowlr-sft-imagenet-vqa-{folder_name}/checkpoint-{checkpoint_step}/imagewikiqa
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb 
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
    checkpoint_steps = list(range(10000, 90000, 10000))
    # Add imagenet evaluation commands
    for step in checkpoint_steps:
        script_content += f"llamafactory-cli train {folder_path}/imagenet-{step}.yaml\n"
    
    script_content += "\n"
    
    # Add imagewikiqa evaluation commands
    for step in checkpoint_steps:
        script_content += f"llamafactory-cli train {folder_path}/imagewikiqa-{step}.yaml\n"
    
    return script_content


def generate_hybrid_ocrvqa_folder(input_number: float):
    """Generate the complete hybrid-ocrvqa folder structure."""
    # Create folder name
    folder_name = f"hybrid-ocrvqa-{input_number}"
    
    # Get the current directory (should be Qwen2.5-VL-3B)
    base_dir = "configs/Qwen/Qwen2.5-VL-3B/mixing-ocrvqa"
    folder_path = os.path.join(base_dir, folder_name)
    output_path = f"outputs/mixing-ocrvqa/full-lowlr-sft-imagenet-vqa-{folder_name}"
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")
    
    # Generate main training config
    full_config_path = os.path.join(folder_path, f"full-lowlr-{folder_name}.yaml")
    with open(full_config_path, 'w') as f:
        f.write(create_full_config_yaml(input_number, 1 - input_number, folder_name))
    print(f"Created: {full_config_path}")
    
    # Generate evaluation configs for each checkpoint step
    checkpoint_steps = list(range(10000, 90000, 10000))
    
    for step in checkpoint_steps:
        # ImageNet evaluation config
        imagenet_config_path = os.path.join(folder_path, f"imagenet-{step}.yaml")
        with open(imagenet_config_path, 'w') as f:
            f.write(create_imagenet_eval_yaml(folder_name, step))
        print(f"Created: {imagenet_config_path}")
        
        # ImageWikiQA evaluation config
        imagewikiqa_config_path = os.path.join(folder_path, f"imagewikiqa-{step}.yaml")
        with open(imagewikiqa_config_path, 'w') as f:
            f.write(create_imagewikiqa_eval_yaml(folder_name, step))
        print(f"Created: {imagewikiqa_config_path}")
    
    # Generate training/evaluation script
    train_eval_script_path = os.path.join(folder_path, "train-eval.sh")
    with open(train_eval_script_path, 'w') as f:
        f.write(create_train_eval_script(str(folder_path), folder_name))
    
    result_filename = f"result.sh"
    result_script_filepath = os.path.join(folder_path, result_filename)
    result_script_content = ""
    for checkpoint in checkpoint_steps:
        result_script_content += f"echo \"-------{checkpoint}--------\"\n"
        result_script_content += f"python evaluators/result_matcher.py --file {output_path}/checkpoint-{checkpoint}/imagenet/generated_predictions.jsonl\n"
        result_script_content += f"python evaluators/result_matcher.py --file {output_path}/checkpoint-{checkpoint}/imagewikiqa/generated_predictions.jsonl\n"
        result_script_content += f"python evaluators/mix_cls_result_matcher.py --generated_results {output_path}/checkpoint-{checkpoint}/imagewikiqa/generated_predictions.jsonl\n"
    with open(result_script_filepath, 'w') as f:
        f.write(result_script_content)
    print(f"Created: {result_script_filepath}")
    
    print(f"\nGenerated complete folder structure for {folder_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid-ocrvqa folder structure with configurable input number"
    )
    parser.add_argument(
        "--hybrid-ratio", 
        type=float, 
        default=0.5,
        help="Hybrid ratio between 0 and 1 (e.g., 0.1, 0.2, 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not 0 <= args.hybrid_ratio <= 1:
        print("Error: Input number must be between 0 and 1")
        sys.exit(1)
    
    try:
        generate_hybrid_ocrvqa_folder(args.hybrid_ratio)
        print(f"\nSuccess! Generated hybrid-ocrvqa-{args.hybrid_ratio} folder structure.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 