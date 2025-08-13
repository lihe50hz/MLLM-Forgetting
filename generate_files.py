#!/usr/bin/env python3

import os

# Dataset configurations
datasets = {
    # 'caltech101': {
    #     'eval_dataset': 'caltech101_vqa_test',
    #     'output_suffix': 'caltech101-test',
    #     'run_suffix': 'caltech101-test'
    # },
    # 'stanfordcars': {
    #     'eval_dataset': 'stanfordcars_vqa_test',
    #     'output_suffix': 'stanfordcars-test',
    #     'run_suffix': 'stanfordcars-test'
    # },
    # 'imagewikiqa': {
    #     'eval_dataset': 'imagewikiqa_vqa',
    #     'output_suffix': 'imagewikiqa',
    #     'run_suffix': 'imagewikiqa'
    # },
    # 'imagenetv2': {
    #     'eval_dataset': 'imagenetv2_vqa',
    #     'output_suffix': 'imagenetv2',
    #     'run_suffix': 'imagenetv2'
    # },
    'imagenet': {
        'eval_dataset': 'imagenet_vqa_val_json',
        'output_suffix': 'eval',
        'run_suffix': 'eval'
    },
    'flowers102': {
        'eval_dataset': 'flowers102_vqa_test_json',
        'output_suffix': 'flowers102-test',
        'run_suffix': 'flowers102-test'
    }
}

# Checkpoints to generate
checkpoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

identifier = 'flowers102'

# Base directory
base_dir = '/pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/' + identifier
os.makedirs(base_dir, exist_ok=True)

checkpoint_dir = 'working/qwen2_5vl-3b/full-lowlr-sft-flowers102'
run_name = 'qwen2_5vl-3b-full-lowlr-sft-flowers102'

# YAML template
yaml_template = """### model
model_name_or_path: {checkpoint_dir}/checkpoint-{checkpoint}
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
eval_dataset: {eval_dataset}
template: qwen2_vl
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
cache_dir: /pasteur/u/lihe50hz/.cache/huggingface/llama-factory

### output
output_dir: {checkpoint_dir}/checkpoint-{checkpoint}/{output_suffix}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: {run_name}-checkpoint-{checkpoint}-{run_suffix}

### eval
per_device_eval_batch_size: 32
predict_with_generate: true
ddp_timeout: 180000000
flash_attn: fa2
"""

filepaths = []

def generate_yaml_files():
    for checkpoint in checkpoints:
        for dataset_name, config in datasets.items():
            filename = f"{dataset_name}-{checkpoint}.yaml"
            filepath = os.path.join(base_dir, filename)
            filepaths.append(filepath)

            content = yaml_template.format(
                checkpoint=checkpoint,
                eval_dataset=config['eval_dataset'],
                output_suffix=config['output_suffix'],
                run_suffix=config['run_suffix'],
                run_name=run_name,
                checkpoint_dir=checkpoint_dir
            )

            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"Created: {filepath}")

def generate_script_files():
    script_content = """
export WANDB_PROJECT="VLM-Transfer"
"""

    for filepath in filepaths:
        script_name = f"eval-{identifier}.sh"
        script_filepath = os.path.join('scripts', script_name)
        script_content += f"llamafactory-cli train {filepath}\n"

    with open(script_filepath, 'w') as f:
        f.write(script_content)
    print(f"Created: {script_filepath}")

if __name__ == "__main__":
    generate_yaml_files()
    print("All YAML files generated successfully!") 
    generate_script_files()
    print("All script files generated successfully!")