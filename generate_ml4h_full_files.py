#!/usr/bin/env python3

import os
import argparse

# Continual Learning Domain configurations
cl_domains = {
    # 'Medical-VQA': {
    #     'dataset': 'medical_vqa_train',
    #     'eval_dataset': 'medical_vqa_test',
    #     'steps': 2000,
    #     'order': 1
    # },
    'Pathology-VQA': {
        'dataset': 'Med_train', 
        'eval_dataset': 'Med_test',
        'steps': 2000,
        'order': 1
    },
    'Surgical-VQA': {
        'dataset': 'surgical_vqa_train',
        'eval_dataset': 'surgical_vqa_test', 
        'steps': 2000,
        'order': 2
    },
    'Cell-VQA': {
        'dataset': 'bsccm_vqa_train',
        'eval_dataset': 'bsccm_vqa_test',
        'steps': 2000,
        'order': 3
    },
    'Radiology-VQA': {
        'dataset': 'radiology_vqa_train',
        'eval_dataset': 'radiology_vqa_test',
        'steps': 2000,
        'order': 4
    },
}

def generate_cl_sweep(learning_rate=1.0e-6, warmup_ratio=0.1, identifier_suffix=""):
    """Generate continual learning domain adaptation sweep files"""
    
    # Create base directory for CL sweep
    cl_base_dir = '/pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep'
    cl_base_dir += f"/{identifier_suffix}"
    
    os.makedirs(cl_base_dir, exist_ok=True)
    
    # Create subdirectories
    train_dir = os.path.join(cl_base_dir, 'train')
    eval_dir = os.path.join(cl_base_dir, 'eval')
    scripts_dir = os.path.join(cl_base_dir, 'scripts')
    
    for dir_path in [train_dir, eval_dir, scripts_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate training configs
    generate_cl_train_configs(train_dir, learning_rate, warmup_ratio, identifier_suffix)
    
    # Generate evaluation configs
    generate_cl_eval_configs(eval_dir, identifier_suffix)
    
    # Generate scripts
    generate_cl_scripts(scripts_dir, cl_base_dir, identifier_suffix)
    
    print(f"Generated complete CL sweep in: {cl_base_dir}")

def generate_cl_train_configs(train_dir, learning_rate, warmup_ratio, identifier_suffix):
    """Generate training configuration files"""
    
    train_template = """### model
model_name_or_path: {model_path}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: false
freeze_language_model: false
deepspeed: cache/ds_z2_config.json

### dataset
dataset: {dataset}
# mix_strategy: concat # interleave_under, interleave_upper
# interleave_probs: [0.2, 0.2, 0.2, 0.2, 0.2]
template: llava
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
cache_dir: /pasteur/u/lihe50hz/.cache/huggingface/llama-factory

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 10000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: {run_name}

### train
per_device_train_batch_size: 1 # using 8 L40s
gradient_accumulation_steps: 4 # using 8 L40s, 8*4
learning_rate: {learning_rate}
max_steps: {steps}
lr_scheduler_type: cosine
warmup_ratio: {warmup_ratio}
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
flash_attn: fa2
"""
    
    # Sort domains by order
    sorted_domains = sorted(cl_domains.items(), key=lambda x: x[1]['order'])
    
    for i, (domain, config) in enumerate(sorted_domains):
        # Determine model path
        if i == 0:  # First domain
            model_path = "llava-hf/llava-1.5-7b-hf"
        else:  # Subsequent domains
            prev_domain = sorted_domains[i-1][0]
            model_path = f"working/llava-1.5-7b/ml4h/full-sft-cl-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{prev_domain}"
        
        # Generate config
        content = train_template.format(
            model_path=model_path,
            dataset=config['dataset'],
            output_dir=f"working/llava-1.5-7b/ml4h/full-sft-cl-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{domain}",
            run_name=f"llava-1.5-7b-ml4h-full-sft-cl-align-projector{'-' + identifier_suffix if identifier_suffix else ''}-{domain.lower()}",
            learning_rate=learning_rate,
            steps=config['steps'],
            warmup_ratio=warmup_ratio,
        )
        
        filename = f"{i+1}-full-{domain}.yaml"
        filepath = os.path.join(train_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Created training config: {filepath}")

def generate_cl_eval_configs(eval_dir, identifier_suffix):
    """Generate evaluation configuration files"""
    
    eval_template = """### model
model_name_or_path: {model_path}
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
media_dir: /pasteur2/u/lihe50hz/MLLM-CL/{cl_domain}
eval_dataset: {eval_dataset}
template: llava
cutoff_len: 16384
# overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
cache_dir: /pasteur/u/lihe50hz/.cache/huggingface/llama-factory

### output
output_dir: {output_dir}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: {run_name}

### eval
per_device_eval_batch_size: 4
predict_with_generate: true
ddp_timeout: 180000000
flash_attn: fa2
"""
    
    # Sort domains by order
    sorted_domains = sorted(cl_domains.items(), key=lambda x: x[1]['order'])
    
    # For each training step, create eval configs for all domains
    for i, (train_domain, train_config) in enumerate(sorted_domains):
        step_dir = os.path.join(eval_dir, f"{i+1}-{train_domain}")
        os.makedirs(step_dir, exist_ok=True)
        
        model_path = f"working/llava-1.5-7b/ml4h/full-sft-cl-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{train_domain}"
        
        # Create eval config for each domain
        for eval_domain, eval_config in cl_domains.items():
            content = eval_template.format(
                model_path=model_path,
                cl_domain="Med" if eval_domain == "Pathology-VQA" else eval_domain,
                eval_dataset=eval_config['eval_dataset'],
                output_dir=f"working/llava-1.5-7b/ml4h/full-sft-cl-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{train_domain}/{eval_domain}",
                run_name=f"llava-1.5-7b-ml4h-full-sft-cl-align-projector{'-' + identifier_suffix if identifier_suffix else ''}-{train_domain.lower()}-eval-{eval_domain.lower()}"
            )
            
            filename = f"{eval_domain}.yaml"
            filepath = os.path.join(step_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"Created eval config: {filepath}")

def generate_cl_scripts(scripts_dir, base_dir, identifier_suffix):
    """Generate combined training and evaluation script"""
    
    # Generate combined script (train + eval)
    script_content = 'export WANDB_PROJECT="VLM-Transfer"\n\n'
    
    sorted_domains = sorted(cl_domains.items(), key=lambda x: x[1]['order'])
    
    for i, (domain, config) in enumerate(sorted_domains):
        train_config = os.path.join(base_dir, 'train', f"{i+1}-full-{domain}.yaml")
        
        # Add training and export
        script_content += f"llamafactory-cli train {train_config}\n"
        
        # Add evaluation commands for all domains immediately after export
        for eval_domain in cl_domains.keys():
            eval_config = os.path.join(base_dir, 'eval', f"{i+1}-{domain}", f"{eval_domain}.yaml")
            # skip all the domains except the last one for faster sweep
            if i == len(sorted_domains) - 1:
                script_content += f"llamafactory-cli train {eval_config}\n"
            else:
                script_content += f"# llamafactory-cli train {eval_config}\n"
        
        script_content += "\n"
    
    script_path = os.path.join(scripts_dir, f"train-eval-cl-sweep{'-' + identifier_suffix if identifier_suffix else ''}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    print(f"Created combined training and evaluation script: {script_path}")
    
    # Generate result evaluation script
    generate_result_script(scripts_dir, identifier_suffix)

def generate_result_script(scripts_dir, identifier_suffix):
    """Generate result evaluation script"""
    
    result_script_content = 'echo "-------baseline--------"\n'
    result_script_content += 'python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl/Baseline\n'
    
    sorted_domains = sorted(cl_domains.items(), key=lambda x: x[1]['order'])
    
    for domain, config in sorted_domains:
        result_script_content += f'echo "-------After {domain}--------"\n'
        result_script_content += f'python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector{"-" + identifier_suffix if identifier_suffix else ""}/{domain}\n'
    
    result_script_path = os.path.join(scripts_dir, f"result-cl-sweep{'-' + identifier_suffix if identifier_suffix else ''}.sh")
    with open(result_script_path, 'w') as f:
        f.write(result_script_content)
    print(f"Created result evaluation script: {result_script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate continual learning sweep files for domain adaptation')
    parser.add_argument('--learning_rate', type=float, default=1.0e-6, help='Learning rate for continual learning sweep')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for continual learning sweep')
    parser.add_argument('--identifier', type=str, default='default', help='Identifier suffix for file naming')
    
    args = parser.parse_args()
    
    generate_cl_sweep(learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, identifier_suffix=args.identifier)
    print("Continual learning sweep generated successfully!")