#!/usr/bin/env python3

import os
import argparse

# Continual Learning Domain configurations
cl_domains = {
    'RS': {
        'dataset': 'RS_train_replay',
        'eval_dataset': 'RS_test',
        'epochs': 1.0,
        'order': 1
    },
    'Med': {
        'dataset': 'Med_train_replay', 
        'eval_dataset': 'Med_test',
        'epochs': 3.0,
        'order': 2
    },
    'AD': {
        'dataset': 'AD_train_replay',
        'eval_dataset': 'AD_test', 
        'epochs': 1.0,
        'order': 3
    },
    'Sci': {
        'dataset': 'Sci_train_replay',
        'eval_dataset': 'Sci_test',
        'epochs': 2.0,
        'order': 4
    },
    'Fin': {
        'dataset': 'Fin_train_replay',
        'eval_dataset': 'Fin_test',
        'epochs': 1.0,
        'order': 5
    }
}

def generate_cl_sweep(learning_rate=2.0e-5, warmup_ratio=0.03, lora_rank=128, identifier_suffix=""):
    """Generate continual learning domain adaptation sweep files"""
    
    # Create base directory for CL sweep
    cl_base_dir = '/pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep'
    cl_base_dir += f"/{identifier_suffix}"
    
    os.makedirs(cl_base_dir, exist_ok=True)
    
    # Create subdirectories
    train_dir = os.path.join(cl_base_dir, 'train')
    export_dir = os.path.join(cl_base_dir, 'export') 
    eval_dir = os.path.join(cl_base_dir, 'eval')
    scripts_dir = os.path.join(cl_base_dir, 'scripts')
    
    for dir_path in [train_dir, export_dir, eval_dir, scripts_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate training configs
    generate_cl_train_configs(train_dir, learning_rate, warmup_ratio, lora_rank, identifier_suffix)
    
    # Generate export configs
    generate_cl_export_configs(export_dir, identifier_suffix)
    
    # Generate evaluation configs
    generate_cl_eval_configs(eval_dir, identifier_suffix)
    
    # Generate scripts
    generate_cl_scripts(scripts_dir, cl_base_dir, identifier_suffix)
    
    print(f"Generated complete CL sweep in: {cl_base_dir}")

def generate_cl_train_configs(train_dir, learning_rate, warmup_ratio, lora_rank, identifier_suffix):
    """Generate training configuration files"""
    
    train_template = """### model
model_name_or_path: {model_path}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {lora_rank}
lora_target: all # language_model
# full projector 
additional_target: [projector]

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
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: {run_name}

### train
per_device_train_batch_size: 1 # using 8 L40s
gradient_accumulation_steps: 16 # using 8 L40s, 8*4
learning_rate: {learning_rate}
num_train_epochs: {epochs}
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
            model_path = f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{prev_domain}/export"
        
        # Generate config
        content = train_template.format(
            model_path=model_path,
            dataset=config['dataset'],
            output_dir=f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{domain}",
            run_name=f"llava-1.5-7b-lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}-{domain.lower()}",
            learning_rate=learning_rate,
            epochs=config['epochs'],
            warmup_ratio=warmup_ratio,
            lora_rank=lora_rank
        )
        
        filename = f"{i+1}-lora-{domain}.yaml"
        filepath = os.path.join(train_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Created training config: {filepath}")

def generate_cl_export_configs(export_dir, identifier_suffix):
    """Generate export configuration files"""
    
    export_template = """### model
model_name_or_path: {model_path}
adapter_name_or_path: {adapter_path}
template: llava
trust_remote_code: true

### export
export_dir: {export_dir}
export_size: 5
export_device: cpu  # choices: [cpu, auto]
export_legacy_format: false
"""
    
    # Sort domains by order
    sorted_domains = sorted(cl_domains.items(), key=lambda x: x[1]['order'])
    
    for i, (domain, config) in enumerate(sorted_domains):
        # Determine model path
        if i == 0:  # First domain
            model_path = "llava-hf/llava-1.5-7b-hf"
        else:  # Subsequent domains
            prev_domain = sorted_domains[i-1][0]
            model_path = f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{prev_domain}/export"
        
        content = export_template.format(
            model_path=model_path,
            adapter_path=f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{domain}",
            export_dir=f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{domain}/export"
        )
        
        filename = f"{i+1}-export-{domain}.yaml"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Created export config: {filepath}")

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
        
        model_path = f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{train_domain}/export"
        
        # Create eval config for each domain
        for eval_domain, eval_config in cl_domains.items():
            content = eval_template.format(
                model_path=model_path,
                cl_domain=eval_domain,
                eval_dataset=eval_config['eval_dataset'],
                output_dir=f"working/llava-1.5-7b/lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}/{train_domain}/{eval_domain}",
                run_name=f"llava-1.5-7b-lora-sft-cl-replay-align-projector{'-' + identifier_suffix if identifier_suffix else ''}-{train_domain.lower()}-eval-{eval_domain.lower()}"
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
        train_config = os.path.join(base_dir, 'train', f"{i+1}-lora-{domain}.yaml")
        export_config = os.path.join(base_dir, 'export', f"{i+1}-export-{domain}.yaml")
        
        # Add training and export
        script_content += f"llamafactory-cli train {train_config}\n"
        script_content += f"llamafactory-cli export {export_config}\n\n"
        
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
    result_script_content += 'python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline\n'
    
    sorted_domains = sorted(cl_domains.items(), key=lambda x: x[1]['order'])
    
    for domain, config in sorted_domains:
        result_script_content += f'echo "-------After {domain}--------"\n'
        result_script_content += f'python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-replay-align-projector{"-" + identifier_suffix if identifier_suffix else ""}/{domain}\n'
    
    result_script_path = os.path.join(scripts_dir, f"result-cl-sweep{'-' + identifier_suffix if identifier_suffix else ''}.sh")
    with open(result_script_path, 'w') as f:
        f.write(result_script_content)
    print(f"Created result evaluation script: {result_script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate continual learning sweep files for domain adaptation')
    parser.add_argument('--learning_rate', type=float, default=2.0e-5, help='Learning rate for continual learning sweep')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio for continual learning sweep')
    parser.add_argument('--lora_rank', type=int, default=128, help='LoRA rank for continual learning sweep')
    parser.add_argument('--identifier', type=str, default='default', help='Identifier suffix for file naming')
    
    args = parser.parse_args()
    
    generate_cl_sweep(learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, lora_rank=args.lora_rank, identifier_suffix=args.identifier)
    print("Continual learning sweep generated successfully!")