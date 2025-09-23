from itertools import tee
import os
import argparse
from yaml_templates import train_full_yaml_template, train_lora_yaml_template, eval_full_yaml_template, eval_lora_yaml_template, export_template

TRAIN_DATASET = 'imagenet'

# Dataset configurations
train_datasets = {
    'imagenet': {
        'train_dataset': 'imagenet_vqa_train_json',
        'output_suffix': 'imagenet',
    },
}
eval_datasets = {
    'imagenet': {
        'eval_dataset': 'imagenet_vqa_val_json',
        'output_suffix': 'imagenet',
    },
    'imagenetv2': {
        'eval_dataset': 'imagenetv2_vqa_json',
        'output_suffix': 'imagenetv2',
    },
    'flowers102': {
        'eval_dataset': 'flowers102_vqa_test_json',
        'output_suffix': 'flowers102',
    },
    'caltech101': {
        'eval_dataset': 'caltech101_vqa_test_json',
        'output_suffix': 'caltech101',
    },
    'stanfordcars': {
        'eval_dataset': 'stanfordcars_vqa_test_json',
        'output_suffix': 'stanfordcars',
    },
    'imagewikiqa': {
        'eval_dataset': 'imagewikiqa_vqa_json',
        'output_suffix': 'imagewikiqa',
    },
}

def generate_files(args):
    # Create subdirectories
    train_dir = os.path.join(args.config_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    if 'lora' in args.finetuning_type:
        export_dir = os.path.join(args.config_dir, 'export') 
        os.makedirs(export_dir, exist_ok=True)
    eval_dir = os.path.join(args.config_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    scripts_dir = os.path.join(args.config_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)


    # train yaml file
    if 'full' in args.finetuning_type:
        train_yaml_file = args.train_yaml_template.format(
            base_model=args.base_model,
            train_dataset=args.train_dataset[TRAIN_DATASET]['train_dataset'],
            output_dir=args.output_dir,
            run_name=f"train-{args.identifier}",
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            freeze_vision_tower=args.freeze_vision_tower,
            freeze_multi_modal_projector=args.freeze_multi_modal_projector,
            freeze_language_model=args.freeze_language_model,
            template=args.template,
            save_steps=args.save_steps,
        )
    elif 'lora' in args.finetuning_type:
        train_yaml_file = args.train_yaml_template.format(
            base_model=args.base_model,
            train_dataset=args.train_dataset[TRAIN_DATASET]['train_dataset'],
            output_dir=args.output_dir,
            run_name=f"train-{args.identifier}",
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            lora_rank=args.lora_rank,
            lora_target=args.lora_target,
            template=args.template,
            save_steps=args.save_steps,
        )

    filename = f"train.yaml"
    train_yaml_filepath = os.path.join(train_dir, filename)
    with open(train_yaml_filepath, 'w') as f:
        f.write(train_yaml_file)
    print(f"Created: {train_yaml_filepath}")

    checkpoints = range(args.save_steps, args.max_steps + args.save_steps, args.save_steps)

    # export yaml files
    if 'lora' in args.finetuning_type:
        for checkpoint in checkpoints:
            filename = f"export-{checkpoint}.yaml"
            export_yaml_filepath = os.path.join(export_dir, filename)
            
            content = args.export_template.format(
                model_path=args.base_model,
                adapter_path=os.path.join(args.output_dir, f"checkpoint-{checkpoint}"),
                export_dir=os.path.join(args.output_dir, f"checkpoint-{checkpoint}/export"),
            )

            with open(export_yaml_filepath, 'w') as f:
                f.write(content)
            print(f"Created: {export_yaml_filepath}")


    # eval yaml files
    for checkpoint in checkpoints:
        for dataset_name, config in args.eval_datasets.items():
            filename = f"{dataset_name}-{checkpoint}.yaml"
            eval_yaml_filepath = os.path.join(eval_dir, filename)

            content = args.eval_yaml_template.format(
                checkpoint=checkpoint,
                eval_dataset=config['eval_dataset'],
                output_suffix=config['output_suffix'],
                run_name=f"eval-{args.identifier}-{dataset_name}-{checkpoint}",
                output_dir=args.output_dir,
                template=args.template,
            )

            with open(eval_yaml_filepath, 'w') as f:
                f.write(content)
            
            print(f"Created: {eval_yaml_filepath}")

    # script files
    filename = f"train-eval.sh"
    script_filepath = os.path.join(scripts_dir, filename)
    script_content = f"llamafactory-cli train {train_yaml_filepath}\n"
    for checkpoint in checkpoints:
        if 'lora' in args.finetuning_type:
            export_yaml_filepath = os.path.join(export_dir, f"export-{checkpoint}.yaml")
            script_content += f"llamafactory-cli export {export_yaml_filepath}\n"
        for dataset_name, config in args.eval_datasets.items():
            eval_yaml_filepath = os.path.join(eval_dir, f"{dataset_name}-{checkpoint}.yaml")
            script_content += f"llamafactory-cli train {eval_yaml_filepath}\n"
    with open(script_filepath, 'w') as f:
        f.write(script_content)
    print(f"Created: {script_filepath}")


    result_filename = f"result.sh"
    result_script_filepath = os.path.join(scripts_dir, result_filename)
    result_script_content = ""
    for checkpoint in checkpoints:
        result_script_content += f"echo \"-------{checkpoint}--------\"\n"
        for dataset_name, config in args.eval_datasets.items():
            result_script_content += f"python evaluators/result_matcher.py --file {args.output_dir}/checkpoint-{checkpoint}/{config['output_suffix']}/generated_predictions.jsonl\n"
    with open(result_script_filepath, 'w') as f:
        f.write(result_script_content)
    print(f"Created: {result_script_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate continual learning sweep files for domain adaptation')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-VL-3B', help='Base model for training, options=[Qwen/Qwen2.5-VL-3B, Qwen/Qwen2.5-VL-7B, llava-hf/llava-1.5-7b-hf]')
    parser.add_argument('--finetuning-type', type=str, default='full-llm', help='Finetuning type for training, options=[full-llm, lora-llm, full-encoder, full-projector]')
    parser.add_argument('--max_steps', type=int, default=80000, help='Max steps for training')
    parser.add_argument('--save_steps', type=int, default=10000, help='Save steps for training')

    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank for continual learning sweep')
    parser.add_argument('--learning_rate', type=float, default=1.0e-6, help='Learning rate for continual learning sweep')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='LR scheduler type for training, options=[cosine, linear]')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for continual learning sweep')

    args = parser.parse_args()

    if args.finetuning_type == 'full-llm':
        args.identifier = f"{TRAIN_DATASET}-{args.finetuning_type}-lr{args.learning_rate}-{args.lr_scheduler_type}-warmup{args.warmup_ratio}"
        args.train_yaml_template = train_full_yaml_template
        args.eval_yaml_template = eval_full_yaml_template
    elif args.finetuning_type == 'lora-llm':
        args.identifier = f"{TRAIN_DATASET}-{args.finetuning_type}-rank{args.lora_rank}-lr{args.learning_rate}-{args.lr_scheduler_type}-warmup{args.warmup_ratio}"
        args.train_yaml_template = train_lora_yaml_template
        args.eval_yaml_template = eval_lora_yaml_template
    elif args.finetuning_type == 'full-encoder':
        args.identifier = f"{TRAIN_DATASET}-{args.finetuning_type}-lr{args.learning_rate}-{args.lr_scheduler_type}-warmup{args.warmup_ratio}"
        args.train_yaml_template = train_full_yaml_template
        args.eval_yaml_template = eval_full_yaml_template
    elif args.finetuning_type == 'full-projector':
        args.identifier = f"{TRAIN_DATASET}-{args.finetuning_type}-lr{args.learning_rate}-{args.lr_scheduler_type}-warmup{args.warmup_ratio}"
        args.train_yaml_template = train_full_yaml_template
        args.eval_yaml_template = eval_full_yaml_template
    
    if 'Qwen' in args.base_model:
        args.template = 'qwen2_vl'
    elif 'llava' in args.base_model:
        args.template = 'llava'

    args.export_template = export_template
    
    args.train_dataset = train_datasets
    args.eval_datasets = eval_datasets

    args.config_dir = os.path.join('configs', args.base_model, args.identifier)
    args.output_dir = os.path.join('outputs', args.base_model, args.identifier)
    os.makedirs(args.config_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if 'full' in args.finetuning_type:
        args.freeze_vision_tower = "true" if 'encoder' in args.finetuning_type else "false"    
        args.freeze_multi_modal_projector = "true" if 'projector' in args.finetuning_type else "false"
        args.freeze_language_model = "true" if 'llm' in args.finetuning_type else "false"

    if 'lora' in args.finetuning_type:
        args.lora_target = 'all'
    

    generate_files(args)

    