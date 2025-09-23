train_full_yaml_template = """### model
model_name_or_path: {base_model}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: {freeze_vision_tower}
freeze_multi_modal_projector: {freeze_multi_modal_projector}
freeze_language_model: {freeze_language_model}
deepspeed: cache/ds_z2_config.json

### dataset
dataset: {train_dataset}
template: {template}
cutoff_len: 16384
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: {save_steps}
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: wandb
run_name: {run_name}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4 # assuming training on 4 GPUs, 4*4=16
learning_rate: {learning_rate}
max_steps: {max_steps}
lr_scheduler_type: {lr_scheduler_type}
warmup_ratio: {warmup_ratio}
bf16: true
ddp_timeout: 180000000
flash_attn: fa2
"""

train_lora_yaml_template = """### model
model_name_or_path: {base_model}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {lora_rank}
lora_target: {lora_target}
deepspeed: cache/ds_z2_config.json

### dataset
dataset: {train_dataset}
template: {template}
cutoff_len: 16384
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: {save_steps}
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: wandb
run_name: {run_name}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8 # assuming training on 4 GPUs, 4*8=32
learning_rate: {learning_rate}
max_steps: {max_steps}
lr_scheduler_type: {lr_scheduler_type}
warmup_ratio: {warmup_ratio}
bf16: true
ddp_timeout: 180000000
flash_attn: fa2
"""


eval_full_yaml_template = """### model
model_name_or_path: {output_dir}/checkpoint-{checkpoint}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_predict: true

### dataset
eval_dataset: {eval_dataset}
template: {template}
cutoff_len: 16384
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}/checkpoint-{checkpoint}/{output_suffix}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb
run_name: {run_name}-checkpoint-{checkpoint}

### eval
per_device_eval_batch_size: 32
predict_with_generate: true
ddp_timeout: 180000000
flash_attn: fa2
"""



eval_lora_yaml_template = """### model
model_name_or_path: {output_dir}/checkpoint-{checkpoint}/export
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_predict: true

### dataset
eval_dataset: {eval_dataset}
template: {template}
cutoff_len: 16384
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}/checkpoint-{checkpoint}/{output_suffix}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb
run_name: {run_name}-checkpoint-{checkpoint}

### eval
per_device_eval_batch_size: 32
predict_with_generate: true
ddp_timeout: 180000000
flash_attn: fa2
"""



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