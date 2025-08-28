export WANDB_PROJECT="VLM-Transfer"

# llamafactory-cli train Qwen2.5-VL-3B/train/CL-ability/1-lora-OCR-8gpu.yaml
# llamafactory-cli export Qwen2.5-VL-3B/temp/CL-ability/1-export-OCR-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/train/CL-ability/2-lora-Math-8gpu.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/CL-ability/2-export-Math-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/train/CL-ability/3-lora-VP-8gpu.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/CL-ability/3-export-VP-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/train/CL-ability/4-lora-APP-8gpu.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/CL-ability/4-export-APP-20000.yaml