export WANDB_PROJECT="VLM-Transfer"

conda activate llama
cd /pasteur/u/lihe50hz/VLMTrans/
llamafactory-cli export Qwen2.5-VL-3B/temp/export-2000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-4000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-6000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-8000.yaml