export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/hybrid-llava-0.9/full-lowlr-hybrid-llava-0.9.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/hybrid-llava-0.9/imagenet-80000.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/hybrid-llava-0.9/imagewikiqa-80000.yaml
