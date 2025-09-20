export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/hybrid-llava-0.97/full-lowlr-hybrid-llava-0.97.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/hybrid-llava-0.97/imagenet-80000.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/hybrid-llava-0.97/imagewikiqa-80000.yaml
