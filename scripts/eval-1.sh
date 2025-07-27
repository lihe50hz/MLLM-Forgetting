export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train LLaVA-1.5-7B/eval/caltech101.yaml
llamafactory-cli train LLaVA-1.5-7B/eval/flowers102.yaml
llamafactory-cli train LLaVA-1.5-7B/eval/imagenet.yaml
llamafactory-cli train LLaVA-1.5-7B/eval/imagenetv2.yaml
llamafactory-cli train LLaVA-1.5-7B/eval/imagewikiqa.yaml
llamafactory-cli train LLaVA-1.5-7B/eval/stanfordcars.yaml