export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-20000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-30000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-40000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-70000.yaml