export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-40000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-60000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-80000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-80000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-80000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-80000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-80000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-80000.yaml