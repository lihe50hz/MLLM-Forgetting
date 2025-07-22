export WANDB_PROJECT="VLM-Transfer"

# Full - Script 3 gets: 10000, 20000
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/caltech101-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/flower102-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/imagenetv2-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/imagewikiqa-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/stanfordcars-10000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full/caltech101-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/flower102-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/imagenetv2-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/imagewikiqa-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full/stanfordcars-20000.yaml

# Encoder - Script 3 gets: 10000, 20000, 30000
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/caltech101-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/flower102-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/imagenetv2-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/imagewikiqa-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/stanfordcars-10000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/caltech101-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/flower102-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/imagenetv2-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/imagewikiqa-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/stanfordcars-20000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/caltech101-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/flower102-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/imagenetv2-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/imagewikiqa-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder/stanfordcars-30000.yaml

# Full+Projector - Script 3 gets: 10000, 20000, 40000
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/caltech101-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/flower102-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/imagenetv2-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/imagewikiqa-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/stanfordcars-10000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/caltech101-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/flower102-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/imagenetv2-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/imagewikiqa-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/stanfordcars-20000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/caltech101-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/flower102-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/imagenetv2-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/imagewikiqa-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full+Projector/stanfordcars-40000.yaml

# Encoder+Projector - Script 3 gets: 10000, 20000
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/caltech101-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/flower102-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/imagenetv2-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/imagewikiqa-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/stanfordcars-10000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/caltech101-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/flower102-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/imagenetv2-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/imagewikiqa-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/stanfordcars-20000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/imagewikiqa-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/stanfordcars-70000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/imagewikiqa-80000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder+Projector/stanfordcars-80000.yaml