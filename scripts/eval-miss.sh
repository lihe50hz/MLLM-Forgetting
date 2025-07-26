export WANDB_PROJECT="VLM-Transfer"

conda activate llama
cd /pasteur/u/lihe50hz/VLMTrans/

llamafactory-cli train Qwen2.5-VL-3B/temp/Baseline-new/caltech101.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Baseline-new/flower102.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Baseline-new/imagenet.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Baseline-new/imagenetv2.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Baseline-new/imagewikiqa.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Baseline-new/stanfordcars.yaml

# 10000 - 60000

llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/caltech101-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/flower102-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenet-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenetv2-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagewikiqa-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/stanfordcars-10000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-10000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-10000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/caltech101-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/flower102-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenet-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenetv2-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagewikiqa-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/stanfordcars-20000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-20000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-20000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/caltech101-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/flower102-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenet-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenetv2-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagewikiqa-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/stanfordcars-30000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-30000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-30000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/caltech101-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/flower102-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenet-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenetv2-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagewikiqa-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/stanfordcars-40000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-40000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-40000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/caltech101-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/flower102-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenet-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenetv2-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagewikiqa-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/stanfordcars-50000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-50000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-50000.yaml

conda activate vlmeval
cd /pasteur/u/lihe50hz/VLMEvalKit/

torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-10000-Full-lowlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-10000-Full-lowlr --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-10000-Encoder-highlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-10000-Encoder-highlr --data MMMU_DEV_VAL

torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-20000-Full-lowlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-20000-Full-lowlr --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-20000-Encoder-highlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-20000-Encoder-highlr --data MMMU_DEV_VAL

torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-30000-Full-lowlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-30000-Full-lowlr --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-30000-Encoder-highlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-30000-Encoder-highlr --data MMMU_DEV_VAL

torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-40000-Full-lowlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-40000-Full-lowlr --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-40000-Encoder-highlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-40000-Encoder-highlr --data MMMU_DEV_VAL

torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-50000-Full-lowlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-50000-Full-lowlr --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-50000-Encoder-highlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-50000-Encoder-highlr --data MMMU_DEV_VAL

torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-60000-Full-lowlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-60000-Full-lowlr --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-60000-Encoder-highlr --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-60000-Encoder-highlr --data MMMU_DEV_VAL

conda activate llama
cd /pasteur/u/lihe50hz/VLMTrans/

llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/caltech101-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/flower102-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenet-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagenetv2-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/imagewikiqa-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Full-lowlr-path/stanfordcars-60000.yaml

llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/caltech101-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/flower102-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenet-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagenetv2-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/imagewikiqa-60000.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/Encoder-highlr/stanfordcars-60000.yaml
