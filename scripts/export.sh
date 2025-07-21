export WANDB_PROJECT="VLM-Transfer"

conda activate llama
cd /pasteur/u/lihe50hz/VLMTrans/
llamafactory-cli export Qwen2.5-VL-3B/temp/export-10000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-12000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-14000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-16000.yaml
llamafactory-cli export Qwen2.5-VL-3B/temp/export-18000.yaml

conda activate vlmeval
cd /pasteur/u/lihe50hz/VLMEvalKit/
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-10000 --data VMCBench_DEV # rerun
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-12000 --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-14000 --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-16000 --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-18000 --data VMCBench_DEV
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-10000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-12000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-14000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-16000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 run.py --model Qwen2.5-VL-3B-Instruct-18000 --data MMMU_DEV_VAL