export WANDB_PROJECT="VLM-Transfer"

# cd /pasteur/u/lihe50hz/VLMTrans
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/Surgical/full-low-lr/surgical-0.yaml


# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/full-lowlr-sft-surgical-vqa/checkpoint-4000/config.json ]; do
#     sleep 10
# done
# echo "Surgical-4000 exported"
# sleep 180

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/Surgical/full/surgical-4000.yaml
cd /pasteur/u/lihe50hz/VLMEvalKit
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-4000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-4000 --data VMCBench_DEV
cd /pasteur/u/lihe50hz/VLMTrans

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/full-lowlr-sft-surgical-vqa/checkpoint-8000/config.json ]; do
#     sleep 10
# done
# echo "Surgical-8000 exported"
# sleep 180

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/Surgical/full/surgical-8000.yaml
cd /pasteur/u/lihe50hz/VLMEvalKit
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-8000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-8000 --data VMCBench_DEV
cd /pasteur/u/lihe50hz/VLMTrans

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/full-lowlr-sft-surgical-vqa/checkpoint-12000/config.json ]; do
#     sleep 10
# done
# echo "Surgical-12000 exported"
# sleep 180

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/Surgical/full/surgical-12000.yaml
cd /pasteur/u/lihe50hz/VLMEvalKit
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-12000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-12000 --data VMCBench_DEV
cd /pasteur/u/lihe50hz/VLMTrans

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/full-lowlr-sft-surgical-vqa/checkpoint-16000/config.json ]; do
#     sleep 10
# done
# echo "Surgical-16000 exported"
# sleep 180

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/Surgical/full/surgical-16000.yaml
cd /pasteur/u/lihe50hz/VLMEvalKit
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-16000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-16000 --data VMCBench_DEV
cd /pasteur/u/lihe50hz/VLMTrans

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/full-lowlr-sft-surgical-vqa/checkpoint-20000/config.json ]; do
#     sleep 10
# done
# echo "Surgical-20000 exported"
# sleep 180

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/Surgical/full/surgical-20000.yaml
cd /pasteur/u/lihe50hz/VLMEvalKit
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-20000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-Surgical-High-20000 --data VMCBench_DEV
cd /pasteur/u/lihe50hz/VLMTrans