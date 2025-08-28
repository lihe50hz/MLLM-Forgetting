export WANDB_PROJECT="VLM-Transfer"

# Train
# cd /pasteur/u/lihe50hz/VLMTrans
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/BSCCM/full-lowlr.yaml

# Eval

# Baseline for BSCCM
cd /pasteur/u/lihe50hz/VLMTrans
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full-low-lr/bsccm-0.yaml

# Full for BSCCM
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full-low-lr/bsccm-4000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-4000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-4000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full-low-lr/bsccm-8000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-8000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-8000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full-low-lr/bsccm-12000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-12000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-12000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full-low-lr/bsccm-16000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-16000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-16000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full-low-lr/bsccm-20000.yaml
cd /pasteur/u/lihe50hz/VLMEvalKit
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-20000 --data MMMU_DEV_VAL
torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-20000 --data VMCBench_DEV
cd /pasteur/u/lihe50hz/VLMTrans

# Train
# cd /pasteur/u/lihe50hz/VLMTrans
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/BSCCM/full.yaml

# Eval

# Baseline for BSCCM
# cd /pasteur/u/lihe50hz/VLMTrans
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full/bsccm-0.yaml

# Full for BSCCM
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full/bsccm-4000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-4000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-4000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full/bsccm-8000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-8000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-8000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full/bsccm-12000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-12000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-12000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full/bsccm-16000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-16000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-16000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/BSCCM/full/bsccm-20000.yaml
# cd /pasteur/u/lihe50hz/VLMEvalKit
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-20000 --data MMMU_DEV_VAL
# torchrun --nproc_per_node=4 --master_port=29501 run.py --model Qwen2.5-VL-3B-Instruct-BSCCM-High-20000 --data VMCBench_DEV
# cd /pasteur/u/lihe50hz/VLMTrans