export WANDB_PROJECT="VLM-Transfer"

conda activate llama
cd /pasteur/u/lihe50hz/VLMTrans/
llamafactory-cli export Eval/General/export.yaml

conda activate vlmeval
cd /pasteur/u/lihe50hz/VLMEvalKit/
torchrun --nproc_per_node=7 run.py --model Qwen2.5-VL-3B-Instruct-Test --data VMCBench_DEV # MMMU_DEV_VAL

cd /pasteur/u/lihe50hz/VLMTrans/
rm -rf /pasteur/u/lihe50hz/VLMEvalKit/outputs/Qwen2.5-VL-3B-Instruct-Test