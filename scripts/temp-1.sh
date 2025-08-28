export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-fix/6-lora-sample20.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/6-export-sample20.yaml

# Eval
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/6-sample20/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/6-sample20/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/6-sample20/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/6-sample20/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/6-sample20/Sci.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align/1-lora-RS.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align/1-export-RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align/2-lora-Med.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align/2-export-Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align/3-lora-AD.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align/3-export-AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align/4-lora-Sci.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align/4-export-Sci.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align/5-lora-Fin.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align/5-export-Fin.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-fix/1-lora-RS.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/1-export-RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-fix/2-lora-Med.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/2-export-Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-fix/3-lora-AD.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/3-export-AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-fix/4-lora-Sci.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/4-export-Sci.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-fix/5-lora-Fin.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/5-export-Fin.yaml



# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/CL/Separate/lora-RS-8gpu.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/CL/Separate/lora-Med-8gpu.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/CL/Separate/lora-AD-8gpu.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/CL/Separate/lora-Sci-8gpu.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/train/CL/Separate/lora-Fin-8gpu.yaml

# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/CL/Separate/export-RS-2000.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/CL/Separate/export-Med-2000.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/CL/Separate/export-AD-2000.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/CL/Separate/export-Sci-2000.yaml
# llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/Qwen2.5-VL-3B/temp/CL/Separate/export-Fin-2000.yaml


# echo "-------baseline--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir saves/qwen2_5vl-3b/baseline-cl
# echo "-------After RS--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/lora-sft-cl/RS
# echo "-------After Med--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/lora-sft-cl/Med
# echo "-------After AD--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/lora-sft-cl/AD
# echo "-------After Sci--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/lora-sft-cl/Sci
# echo "-------After Fin--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/lora-sft-cl/Fin

# echo "-------baseline--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir saves/qwen2_5vl-3b/baseline-cl
# echo "-------After RS--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/full-lowlr-sft-cl/RS
# echo "-------After Med--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/full-lowlr-sft-cl/Med
# echo "-------After AD--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/full-lowlr-sft-cl/AD
# echo "-------After Sci--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/full-lowlr-sft-cl/Sci
# echo "-------After Fin--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/qwen2_5vl-3b/full-lowlr-sft-cl/Fin