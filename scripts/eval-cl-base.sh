export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/0-Baseline/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/0-Baseline/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/0-Baseline/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/0-Baseline/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/0-Baseline/Sci.yaml