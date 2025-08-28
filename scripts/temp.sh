export WANDB_PROJECT="VLM-Transfer"


llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/1-lora-RS.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/1-export-RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/2-lora-Med.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/2-export-Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/3-lora-AD.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/3-export-AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/4-lora-Sci.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/4-export-Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/5-lora-Fin.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/5-export-Fin.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/eval/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/eval/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/eval/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/eval/Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/LoRA-CL-Merge/eval/Fin.yaml
