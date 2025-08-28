export WANDB_PROJECT="VLM-Transfer"


llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-Projector/1-lora-RS.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/1-export-RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-Projector/2-lora-Med.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/2-export-Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-Projector/3-lora-AD.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/3-export-AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-Projector/4-lora-Sci.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/4-export-Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/train/LoRA-CL-Align-Projector/5-lora-Fin.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/5-export-Fin.yaml
