export WANDB_PROJECT="VLM-Transfer"

# Train
cd /pasteur/u/lihe50hz/VLMTrans
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/train.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/export.yaml

# Eval
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/eval/RS.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/eval/Med.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/eval/AD.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/eval/Sci.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/MTL-domain/eval/Fin.yaml