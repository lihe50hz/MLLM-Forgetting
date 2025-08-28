export WANDB_PROJECT="VLM-Transfer"

# Train
cd /pasteur/u/lihe50hz/VLMTrans
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/1-full-RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/2-full-Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/3-full-AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/4-full-Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/5-full-Fin.yaml

# Eval
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/eval/RS.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/eval/Med.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/eval/AD.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/eval/Sci.yaml
llamafactory-cli predict /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL-Merge/eval/Fin.yaml