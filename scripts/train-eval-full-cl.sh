export WANDB_PROJECT="VLM-Transfer"

# Train
cd /pasteur/u/lihe50hz/VLMTrans
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/1-full-RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/2-full-Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/3-full-AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/4-full-Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/5-full-Fin.yaml

# Eval
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/eval/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/eval/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/eval/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/eval/Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/Full-lowlr-CL/eval/Fin.yaml