export WANDB_PROJECT="VLM-Transfer"

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/train/1-lora-RS.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/export/1-export-RS.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/1-RS/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/1-RS/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/1-RS/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/1-RS/Sci.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/1-RS/Fin.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/train/2-lora-Med.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/export/2-export-Med.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/2-Med/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/2-Med/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/2-Med/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/2-Med/Sci.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/2-Med/Fin.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/train/3-lora-AD.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/export/3-export-AD.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/3-AD/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/3-AD/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/3-AD/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/3-AD/Sci.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/3-AD/Fin.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/train/4-lora-Sci.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/export/4-export-Sci.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/4-Sci/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/4-Sci/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/4-Sci/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/4-Sci/Sci.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/4-Sci/Fin.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/train/5-lora-Fin.yaml
llamafactory-cli export /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/export/5-export-Fin.yaml

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/5-Fin/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/5-Fin/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/5-Fin/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/5-Fin/Sci.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/CL-domain-replay-sweep/lr1e-4-warm0.1-rank8/eval/5-Fin/Fin.yaml

