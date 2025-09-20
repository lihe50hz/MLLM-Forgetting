export WANDB_PROJECT="VLM-Transfer"

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/train/1-full-Pathology-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/1-Pathology-VQA/Pathology-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/1-Pathology-VQA/Surgical-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/1-Pathology-VQA/Cell-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/1-Pathology-VQA/Radiology-VQA.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/train/2-full-Surgical-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/2-Surgical-VQA/Pathology-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/2-Surgical-VQA/Surgical-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/2-Surgical-VQA/Cell-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/2-Surgical-VQA/Radiology-VQA.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/train/3-full-Cell-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/3-Cell-VQA/Pathology-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/3-Cell-VQA/Surgical-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/3-Cell-VQA/Cell-VQA.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/3-Cell-VQA/Radiology-VQA.yaml

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/train/4-full-Radiology-VQA.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/4-Radiology-VQA/Pathology-VQA.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/4-Radiology-VQA/Surgical-VQA.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/4-Radiology-VQA/Cell-VQA.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/full-ml4h-sweep/lr1e-6-warm0.1/eval/4-Radiology-VQA/Radiology-VQA.yaml

