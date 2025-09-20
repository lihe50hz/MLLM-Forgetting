# Replay-free Sequential Fine-tuning of Medical VLMs

## Environment

```bash
conda env create -f environment.yml
conda activate ml4h
```

## Download the dataset

For Radiology-VQA and Surgical-VQA, you can use the following script to download the original dataset.
```bash
huggingface-cli download eltorio/ROCOv2-radiology --repo-type dataset --local-dir ./ROCOv2
huggingface-cli download UCL-WEISS/PitVis-2023  --repo-type dataset --local-dir ./PitVis-2023
```

For Pathology-VQA, you can use the intruction in the [MR-LoRA](https://github.com/bjzhb666/MLLM-CL) to download the dataset, we are using the **Med** subset of the domain continual learning. This is also the general domain continual learning benchmark we use in the paper.

For Cell-VQA, you can use the instruction in the [BSCCM](https://github.com/Waller-Lab/BSCCM/blob/main/Getting_started.ipynb) to download the dataset.

## Convert the dataset to Medical-CL
The dataset is in the format of VQA, you can use the following script to convert the dataset to VQA format.

```bash
python converters/rocov2_to_vqa.py
python converters/bsccm_to_vqa.py
python converters/surgical_to_vqa.py
```

## Train the model

We directly use the LLaVA-1.5-7B model for the training. And our codebase doesn't contain any fancy method and can be easily reproduced using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). You can follow the official instruction to configure the dataset of MLLM-CL and Medical-CL.

## Evaluation

### MLLM-CL

We strictly follow the instruction in the [MLLM-CL](https://github.com/bjzhb666/MLLM-CL) to evaluate the model.

### Medical-CL

For evaluation, you can use the LLaMA-Factory `predict` argument to generate the predictions. And our evaluation base on the `generated_predictions.jsonl` file and `all_results.jsonl` file it provides.

For Pathology-VQA, you can use the following script to evaluate the model.

```bash
python evaluators/eval_pvqa.py --result-file /path/to/your/results/generated_predictions.jsonl
```

For Surgical-VQA or Cell-VQA, you can use the following script to evaluate the model.

```bash
python evaluators/result_matcher.py --file /path/to/your/results/generated_predictions.jsonl
```

For Radiology-VQA, LLaMA-Factory provides automatic evaluation for **BLEU-4** metrics. You can directly use the `all_results.json` file it provides.
