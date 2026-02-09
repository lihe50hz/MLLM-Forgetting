# Fine-tuning MLLMs Without Forgetting Is Easier Than You Think

This repo provides the source code for reproducing the result in paper **Fine-tuning MLLMs Without Forgetting Is Easier Than You Think**

## Abstract

*This paper does not propose a new method; rather, we find that simple adjustments of fine-tuning recipes of multimodal large language models (MLLM) are sufficient to mitigate catastrophic forgetting.* Using visual question answering tasks, we design a 2×2 experimental framework to assess model performance across in-distribution and out-of-distribution image and text inputs. Our results show that appropriate regularization—such as constraining the number of trainable parameters or adopting a low learning rate—effectively prevents forgetting when dealing with out-of-distribution images. However, we uncover a distinct form of forgetting in settings with in-distribution images and out-of-distribution text. We attribute this forgetting as task-specific overfitting and address this issue by introducing a data-hybrid training strategy that combines datasets and tasks. Finally, we demonstrate that this approach naturally extends to multi-task learning, outperforming existing methods without the need for complex auxiliary mechanisms. Overall, our findings challenge prevailing assumptions by highlighting the inherent robustness of MLLMs and provide practical guidelines for adapting them while preserving their general-purpose capabilities.

## Environments

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as our training codebase. You can either following the official instruction of LLaMA-Factory or using the command below to create the training environment:

```bash
conda env create -f factory.yml
conda activate factory
```


We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as our training codebase. You can either following the official instruction of VLMEvalKit or using the command below to create the training environment:

```bash
conda env create -f vlmeval.yml
conda activate vlmeval
```

## Data Preparation

You can dowdload the dataset from the annoymous link [VLM-Forgetting](https://huggingface.co/datasets/VLM-Forgetting/vlm-forgetting-datasets), please put the files under the directory `vlm-forgetting-datasets/`. This path has already been included in the `.gitignore` file.

After dowdload the raw datasets, please extract the images in each subfolder directly with:
```
tar -xzvf *.tar.gz
```

The path should be matched with the configuration in `data/dataset_info.json`. Due the 50GB limitation of huggingface on each single file, you should run this in the `vlm-forgetting-datasets/imagenet-vqa-json/` folder to get the original `.tar.gz` file to extract:
```
cat train.tar.gz.part-* > train.tar.gz
```



## Reproducing the results

We provide the python code for automized the generation of LLaMA-Factory configuration yaml files. All the generated files will be put under `configs/`.

### Fine-tuning of ImageNet-VQA, BSCCM and PitVis

You can run `config_generation/single-task/bsccm_vqa.py`, `config_generation/single-task/imagenet_vqa.py`, `config_generation/single-task/surgical_vqa.py` to generate yaml files for LLaMA-Factory. 

Then you can run the generated `train-eval.sh` for training and evaluation. 

Finally, you can use `results.sh` to get the result in percentage.

### Mixing Datasets Fine-tuning

You can run `config_generation/mixing-dataset/mixing_flowers_102.py`, `config_generation/mixing-dataset/mixing_ocrvqa.py`, `config_generation/mixing-dataset/mixing_llava_665k.py` to generate yaml files for LLaMA-Factory. 

Then you can run the generated `train-eval.sh` for training and evaluation. 

Finally, you can use `results.sh` to get the result in percentage.

### Fine-tuning on [MLLM-CL](https://github.com/bjzhb666/MLLM-CL) Dataset

You can run `config_generation/multiple-task/mllm_cl_full_replay.py`, `config_generation/multiple-task/mllm_cl_full.py`, `config_generation/multiple-task/mllm_cl_lora_replay.py`, `config_generation/multiple-task/mllm_cl_lora.py` to generate yaml files for LLaMA-Factory. 

Then you can run the generated `train-eval.sh` for training and evaluation. 

Finally, you can use `results.sh` to get the result in percentage.

### MMMU and VMCBench

For evaluation of MMMU and VMCBench, please use the VLMEvalKit and use `--data MMMU_DEV_VAL` and
`--data VMCBench_DEV` corresponding evaluation. Notice that for all result in the paper, we use the **VAL** split of MMMU. 

## Support Code

We include the evaluation scripts in the folder `evaluators/`. You can check the detail of the code to understand our matching rules. We adapt the code from [MLLM-CL](https://github.com/bjzhb666/MLLM-CL) and put them under `evaluators/CL-evaluators/`.

The `data/dataset_info.json` is the configuration file used by LLaMA-Factory and `data/imagewikiqa.jsonl` is used by `evaluators/mix_cls_result_matcher.py` for calculating the mischoice on classification.

The `cache/` folder contains the deepspeed configuration files for LLaMA-Factory.