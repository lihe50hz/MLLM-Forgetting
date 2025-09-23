# Fine-tuning VLMs Without Forgetting Is Easier Than You Think

This repo provides the source code for reproducing the result in paper **Fine-tuning VLMs Without Forgetting Is Easier Than You Think**

## Abstract

*This paper does not propose a new method; rather, we find that simple adjustments of fine-tuning recipes of vision language models (VLM) are sufficient to mitigate catastrophic forgetting.* Using visual question answering tasks, we design a 2×2 experimental framework to assess model performance across in-distribution and out-of-distribution image and text inputs. Our results show that appropriate regularization—such as constraining the number of trainable parameters or adopting a low learning rate—effectively prevents forgetting when dealing with out-of-distribution images. However, we uncover a distinct form of forgetting in settings with in-distribution images and out-of-distribution text. We attribute this forgetting as task-specific overfitting and address this issue by introducing a data-hybrid training strategy that combines datasets and tasks. Finally, we demonstrate that this approach naturally extends to multi-task learning, outperforming existing methods without the need for complex auxiliary mechanisms. Overall, our findings challenge prevailing assumptions by highlighting the inherent robustness of VLMs and provide practical guidelines for adapting them while preserving their general-purpose capabilities.

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



## Reproducing the results

We provide the python code for automized the generation of LLaMA-Factory configuration yaml files. 

### Fine-tuning of ImageNet-VQA, BSCCM and PitVis



### Mixing Datasets Fine-tuning



### Fine-tuning on [MLLM-CL](https://github.com/bjzhb666/MLLM-CL) Dataset