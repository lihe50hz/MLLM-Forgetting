---
license: mit
task_categories:
- visual-question-answering
language:
- en
tags:
- medical
pretty_name: PathVQA
paperswithcode_id: pathvqa
size_categories:
- 10K<n<100K
dataset_info:
  features:
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 3171303616.326
    num_examples: 19654
  - name: test
    num_bytes: 1113474813.05
    num_examples: 6719
  - name: validation
    num_bytes: 1191658832.096
    num_examples: 6259
  download_size: 785414952
  dataset_size: 5476437261.472
---

# Dataset Card for PathVQA

## Dataset Description
PathVQA is a dataset of question-answer pairs on pathology images. The dataset is intended to be used for training and testing 
Medical Visual Question Answering (VQA) systems. The dataset includes both open-ended questions and binary "yes/no" questions. 
The dataset is built from two publicly-available pathology textbooks: "Textbook of Pathology" and "Basic Pathology", and a 
publicly-available digital library: "Pathology Education Informational Resource" (PEIR). The copyrights of images and captions 
belong to the publishers and authors of these two books, and the owners of the PEIR digital library.<br> 

**Repository:** [PathVQA Official GitHub Repository](https://github.com/UCSD-AI4H/PathVQA)<br>
**Paper:** [PathVQA: 30000+ Questions for Medical Visual Question Answering](https://arxiv.org/abs/2003.10286)<br>
**Leaderboard:** [Papers with Code Leaderboard](https://paperswithcode.com/sota/medical-visual-question-answering-on-pathvqa)

### Dataset Summary
The dataset was obtained from the updated Google Drive link shared by the authors on Feb 15, 2023, 
see the [commit](https://github.com/UCSD-AI4H/PathVQA/commit/117e7f4ef88a0e65b0e7f37b98a73d6237a3ceab)
in the GitHub repository. This version of the dataset contains a total of 5,004 images and 32,795 question-answer pairs. 
Out of the 5,004 images, 4,289 images are referenced by a question-answer pair, while 715 images are not used.
There are a few image-question-answer triplets which occur more than once in the same split (training, validation, test). 
After dropping the duplicate image-question-answer triplets, the dataset contains 32,632 question-answer pairs on 4,289 images.

#### Supported Tasks and Leaderboards
The PathVQA dataset has an active leaderboard on [Papers with Code](https://paperswithcode.com/sota/medical-visual-question-answering-on-pathvqa) 
where models are ranked based on three metrics: "Yes/No Accuracy", "Free-form accuracy" and "Overall accuracy". "Yes/No Accuracy" is
the accuracy of a model's generated answers for the subset of binary "yes/no" questions. "Free-form accuracy" is the accuracy 
of a model's generated answers for the subset of open-ended questions. "Overall accuracy" is the accuracy of a model's generated 
answers across all questions.

#### Languages
The question-answer pairs are in English.

## Dataset Structure

### Data Instances
Each instance consists of an image-question-answer triplet.
```
{
  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=CMYK size=309x272>,
  'question': 'where are liver stem cells (oval cells) located?',
  'answer': 'in the canals of hering'
}
```
### Data Fields
- `'image'`: the image referenced by the question-answer pair. 
- `'question'`: the question about the image.
- `'answer'`: the expected answer.

### Data Splits
The dataset is split into training, validation and test. The split is provided directly by the authors.

|                         | Training Set | Validation Set | Test Set |
|-------------------------|:------------:|:--------------:|:--------:|
| QAs                     |19,654        |6,259           |6,719     |
| Images                  |2,599         |832             |858       |
  
## Additional Information

### Licensing Information
The authors have released the dataset under the [MIT License](https://github.com/UCSD-AI4H/PathVQA/blob/master/LICENSE).

### Citation Information
```
@article{he2020pathvqa,
    title={PathVQA: 30000+ Questions for Medical Visual Question Answering},
    author={He, Xuehai and Zhang, Yichen and Mou, Luntian and Xing, Eric and Xie, Pengtao},
    journal={arXiv preprint arXiv:2003.10286},
    year={2020}
}
```