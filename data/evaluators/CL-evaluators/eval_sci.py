import os
import argparse
import json
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='results/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='output_dir')
    return parser.parse_args()



def eval_single(result_file, output_dir, prefix='Default'):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    pred_list = []
    for result in results:
        ground_truth = result['label'].strip()
        problem = result['prompt']
        # if 'Unanswerable' in result['predict'] :
        #     continue
        
        pred: str = result['predict'].strip().lower().replace('.', '').replace(',', '').replace('neither', 'no')
        gt: str =  ground_truth.strip().lower().replace('.', '').replace(',', '').replace('neither', 'no')
        # if image.split('/')[-1].split('_')[0]=="AI2D" or image.split('/')[-1].split('_')[0]=="TQA" or image.split('/')[-1].split('_')[0]=="VQA" or image.split('/')[-1].split('_')[0]=="SciVerse":
        if len(gt) == 1: # multiple choice questions
            if gt == pred: 
                item_score = 1
                right += item_score
            else:
                item_score = 0
        else: # MapQA
            if 'Which states' in problem:
                gt_list = gt.split(',')
                len_gt = len(gt_list)
                pred_map_list = pred.split(',')
         
                count = 0
                for gt in gt_list:
                    if gt in pred_map_list:
                        count += 1
                item_score = count / len_gt
                right += item_score
            elif gt in pred or pred in gt:
                item_score = 1
                right += item_score
            else:
                item_score = 0

        # save the result as jsonl
        pred_list.append(dict(
            # question=problem,
            pred=result['predict'].strip().lower().replace('.', '').replace(',', ''),
            ground_truth=ground_truth.lower(),
            score=item_score,
        ))
    print('{} Accuracy: {:.2f}%'.format(prefix.ljust(4), 100. * right / total))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Result.text')
    with open(output_file, 'w') as f:
        f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    
    output_file = os.path.join(output_dir, 'Result.json')
    with open(output_file, 'w') as f:
        for item in pred_list:
            json.dump(item, f)
            f.write('\n')    

def eval_single_original(result_file, output_dir, prefix='Default'):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    pred_list = []
    for result in results:
        ground_truth = result['label']
        problem = result['prompt']
        if 'Unanswerable' in result['predict'] :
            continue
        
        pred: str = result['predict'].lower()
        gt: str =  ground_truth.lower()
        # if image.split('/')[-1].split('_')[0]=="AI2D" or image.split('/')[-1].split('_')[0]=="TQA" or image.split('/')[-1].split('_')[0]=="VQA" or image.split('/')[-1].split('_')[0]=="SciVerse":
        if gt in ['a', 'b', 'c', 'd']: # multiple choice or TF questions
            if gt == pred: 
                item_score = 1
                right += item_score
            else:
                item_score = 0
        else: # MapQA
            if 'Which states' in problem:
                gt_list = gt.split(',')
                len_gt = len(gt_list)
                pred_map_list = pred.split(',')
         
                count = 0
                for gt in gt_list:
                    if gt in pred_map_list:
                        count += 1
                item_score = count / len_gt
                right += item_score
            elif gt in pred:
                item_score = 1
                right += item_score
            else:
                item_score = 0

        # save the result as jsonl
        pred_list.append(dict(
            # question=problem,
            pred=result['predict'].strip().lower().replace('.', '').replace(',', ''),
            ground_truth=ground_truth.lower(),
            score=item_score,
        ))
    print('{} Accuracy: {:.2f}%'.format(prefix.ljust(4), 100. * right / total))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Result.text')
    with open(output_file, 'w') as f:
        f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    
    output_file = os.path.join(output_dir, 'Result.json')
    with open(output_file, 'w') as f:
        for item in pred_list:
            json.dump(item, f)
            f.write('\n')        

if __name__ == "__main__":
    args = get_args()
    eval_single(args.result_file, args.output_dir)