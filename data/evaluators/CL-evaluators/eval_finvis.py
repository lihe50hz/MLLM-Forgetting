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
        # problem = result['prompt']
        # if 'Unanswerable' in result['predict'] :
        #     continue
        
        pred: str = result['predict'].strip().lower().replace('.', '').replace(',', '')
        gt: str =  ground_truth.lower()
        if gt == pred or pred.startswith(gt+' ') or pred.endswith(' '+gt): # all are multiple choice questions or TF questions
            right += 1
        # save the result as jsonl
        pred_list.append(dict(
            # question=problem,
            pred=result['predict'].strip().lower().replace('.', '').replace(',', ''),
            ground_truth=ground_truth.lower(),
            score=1 if (gt == pred or pred.startswith(gt+' ') or pred.endswith(' '+gt)) else 0,
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
    return 100. * right / total

def eval_single_original(result_file, output_dir, prefix='Default'):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    pred_list = []
    for result in results:
        ground_truth = result['label']
        # problem = result['prompt']
        # if 'Unanswerable' in result['predict'] :
        #     continue
        
        pred: str = result['predict'].lower().replace(' ', '').replace('.', '')
        gt: str =  ground_truth.lower()
        score = 0
        if gt == pred: # all are multiple choice questions or TF questions
            right += 1
            score = 1
        # save the result as jsonl
        pred_list.append(dict(
            # question=problem,
            pred=result['predict'].lower().replace(' ', '').replace('.', ''),
            ground_truth=ground_truth.lower(),
            score=score,
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
    return 100. * right / total

if __name__ == "__main__":
    args = get_args()
    eval_single(args.result_file, args.output_dir)