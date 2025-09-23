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
        ground_truth = result['label']
        # problem = result['prompt']
        if 'Unanswerable' in result['predict'] :
            continue
        
        pred: str = result['predict'].lower()
        gt: str =  ground_truth.lower()
        # if pred.startswith(gt) or gt.startswith(pred):
        #     right += 1
        # if pred == gt:
        #     right += 1
        score = 0
        if ' ' in gt: 
            if gt in pred:
                right += 1
                score = 1
        else: 
            gt = gt.replace('.', '')
            if ' ' in pred:
                if (' '+gt) in pred or (gt+' ') in pred or (gt+'.') in pred or (gt+',') in pred:
                    right += 1
                    score = 1
            else: 
                if gt in pred:
                    right += 1
                    score = 1
        # save the result as jsonl
        pred_list.append(dict(
            # question=problem,
            pred=result['predict'].lower(),
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