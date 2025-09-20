import os
import argparse
import json
import re
import eval_pvqa as pvqa
from result_matcher import compute_accuracy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default='results/CL')
    return parser.parse_args()


def eval_all(result_dir):
    subfolders = ["Pathology-VQA", "Surgical-VQA", "Cell-VQA", "Radiology-VQA"]
    acc_list = []
    for subfolder in subfolders:
        result_file = os.path.join(result_dir, subfolder, "generated_predictions.jsonl")
        output_dir = os.path.join(result_dir, subfolder, "eval_results")
        if subfolder == "Surgical-VQA" or subfolder == "Cell-VQA":
            acc = compute_accuracy(result_file)
            print(f"{subfolder}: {acc:.2f}%")
        elif subfolder == "Pathology-VQA":
            acc = pvqa.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Radiology-VQA":
            result_file = os.path.join(result_dir, subfolder, "all_results.json")
            with open(result_file, 'r', encoding='utf-8') as f:
                entry = json.loads(f.read())
                print(f"{subfolder}: {entry['predict_bleu-4']}")
                acc = int(entry['predict_bleu-4'])
        else:
            raise ValueError(f"Invalid subfolder: {subfolder}")
        acc_list.append(acc)
        print(f"Average Accuracy: {sum(acc_list) / len(acc_list)}")


if __name__ == "__main__":
    args = get_args()
    eval_all(args.result_dir)