import os
import argparse
import json
import re
import eval_ai2d as ai2d
import eval_finvis as finvis
import eval_pvqa as pvqa
import eval_sci as sci


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default='results/CL')
    return parser.parse_args()


def eval_all(result_dir):
    subfolders = ["RS", "Med", "AD", "Sci", "Fin"]
    acc_list = []
    for subfolder in subfolders:
        result_file = os.path.join(result_dir, subfolder, "generated_predictions.jsonl")
        output_dir = os.path.join(result_dir, subfolder, "eval_results")
        if subfolder == "RS" or subfolder == "AD":
            acc = ai2d.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Med":
            acc = pvqa.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Sci":
            acc = sci.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Fin":
            acc = finvis.eval_single(result_file, output_dir, prefix=subfolder)
        else:
            raise ValueError(f"Invalid subfolder: {subfolder}")
        acc_list.append(acc)
        print(f"Average Accuracy: {sum(acc_list) / len(acc_list)}")
    return acc_list

if __name__ == "__main__":
    args = get_args()
    eval_all(args.result_dir)