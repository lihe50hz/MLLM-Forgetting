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
    for subfolder in subfolders:
        result_file = os.path.join(result_dir, subfolder, "generated_predictions.jsonl")
        output_dir = os.path.join(result_dir, subfolder, "eval_results")
        if subfolder == "RS" or subfolder == "AD":
            ai2d.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Med":
            pvqa.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Sci":
            sci.eval_single(result_file, output_dir, prefix=subfolder)
        elif subfolder == "Fin":
            finvis.eval_single(result_file, output_dir, prefix=subfolder)
        else:
            raise ValueError(f"Invalid subfolder: {subfolder}")


if __name__ == "__main__":
    args = get_args()
    eval_all(args.result_dir)