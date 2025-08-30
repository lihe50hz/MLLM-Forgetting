import json
from typing import List, Dict, Any
import random


if __name__ == "__main__":
    # Domain
    input_file_ad = "/pasteur2/u/lihe50hz/MLLM-CL/AD/train.json"
    input_file_rs = "/pasteur2/u/lihe50hz/MLLM-CL/RS/train.json"
    input_file_med = "/pasteur2/u/lihe50hz/MLLM-CL/Med/train.json"
    input_file_sci = "/pasteur2/u/lihe50hz/MLLM-CL/Sci/train.json"
    input_file_fin = "/pasteur2/u/lihe50hz/MLLM-CL/Fin/train.json"
    input_file_list = [input_file_ad, input_file_rs, input_file_med, input_file_sci, input_file_fin]

    # Domain
    for i, input_file in enumerate(input_file_list):
        output_file = input_file.replace("train.json", "train_replay.json")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for append_file in input_file_list[:i]:
            with open(append_file, "r", encoding="utf-8") as f:
                data_append = json.load(f)
            data_new = random.sample(data_append, 20)
            for item in data_new:
                item.pop('question_id', None)
            data.extend(data_new)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {len(data)} samples for {input_file} and saved to {output_file}")
 