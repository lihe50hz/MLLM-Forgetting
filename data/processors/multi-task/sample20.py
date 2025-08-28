import json
from typing import List, Dict, Any
import random

def append_image_token_to_text(json_file_path: str, output_file_path: str = None, data_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Read a JSON file and add "<image>\n" to the beginning of the "text" entry of each sample.
    
    Args:
        json_file_path (str): Path to the input JSON file
        output_file_path (str, optional): Path to save the modified JSON. If None, only returns the data.
    
    Returns:
        List[Dict[str, Any]]: List of modified samples
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Randomly select 20 samples
    data_new = random.sample(data, 20)
    # remove question_id key
    for item in data_new:
        item.pop('question_id', None)
    print(data_new[0])

    # Save to output file if specified
    # if output_file_path:
    #     with open(output_file_path, 'w', encoding='utf-8') as file:
    #         json.dump(data_new, file, indent=2, ensure_ascii=False)
    #     print(f"Modified data saved to: {output_file_path}")
    data_list.extend(data_new)
    return data_list


# def process_json_file(input_path: str, output_path: str = None):
#     """
#     Convenience function to process a JSON file and add image tokens.
    
#     Args:
#         input_path (str): Path to the input JSON file
#         output_path (str, optional): Path to save the modified JSON
#     """
#     modified_data = append_image_token_to_text(input_path, output_path)
#     print(f"Processed {len(modified_data)} samples")
#     return modified_data


if __name__ == "__main__":
    # Domain
    input_file_ad = "/pasteur2/u/lihe50hz/MLLM-CL/AD/train.json"
    input_file_rs = "/pasteur2/u/lihe50hz/MLLM-CL/RS/train.json"
    input_file_med = "/pasteur2/u/lihe50hz/MLLM-CL/Med/train.json"
    input_file_sci = "/pasteur2/u/lihe50hz/MLLM-CL/Sci/train.json"
    input_file_fin = "/pasteur2/u/lihe50hz/MLLM-CL/Fin/train.json"
    # Ability
    input_file_app = "/pasteur2/u/lihe50hz/MLLM-CL/APP/train.json"
    input_file_ocr = "/pasteur2/u/lihe50hz/MLLM-CL/OCR/train.json"
    input_file_vp = "/pasteur2/u/lihe50hz/MLLM-CL/VP/train.json"
    input_file_math = "/pasteur2/u/lihe50hz/MLLM-CL/Math/train.json"

    # Domain
    Domain_data_list = []
    Domain_data_list = append_image_token_to_text(input_file_ad, None, Domain_data_list)
    Domain_data_list = append_image_token_to_text(input_file_rs, None, Domain_data_list)
    Domain_data_list = append_image_token_to_text(input_file_med, None, Domain_data_list)
    Domain_data_list = append_image_token_to_text(input_file_sci, None, Domain_data_list)
    Domain_data_list = append_image_token_to_text(input_file_fin, None, Domain_data_list)
    # Ability
    Ability_data_list = []
    Ability_data_list = append_image_token_to_text(input_file_app, None, Ability_data_list)
    Ability_data_list = append_image_token_to_text(input_file_ocr, None, Ability_data_list)
    Ability_data_list = append_image_token_to_text(input_file_vp, None, Ability_data_list)
    Ability_data_list = append_image_token_to_text(input_file_math, None, Ability_data_list)

    # Save the data
    with open("/pasteur2/u/lihe50hz/MLLM-CL/Domain_data_list_sample20.json", "w", encoding='utf-8') as f:
        json.dump(Domain_data_list, f, indent=2, ensure_ascii=False)
    print(f"Domain data saved to: /pasteur2/u/lihe50hz/MLLM-CL/Domain_data_list_sample20.json")
    with open("/pasteur2/u/lihe50hz/MLLM-CL/Ability_data_list_sample20.json", "w", encoding='utf-8') as f:
        json.dump(Ability_data_list, f, indent=2, ensure_ascii=False)
    print(f"Ability data saved to: /pasteur2/u/lihe50hz/MLLM-CL/Ability_data_list_sample20.json")   