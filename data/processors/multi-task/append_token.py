import json
from typing import List, Dict, Any


def append_image_token_to_text(json_file_path: str, output_file_path: str = None) -> List[Dict[str, Any]]:
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
    
    # Process each sample
    for sample in data:
        if 'text' in sample:
            # Add "<image>\n" to the beginning of the text
            sample['text'] = "<image>\n" + sample['text']
    
    # Save to output file if specified
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"Modified data saved to: {output_file_path}")
    
    return data


def process_json_file(input_path: str, output_path: str = None):
    """
    Convenience function to process a JSON file and add image tokens.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str, optional): Path to save the modified JSON
    """
    modified_data = append_image_token_to_text(input_path, output_path)
    print(f"Processed {len(modified_data)} samples")
    return modified_data


if __name__ == "__main__":
    # Example usage
    # input_file = "/pasteur2/u/lihe50hz/MLLM-CL/AD/test.json"
    # output_file = "/pasteur2/u/lihe50hz/MLLM-CL/AD/test_with_token.json"
    # input_file = "/pasteur2/u/lihe50hz/MLLM-CL/RS/test.json"
    # output_file = "/pasteur2/u/lihe50hz/MLLM-CL/RS/test_with_token.json"
    # input_file = "/pasteur2/u/lihe50hz/MLLM-CL/Med/test.json"
    # output_file = "/pasteur2/u/lihe50hz/MLLM-CL/Med/test_with_token.json"
    # input_file = "/pasteur2/u/lihe50hz/MLLM-CL/Sci/test.json"
    # output_file = "/pasteur2/u/lihe50hz/MLLM-CL/Sci/test_with_token.json"
    # input_file = "/pasteur2/u/lihe50hz/MLLM-CL/Fin/test.json"
    # output_file = "/pasteur2/u/lihe50hz/MLLM-CL/Fin/test_with_token.json"
    # input_file = "/pasteur2/u/lihe50hz/MLLM-CL/APP_test/test.json"
    # output_file = "/pasteur2/u/lihe50hz/MLLM-CL/APP_test/test_with_token.json"
    input_file = "/pasteur2/u/lihe50hz/MLLM-CL/OCR_test/test.json"
    output_file = "/pasteur2/u/lihe50hz/MLLM-CL/OCR_test/test_with_token.json"
    # Process the file
    process_json_file(input_file, output_file)
