import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

def load_imagewikiqa_data(jsonl_path):
    """
    Load ImageWikiQA dataset from JSONL file.
    
    Args:
        jsonl_path (str): Path to the imagewikiqa.jsonl file
    
    Returns:
        list: List of dictionaries with image paths, questions, and answers
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data

def main():
    parser = argparse.ArgumentParser(description="Convert ImageWikiQA dataset to VQA format (JSON).")
    parser.add_argument(
        "--imagewikiqa_jsonl",
        type=str,
        default="/pasteur/u/lihe50hz/VLMTrans/data/raw_datasets/imagewikiqa.jsonl",
        help="Path to the imagewikiqa.jsonl file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pasteur2/u/lihe50hz/imagewikiqa-vqa-json",
        help="Directory to save the generated dataset.",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ImageWikiQA dataset
    print(f"Loading ImageWikiQA dataset from: {args.imagewikiqa_jsonl}")
    try:
        dataset = load_imagewikiqa_data(args.imagewikiqa_jsonl)
        print(f"Loaded {len(dataset)} samples")
        total_samples = len(dataset)
    except Exception as e:
        print(f"Failed to load ImageWikiQA dataset from {args.imagewikiqa_jsonl}: {e}")
        print("Please ensure you have the imagewikiqa.jsonl file at the specified path.")
        return

    # Check if output file already exists
    output_json_path = os.path.join(args.output_dir, "imagewikiqa.json")
    if os.path.exists(output_json_path):
        print(f"Output file {output_json_path} already exists. Exiting.")
        return

    vqa_dataset = []
    
    print(f"Processing {total_samples} samples...")
    
    progress_bar = tqdm(
        enumerate(dataset, start=0),
        initial=0,
        total=total_samples,
        desc="Processing samples"
    )

    for i, sample in progress_bar: # type: ignore
        try:
            # Load image
            image_path = sample['image']
            question_text = sample['text']
            correct_answer = sample['label']
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}, skipping...")
                continue
                
            # Verify image can be loaded
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Format the prompt with <image> prefix
            prompt = f"<image>\n{question_text}"

            vqa_sample = {
                "image": image_path,
                "prompt": prompt,
                "response": correct_answer,
            }
            vqa_dataset.append(vqa_sample)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save the complete dataset
    print(f"\nSaving dataset to {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vqa_dataset, f, indent=2, ensure_ascii=False)

    print("\nDataset creation complete.")
    final_message = f"""
The dataset has been saved as:
{output_json_path}

To use this dataset in LLaMA-Factory, you can add the following to your `dataset_info.json`:

"imagewikiqa_vqa": {{
  "file_name": "{output_json_path}",
  "columns": {{"prompt": "prompt", "response": "response", "images": "image"}}
}}

Please ensure the path is correctly configured relative to your `dataset_dir`.
"""
    print(final_message)


if __name__ == "__main__":
    main() 