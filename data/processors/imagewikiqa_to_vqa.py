import argparse
import io
import json
import os
import re

import pandas as pd
from PIL import Image
from tqdm import tqdm
import gc

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
    parser = argparse.ArgumentParser(description="Convert ImageWikiQA dataset to VQA format.")
    parser.add_argument(
        "--imagewikiqa_jsonl",
        type=str,
        default="/pasteur/u/lihe50hz/VLMTrans/data/raw_datasets/imagewikiqa.jsonl",
        help="Path to the imagewikiqa.jsonl file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pasteur2/u/lihe50hz/imagewikiqa-vqa",
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
    except Exception as e:
        print(f"Failed to load ImageWikiQA dataset from {args.imagewikiqa_jsonl}: {e}")
        print("Please ensure you have the imagewikiqa.jsonl file at the specified path.")
        return

    chunk_size = 500  # Smaller chunk size for ImageWikiQA dataset
    total_samples = len(dataset)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size if total_samples > 0 else 1

    # --- Resume logic ---
    start_index = 0
    chunk_count = 0
    # Pattern to match filenames like 'imagewikiqa-00001-of-00004.parquet'
    pattern = re.compile(r"imagewikiqa-(\d+)-of-\d+\.parquet")

    if os.path.exists(args.output_dir):
        existing_chunks = [
            int(match.group(1))
            for filename in os.listdir(args.output_dir)
            if (match := pattern.match(filename))
        ]
        if existing_chunks:
            chunk_count = max(existing_chunks)
            start_index = chunk_count * chunk_size
            print(f"Found {len(existing_chunks)} existing chunks. Resuming from sample {start_index} (after chunk {chunk_count}).")

    if start_index >= total_samples and total_samples > 0:
        print("Dataset already fully processed. Exiting.")
        return

    vqa_dataset = []
    
    print(f"Processing {total_samples - start_index} samples (from index {start_index})...")
    
    # Create iterator from the start index
    dataset_iterator = dataset[start_index:]
    
    progress_bar = tqdm(
        enumerate(dataset_iterator, start=start_index),
        initial=start_index,
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
                
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert image to bytes
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # Format the prompt with <image> prefix
            prompt = f"<image>\n{question_text}"

            vqa_sample = {
                "image": image_bytes,
                "prompt": prompt,
                "response": correct_answer,
            }
            vqa_dataset.append(vqa_sample)

            # If chunk is full, save it
            if len(vqa_dataset) >= chunk_size:
                chunk_count += 1
                file_name = f"imagewikiqa-{chunk_count:05d}-of-{total_chunks:05d}.parquet"
                output_parquet_path = os.path.join(args.output_dir, file_name)
                
                print(f"\nSaving chunk {chunk_count}/{total_chunks} to {output_parquet_path}")
                df = pd.DataFrame(vqa_dataset)
                df.to_parquet(output_parquet_path, index=False)
                
                vqa_dataset = [] # Reset for next chunk
                
                # Garbage collection to free memory
                gc.collect()

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save the final chunk if any samples are left
    if vqa_dataset:
        chunk_count += 1
        file_name = f"imagewikiqa-{chunk_count:05d}-of-{total_chunks:05d}.parquet"
        output_parquet_path = os.path.join(args.output_dir, file_name)
        
        print(f"\nSaving final chunk {chunk_count}/{total_chunks} to {output_parquet_path}")
        df = pd.DataFrame(vqa_dataset)
        df.to_parquet(output_parquet_path, index=False)

    print("\nDataset creation complete.")
    final_message = f"""
The ImageWikiQA VQA dataset has been saved in {chunk_count} chunk(s) in the directory:
{args.output_dir}

To use this dataset in LLaMA-Factory, you may need to point to this directory
in your `dataset_info.json`. If LLaMA-Factory supports reading all Parquet files
from a directory, you can use:

"imagewikiqa_vqa": {{
  "file_name": "{args.output_dir}",
  "columns": {{"prompt": "prompt", "response": "response", "images": "image"}}
}}

Please ensure the path is correctly configured relative to your `dataset_dir`.
"""
    print(final_message)


if __name__ == "__main__":
    main()
