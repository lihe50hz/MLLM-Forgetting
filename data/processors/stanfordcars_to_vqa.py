import argparse
import io
import json
import os
import random
import re
from itertools import islice

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import gc

def load_cars_data(jsonl_path, split):
    """
    Load car dataset from JSONL file for a specific split.
    
    Args:
        jsonl_path (str): Path to the cars.jsonl file
        split (str): Dataset split to load ('train', 'test', or 'all')
    
    Returns:
        list: List of dictionaries with image paths and labels
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            if split == 'all' or entry['split'] == split:
                data.append(entry)
    return data

def load_car_classes(classes_path):
    """
    Load car class names from JSON file.
    
    Args:
        classes_path (str): Path to the cars_classes.json file
    
    Returns:
        list: List of car class names
    """
    with open(classes_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Turn Stanford Cars into a VQA dataset.")
    parser.add_argument(
        "--cars_jsonl",
        type=str,
        default="/pasteur/u/lihe50hz/VLMTrans/data/raw_datasets/cars.jsonl",
        help="Path to the cars.jsonl file.",
    )
    parser.add_argument(
        "--cars_classes",
        type=str,
        default="/pasteur/u/lihe50hz/VLMTrans/data/raw_datasets/cars_classes.json",
        help="Path to the cars_classes.json file.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use ('train', 'test', or 'all').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pasteur2/u/lihe50hz/stanfordcars-vqa",
        help="Directory to save the generated dataset.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model to use for scoring.",
    )
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directories
    output_dir = os.path.join(args.output_dir, args.dataset_split)
    os.makedirs(output_dir, exist_ok=True)

    # Load CLIP model and processor
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model.to(device) # type: ignore

    # Load Stanford Cars dataset
    print(f"Loading Stanford Cars dataset from: {args.cars_jsonl}")
    try:
        dataset = load_cars_data(args.cars_jsonl, args.dataset_split)
        print(f"Loaded {len(dataset)} samples for split '{args.dataset_split}'")
    except Exception as e:
        print(f"Failed to load Stanford Cars dataset from {args.cars_jsonl}: {e}")
        print("Please ensure you have the cars.jsonl file at the specified path.")
        return

    # Load car class names
    try:
        all_car_classes = load_car_classes(args.cars_classes)
        print(f"Loaded {len(all_car_classes)} car classes")
    except Exception as e:
        print(f"Failed to load car classes from {args.cars_classes}: {e}")
        return

    chunk_size = 2000  # Moderate chunk size for cars dataset
    total_samples = len(dataset)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size if total_samples > 0 else 1

    # --- Resume logic ---
    start_index = 0
    chunk_count = 0
    # Pattern to match filenames like 'test-00001-of-00005.parquet'
    pattern = re.compile(rf"{args.dataset_split}-(\d+)-of-\d+\.parquet")

    if os.path.exists(output_dir):
        existing_chunks = [
            int(match.group(1))
            for filename in os.listdir(output_dir)
            if (match := pattern.match(filename))
        ]
        if existing_chunks:
            chunk_count = max(existing_chunks)
            start_index = chunk_count * chunk_size
            print(f"Found {len(existing_chunks)} existing chunks. Resuming from sample {start_index} (after chunk {chunk_count}).")

    if start_index >= total_samples and total_samples > 0:
        print("Dataset already fully processed. Exiting.")
        return

    # Create text prompts for all car classes
    text_prompts = [f"a photo of {car_name}" for car_name in all_car_classes]

    print("Pre-computing text embeddings for all car classes...")
    with torch.no_grad():
        # Batch process text embeddings to avoid memory issues
        text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True) # type: ignore
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()} # type: ignore
        text_features = clip_model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

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
            correct_label = sample['label']
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}, skipping...")
                continue
                
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Find the correct class index
            try:
                correct_label_idx = all_car_classes.index(correct_label)
            except ValueError:
                print(f"Warning: Label '{correct_label}' not found in class list, skipping...")
                continue

            # Convert image to bytes
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # Process image with CLIP
            with torch.no_grad():
                image_inputs = clip_processor(images=image, return_tensors="pt") # type: ignore
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()} # type: ignore
                image_features = clip_model.get_image_features(**image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores
            similarity_scores = (image_features @ text_features.T)[0]

            # Get top incorrect predictions as distractors
            sorted_indices = torch.argsort(similarity_scores, descending=True)

            distractors = []
            for idx in sorted_indices:
                if idx.item() != correct_label_idx:
                    distractors.append(all_car_classes[int(idx.item())])
                if len(distractors) >= 5:  # Get top 5 distractors
                    break

            # Sample 3 distractors from the top distractors
            if len(distractors) >= 3:
                distractor_choices = random.sample(distractors, 3)
            else:
                # If we don't have enough good distractors, sample randomly from all classes
                remaining_classes = [cls for cls in all_car_classes if cls != correct_label]
                additional_needed = 3 - len(distractors)
                additional_distractors = random.sample(remaining_classes, additional_needed)
                distractor_choices = distractors + additional_distractors

            # Create choices and shuffle
            choices = [correct_label] + distractor_choices
            random.shuffle(choices)

            correct_answer_char = chr(ord("A") + choices.index(correct_label))

            question = "<image>\nWhat is the make and model of the car in this image?\n"
            for idx, choice in enumerate(choices):
                question += f"{chr(ord('A') + idx)}. {choice}\n"

            vqa_sample = {
                "image": image_bytes,
                "prompt": question.strip(),
                "response": correct_answer_char,
            }
            vqa_dataset.append(vqa_sample)

            # If chunk is full, save it
            if len(vqa_dataset) >= chunk_size:
                chunk_count += 1
                file_name = f"{args.dataset_split}-{chunk_count:05d}-of-{total_chunks:05d}.parquet"
                output_parquet_path = os.path.join(output_dir, file_name)
                
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
        file_name = f"{args.dataset_split}-{chunk_count:05d}-of-{total_chunks:05d}.parquet"
        output_parquet_path = os.path.join(output_dir, file_name)
        
        print(f"\nSaving final chunk {chunk_count}/{total_chunks} to {output_parquet_path}")
        df = pd.DataFrame(vqa_dataset)
        df.to_parquet(output_parquet_path, index=False)

    print("\nDataset creation complete.")
    final_message = f"""
The Stanford Cars VQA dataset has been saved in {chunk_count} chunk(s) in the directory:
{output_dir}

To use this dataset in LLaMA-Factory, you may need to point to this directory
in your `dataset_info.json`. If LLaMA-Factory supports reading all Parquet files
from a directory, you can use:

"stanfordcars_vqa": {{
  "file_name": "{output_dir}",
  "columns": {{"prompt": "prompt", "response": "response", "images": "image"}}
}}

Please ensure the path is correctly configured relative to your `dataset_dir`.
"""
    print(final_message)


if __name__ == "__main__":
    main()
