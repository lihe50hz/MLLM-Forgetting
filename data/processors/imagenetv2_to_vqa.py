import argparse
import io
import json
import os
import random
import re
from itertools import islice
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import ast
import gc

def load_dict_from_txt(filepath):
    """
    Loads a dictionary from a text file where the content is a Python dictionary literal.

    Args:
        filepath (str): The path to the .txt file.

    Returns:
        dict: The loaded dictionary.

    Raises:
        ValueError: If the file content is not a valid Python literal.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        loaded_dict = ast.literal_eval(content)
        if not isinstance(loaded_dict, dict):
            raise ValueError("The content of the file is not a dictionary.")
        return loaded_dict
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing dictionary from file '{filepath}': {e}")

def load_imagenetv2_dataset(imagenetv2_root):
    """
    Load ImageNetV2 dataset from numbered folders.
    
    Args:
        imagenetv2_root (str): Root directory of ImageNetV2 dataset
        
    Returns:
        list: List of (image_path, class_index) tuples
    """
    dataset = []
    imagenetv2_path = Path(imagenetv2_root)
    
    # Iterate through numbered folders (0-999)
    for class_folder in sorted(imagenetv2_path.iterdir()):
        if class_folder.is_dir() and class_folder.name.isdigit():
            class_idx = int(class_folder.name)
            
            # Load all images in this class folder
            for image_file in class_folder.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dataset.append((str(image_file), class_idx))
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Turn ImageNetV2 into a VQA dataset.")
    parser.add_argument(
        "--imagenetv2_root",
        type=str,
        default="/pasteur/u/yuhuiz/data/ImageNetV2/imagenetv2-matched-frequency-format-val",
        help="Root directory of the ImageNetV2 dataset.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/pasteur2/u/lihe50hz/imagenet/metadata.txt",
        help="Path to the ImageNet metadata.txt file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pasteur2/u/lihe50hz/imagenetv2-vqa",
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP model and processor
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model.to(device) # type: ignore

    # Load ImageNetV2 dataset
    print(f"Loading ImageNetV2 dataset from: {args.imagenetv2_root}")
    try:
        dataset = load_imagenetv2_dataset(args.imagenetv2_root)
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load ImageNetV2 dataset from {args.imagenetv2_root}: {e}")
        print("Please ensure you have the ImageNetV2 dataset at the specified path.")
        return

    # Load class metadata
    print(f"Loading class metadata from: {args.metadata_path}")
    try:
        class_metadata = load_dict_from_txt(args.metadata_path)
        print(f"Loaded metadata for {len(class_metadata)} classes")
    except Exception as e:
        print(f"Failed to load metadata from {args.metadata_path}: {e}")
        return

    chunk_size = 5000  # Smaller chunk size for ImageNetV2
    total_samples = len(dataset)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size if total_samples > 0 else 1

    # --- Resume logic ---
    start_index = 0
    chunk_count = 0
    # Pattern to match filenames like 'imagenetv2-00001-of-00002.parquet'
    pattern = re.compile(r"imagenetv2-(\d+)-of-\d+\.parquet")

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

    # Create class names mapping from indices to descriptions
    class_names_mapping = {}
    for i in range(1000):  # ImageNet has 1000 classes (0-999)
        if i in class_metadata:
            class_names_mapping[i] = class_metadata[i].split(',')[0]
        else:
            print(f"Warning: Class {i} not found in metadata")
            class_names_mapping[i] = f"class_{i}"

    print(f"Created class mapping for {len(class_names_mapping)} classes")
    
    # Create text prompts for all classes
    text_prompts = [f"a photo of {class_names_mapping[i]}" for i in range(1000)]

    print("Pre-computing text embeddings for all classes...")
    with torch.no_grad():
        # This can be memory intensive for all classes at once.
        # If you have memory issues, consider batching this part as well.
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

    for i, (image_path, correct_label_idx) in progress_bar: # type: ignore
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}, skipping...")
                continue
                
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            correct_class_name = class_names_mapping[correct_label_idx]

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

            # Get top 5 incorrect predictions
            sorted_indices = torch.argsort(similarity_scores, descending=True)

            distractors = []
            for idx in sorted_indices:
                if idx.item() != correct_label_idx:
                    distractors.append(class_names_mapping[idx.item()])
                if len(distractors) == 5:
                    break

            # Sample 3 distractors
            distractor_choices = random.sample(distractors, 3)

            # Create choices and shuffle
            choices = [correct_class_name] + distractor_choices
            random.shuffle(choices)

            correct_answer_char = chr(ord("A") + choices.index(correct_class_name))

            question = "<image>\nWhat is the class of this image?\n"
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
                file_name = f"imagenetv2-{chunk_count:05d}-of-{total_chunks:05d}.parquet"
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
        file_name = f"imagenetv2-{chunk_count:05d}-of-{total_chunks:05d}.parquet"
        output_parquet_path = os.path.join(args.output_dir, file_name)
        
        print(f"\nSaving final chunk {chunk_count}/{total_chunks} to {output_parquet_path}")
        df = pd.DataFrame(vqa_dataset)
        df.to_parquet(output_parquet_path, index=False)

    print("\nDataset creation complete.")
    final_message = f"""
The ImageNetV2 VQA dataset has been saved in {chunk_count} chunk(s) in the directory:
{args.output_dir}

To use this dataset in LLaMA-Factory, you may need to point to this directory
in your `dataset_info.json`. If LLaMA-Factory supports reading all Parquet files
from a directory, you can use:

"imagenetv2_vqa": {{
  "file_name": "{args.output_dir}",
  "columns": {{"prompt": "prompt", "response": "response", "images": "image"}}
}}

Please ensure the path is correctly configured relative to your `dataset_dir`.
"""
    print(final_message)


if __name__ == "__main__":
    main()
