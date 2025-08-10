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

def load_flowers_data(jsonl_path, split):
    """
    Load flower dataset from JSONL file for a specific split.
    
    Args:
        jsonl_path (str): Path to the flowers.jsonl file
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

def load_flower_classes(classes_path):
    """
    Load flower class names from JSON file.
    
    Args:
        classes_path (str): Path to the flowers_classes.json file
    
    Returns:
        list: List of flower class names
    """
    with open(classes_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Turn ImageNet into a VQA dataset.")
    parser.add_argument(
        "--flowers_jsonl",
        type=str,
        default="/pasteur/u/lihe50hz/VLMTrans/data/raw_datasets/flowers.jsonl",
        help="Path to the flowers.jsonl file.",
    )
    parser.add_argument(
        "--flowers_classes",
        type=str,
        default="/pasteur/u/lihe50hz/VLMTrans/data/raw_datasets/flowers_classes.json",
        help="Path to the flowers_classes.json file.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'train', 'val').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pasteur2/u/lihe50hz/flowers102-vqa-json",
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

    # Load Flowers-102 dataset
    print(f"Loading Flowers-102 dataset from: {args.flowers_jsonl}")
    try:
        dataset = load_flowers_data(args.flowers_jsonl, args.dataset_split)
        print(f"Loaded {len(dataset)} samples for split '{args.dataset_split}'")
        total_samples = len(dataset)
    except Exception as e:
        print(f"Failed to load Flowers-102 dataset from {args.flowers_jsonl}: {e}")
        print("Please ensure you have the flowers.jsonl file at the specified path.")
        return

    # Load flower class names
    try:
        all_flower_classes = load_flower_classes(args.flowers_classes)
        print(f"Loaded {len(all_flower_classes)} flower classes")
    except Exception as e:
        print(f"Failed to load flower classes from {args.flowers_classes}: {e}")
        return

    # Check if output file already exists
    output_json_path = os.path.join(output_dir, f"{args.dataset_split}.json")
    if os.path.exists(output_json_path):
        print(f"Output file {output_json_path} already exists. Exiting.")
        return

    text_prompts = [f"a photo of {flower_name}" for flower_name in all_flower_classes]

    print("Pre-computing text embeddings for all classes...")
    with torch.no_grad():
        # This can be memory intensive for all classes at once.
        # If you have memory issues, consider batching this part as well.
        text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True) # type: ignore
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()} # type: ignore
        text_features = clip_model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    vqa_dataset = []
    
    print(f"Processing {total_samples} samples...")
    
    progress_bar = tqdm(
        enumerate(dataset, start=0),
        initial=0,
        total=total_samples,
        desc="Processing samples"
    )

    for i, sample in progress_bar: # type: ignore
        # Load image
        image_path = sample['image']
        correct_label = sample['label']

        # Load image from path
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        correct_label_idx = all_flower_classes.index(correct_label)
        

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
                distractors.append(all_flower_classes[int(idx.item())])
            if len(distractors) >= 5:  # Get top 5 distractors
                break

        # Sample 3 distractors from the top distractors
        if len(distractors) >= 3:
            distractor_choices = random.sample(distractors, 3)
        else:
            # If we don't have enough good distractors, sample randomly from all classes
            remaining_classes = [cls for cls in all_flower_classes if cls != correct_label]
            additional_needed = 3 - len(distractors)
            additional_distractors = random.sample(remaining_classes, additional_needed)
            distractor_choices = distractors + additional_distractors

        # Create choices and shuffle
        choices = [correct_label] + distractor_choices
        random.shuffle(choices)

        correct_answer_char = chr(ord("A") + choices.index(correct_label))

        question = "<image>\nWhat is the class of this image? Please answer with a single letter (A, B, C or D).\n"
        for idx, choice in enumerate(choices):
            question += f"{chr(ord('A') + idx)}. {choice}\n"

        vqa_sample = {
            "image": image_path,
            "prompt": question.strip(),
            "response": correct_answer_char,
        }
        vqa_dataset.append(vqa_sample)

    # Save the complete dataset
    print(f"\nSaving dataset to {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vqa_dataset, f, indent=2, ensure_ascii=False)

    print("\nDataset creation complete.")
    final_message = f"""
The dataset has been saved as:
{output_json_path}

To use this dataset in LLaMA-Factory, you can add the following to your `dataset_info.json`:

"imagenet_vqa": {{
  "file_name": "{output_json_path}",
  "columns": {{"prompt": "prompt", "response": "response", "images": "image"}}
}}

Please ensure the path is correctly configured relative to your `dataset_dir`.
"""
    print(final_message)


if __name__ == "__main__":
    main()
