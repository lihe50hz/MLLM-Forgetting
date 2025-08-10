import argparse
import json
import os
import random
import re

import torch
from PIL import Image
from torchvision.datasets import ImageFolder
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

def main():
    parser = argparse.ArgumentParser(description="Turn ImageNet into a VQA dataset.")
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default="/pasteur2/u/lihe50hz/imagenet",
        help="Root directory of the ImageNet dataset.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="val",
        help="Dataset split to use (e.g., 'train', 'val').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pasteur2/u/lihe50hz/imagenet-vqa-json",
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

    # Load ImageNet dataset
    print(f"Loading ImageNet dataset from: {args.imagenet_root}")
    try:
        dataset = ImageFolder(root=os.path.join(args.imagenet_root, args.dataset_split))
    except Exception as e:
        print(f"Failed to load ImageNet from {args.imagenet_root}: {e}")
        print("Please ensure you have the ImageNet dataset at the specified path.")
        return

    total_samples = len(dataset)

    # Check if output file already exists
    output_json_path = os.path.join(output_dir, f"{args.dataset_split}.json")
    if os.path.exists(output_json_path):
        print(f"Output file {output_json_path} already exists. Exiting.")
        return

    class_metadata = load_dict_from_txt(os.path.join(args.imagenet_root, "metadata.txt"))
    class_names = sorted(dataset.classes)
    class_names_mapping = {filename: class_metadata[i].split(',')[0] for i, filename in enumerate(class_names)}
    print(f"Class names mapping: {class_names_mapping}")
    text_prompts = [f"a photo of {class_names_mapping[filename]}" for filename in class_names]

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
        dataset.samples,
        total=total_samples,
        desc="Processing samples"
    )

    for image_path, correct_label_idx in progress_bar: # type: ignore
        # Load image from path
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        correct_class_name = class_names_mapping[class_names[correct_label_idx]]

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
                distractors.append(class_names_mapping[class_names[int(idx.item())]])
            if len(distractors) == 5:
                break

        # Sample 3 distractors
        distractor_choices = random.sample(distractors, 3)

        # Create choices and shuffle
        choices = [correct_class_name] + distractor_choices
        random.shuffle(choices)

        correct_answer_char = chr(ord("A") + choices.index(correct_class_name))

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
