#!/usr/bin/env python3
"""
Script to randomly sample 10% of data from ImageNet VQA train.json
and save it to train_10_percent.json
"""

import json
import random
import os
from pathlib import Path


def sample_10_percent_data():
    """
    Read the ImageNet VQA train.json file, randomly sample 10% of the data,
    and save it to train_10_percent.json
    """
    # Input and output file paths
    input_file = "/pasteur2/u/lihe50hz/imagenet-vqa-json/train/train.json"
    output_file = "/pasteur2/u/lihe50hz/imagenet-vqa-json/train/train_10_percent.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading data from: {input_file}")
    
    # Read the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine the data structure and sample accordingly
    if isinstance(data, list):
        # If data is a list of items
        total_items = len(data)
        sample_size = max(1, int(total_items * 0.1))  # At least 1 item
        
        print(f"Total items: {total_items}")
        print(f"Sampling {sample_size} items (10%)")
        
        # Randomly sample 10% of the data
        sampled_data = random.sample(data, sample_size)
        
    elif isinstance(data, dict):
        # If data is a dictionary, we need to handle it differently
        # Common structure might be {"questions": [...], "annotations": [...]}
        if "questions" in data and "annotations" in data:
            # Handle VQA format with questions and annotations
            questions = data["questions"]
            annotations = data["annotations"]
            
            total_items = len(questions)
            sample_size = max(1, int(total_items * 0.1))
            
            print(f"Total questions: {total_items}")
            print(f"Sampling {sample_size} questions (10%)")
            
            # Sample indices
            sampled_indices = random.sample(range(total_items), sample_size)
            
            # Sample questions and corresponding annotations
            sampled_questions = [questions[i] for i in sampled_indices]
            sampled_annotations = [annotations[i] for i in sampled_indices]
            
            sampled_data = {
                "questions": sampled_questions,
                "annotations": sampled_annotations
            }
        else:
            # For other dictionary structures, sample values if they are lists
            sampled_data = {}
            for key, value in data.items():
                if isinstance(value, list):
                    total_items = len(value)
                    sample_size = max(1, int(total_items * 0.1))
                    sampled_data[key] = random.sample(value, sample_size)
                    print(f"Sampled {sample_size} items from '{key}' (10% of {total_items})")
                else:
                    sampled_data[key] = value
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the sampled data
    print(f"Saving sampled data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print("Sampling completed successfully!")
    
    # Print some statistics
    if isinstance(sampled_data, list):
        print(f"Sampled {len(sampled_data)} items")
    elif isinstance(sampled_data, dict) and "questions" in sampled_data:
        print(f"Sampled {len(sampled_data['questions'])} questions and {len(sampled_data['annotations'])} annotations")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    try:
        sample_10_percent_data()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
