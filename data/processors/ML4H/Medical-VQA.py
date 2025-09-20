#!/usr/bin/env python3

import json
import random
import os
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the dataset from JSON file."""
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")
    return data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.9) -> tuple:
    """Split dataset into training and testing sets."""
    print(f"Splitting dataset with ratio {train_ratio}:{1-train_ratio}")
    
    # Shuffle the data for random split
    data_copy = data.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(data_copy)
    
    # Calculate split point
    split_point = int(len(data_copy) * train_ratio)
    
    train_data = data_copy[:split_point]
    test_data = data_copy[split_point:]
    
    print(f"Train set: {len(train_data)} entries")
    print(f"Test set: {len(test_data)} entries")
    
    return train_data, test_data

def convert_to_alpaca_format(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a multi-turn conversation entry to multiple single-round Alpaca format entries."""
    alpaca_entries = []
    
    # Extract the image and metadata
    base_entry = {
        "id": entry["id"],
        "image": entry["image"],
        "domain": entry["domain"]
    }
    
    # Convert each human-gpt pair to a single Alpaca entry
    conversations = entry["conversations"]
    i = 0
    pair_idx = 1
    
    while i < len(conversations) - 1:
        if conversations[i]["from"] == "human" and conversations[i + 1]["from"] == "gpt":
            alpaca_entry = base_entry.copy()
            alpaca_entry["id"] = f"{entry['id']}_pair_{pair_idx}"
            
            # Alpaca format
            alpaca_entry["instruction"] = conversations[i]["value"]
            alpaca_entry["input"] = ""  # No additional input for VQA
            alpaca_entry["output"] = conversations[i + 1]["value"]
            
            alpaca_entries.append(alpaca_entry)
            pair_idx += 1
            i += 2
        else:
            i += 1
    
    return alpaca_entries

def process_test_data_to_alpaca(test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process test data from ShareGPT to Alpaca format."""
    print("Converting test data to Alpaca format...")
    
    alpaca_data = []
    for entry in test_data:
        alpaca_entries = convert_to_alpaca_format(entry)
        alpaca_data.extend(alpaca_entries)
    
    print(f"Converted {len(test_data)} multi-turn entries to {len(alpaca_data)} single-round Q&A pairs")
    return alpaca_data

def save_dataset(data: List[Dict[str, Any]], file_path: str, format_name: str):
    """Save dataset to JSON file."""
    print(f"Saving {format_name} format dataset to {file_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} entries to {file_path}")

def main():
    """Main processing function."""
    # File paths
    source_file = "/pasteur2/u/lihe50hz/ML4H/Medical-VQA/vqa-rad/train/dataset.json"
    train_output = "/pasteur2/u/lihe50hz/ML4H/Medical-VQA/vqa-rad/train.json"
    test_output = "/pasteur2/u/lihe50hz/ML4H/Medical-VQA/vqa-rad/test.json"
    
    try:
        # Load the dataset
        data = load_dataset(source_file)
        filtered_data = []
        for entry in data:
            entry["image"] = entry["image"].replace("../images/", "/pasteur2/u/lihe50hz/ML4H/Medical-VQA/vqa-rad/images/")
            if os.path.exists(entry["image"]):
                # remove the entry
                filtered_data.append(entry)
                
            
        # Split into train and test sets
        train_data, test_data = split_dataset(filtered_data, train_ratio=0.95)
        
        # Process test data to Alpaca format
        test_data_alpaca = process_test_data_to_alpaca(test_data)
        
        # Save the datasets
        save_dataset(train_data, train_output, "ShareGPT")
        save_dataset(test_data_alpaca, test_output, "Alpaca")
        
        print("\n=== Processing Complete ===")
        print(f"Training data (ShareGPT): {len(train_data)} entries -> {train_output}")
        print(f"Testing data (Alpaca): {len(test_data_alpaca)} Q&A pairs -> {test_output}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
