#!/usr/bin/env python3
"""
Convert ROCOv2 parquet data to VQA JSON format.

This script processes ROCOv2 parquet files and converts them to VQA format with:
- Fixed captioning question prompt
- Image extraction to PNG files
- Separate JSON files for train/validation/test splits
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import io
from tqdm import tqdm
import glob

# Configuration
SOURCE_DIR = "/pasteur2/u/lihe50hz/ML4H/ROCOv2/data"
OUTPUT_DIR = "/pasteur2/u/lihe50hz/ML4H/ROCOv2"
IMAGES_DIR = "/pasteur2/u/lihe50hz/ML4H/ROCOv2/images"

# Fixed question template for VQA captioning task
CAPTIONING_QUESTION = "<image>\nProvide a detailed medical caption for this image."

def setup_directories():
    """Create necessary output directories."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"Created images directory: {IMAGES_DIR}")

def extract_image_from_bytes(image_data: Dict, image_id: str) -> str:
    """
    Extract image from bytes and save as PNG file.
    
    Args:
        image_data: Dictionary containing image bytes
        image_id: Unique image identifier
    
    Returns:
        Absolute path to saved PNG file
    """
    image_bytes = image_data['bytes']
    image_path = os.path.join(IMAGES_DIR, f"{image_id}.png")
    
    # Convert bytes to PIL Image and save
    image = Image.open(io.BytesIO(image_bytes))
    image.save(image_path, 'PNG')
    
    return image_path

def process_parquet_files(split: str) -> List[Dict[str, Any]]:
    """
    Process all parquet files for a given split.
    
    Args:
        split: One of 'train', 'validation', 'test'
    
    Returns:
        List of VQA samples
    """
    pattern = os.path.join(SOURCE_DIR, f"{split}-*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    
    print(f"Processing {len(parquet_files)} {split} files...")
    
    vqa_samples = []
    
    for parquet_file in tqdm(parquet_files, desc=f"Processing {split}"):
        df = pd.read_parquet(parquet_file)
        
        for _, row in df.iterrows():
            try:
                # Extract image and save as PNG
                image_path = extract_image_from_bytes(row['image'], row['image_id'])
                
                # Create VQA sample
                vqa_sample = {
                    "image": image_path,
                    "question": CAPTIONING_QUESTION,
                    "answer": row['caption']
                }
                
                vqa_samples.append(vqa_sample)
                
            except Exception as e:
                print(f"Error processing sample {row['image_id']}: {e}")
                continue
    
    print(f"Processed {len(vqa_samples)} samples for {split} split")
    return vqa_samples

def save_vqa_json(vqa_samples: List[Dict[str, Any]], split: str):
    """
    Save VQA samples to JSON file.
    
    Args:
        vqa_samples: List of VQA samples
        split: Split name (train/validation/test)
    """
    output_file = os.path.join(OUTPUT_DIR, f"{split}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vqa_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(vqa_samples)} samples to {output_file}")

def main():
    """Main conversion process."""
    print("Starting ROCOv2 to VQA conversion...")
    
    # Setup output directories
    setup_directories()
    
    # Process each split
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*50}")
        
        try:
            # Process parquet files and convert to VQA format
            vqa_samples = process_parquet_files(split)
            
            # Save to JSON file
            save_vqa_json(vqa_samples, split)
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Conversion completed!")
    print(f"Output files saved in: {OUTPUT_DIR}")
    print(f"Images saved in: {IMAGES_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 