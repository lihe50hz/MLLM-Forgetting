#!/usr/bin/env python3
"""
Test script for ROCOv2 to VQA conversion.
This script processes a small subset to verify the conversion works correctly.
"""

import os
import json
import pandas as pd
from PIL import Image
import io

# Configuration for testing
SOURCE_FILE = "/pasteur2/u/lihe50hz/ML4H/ROCOv2/data/train-00000-of-00027.parquet"
TEST_OUTPUT_DIR = "/pasteur2/u/lihe50hz/ML4H/ROCOv2/test_output"
TEST_IMAGES_DIR = "/pasteur2/u/lihe50hz/ML4H/ROCOv2/test_output/images"

# Fixed question template for VQA captioning task
CAPTIONING_QUESTION = "<image>\nProvide a detailed medical caption for this image."

def setup_test_directories():
    """Create test directories."""
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    print(f"Created test directories: {TEST_OUTPUT_DIR}")

def test_conversion():
    """Test the conversion process with a small sample."""
    print("Testing ROCOv2 to VQA conversion...")
    
    setup_test_directories()
    
    # Load first parquet file
    df = pd.read_parquet(SOURCE_FILE)
    print(f"Loaded {len(df)} samples from test file")
    
    # Process first 3 samples
    vqa_samples = []
    
    for i, row in df.head(3).iterrows():
        try:
            # Extract image data
            image_data = row['image']
            image_id = row['image_id']
            caption = row['caption']
            
            print(f"\nProcessing sample {i+1}:")
            print(f"  Image ID: {image_id}")
            print(f"  Caption: {caption[:100]}...")
            
            # Extract and save image
            image_bytes = image_data['bytes']
            image_path = os.path.join(TEST_IMAGES_DIR, f"{image_id}.png")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            print(f"  Image size: {image.size}")
            print(f"  Image mode: {image.mode}")
            
            # Save image
            image.save(image_path, 'PNG')
            print(f"  Saved image to: {image_path}")
            
            # Create VQA sample
            vqa_sample = {
                "image": image_path,
                "question": CAPTIONING_QUESTION,
                "answer": caption
            }
            
            vqa_samples.append(vqa_sample)
            print(f"  Created VQA sample successfully")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue
    
    # Save test JSON
    test_json_path = os.path.join(TEST_OUTPUT_DIR, "test_samples.json")
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(vqa_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nTest completed!")
    print(f"Processed {len(vqa_samples)} samples")
    print(f"Test JSON saved to: {test_json_path}")
    
    # Display sample VQA format
    if vqa_samples:
        print(f"\nSample VQA format:")
        sample = vqa_samples[0]
        print(json.dumps(sample, indent=2)[:500] + "...")

if __name__ == "__main__":
    test_conversion() 