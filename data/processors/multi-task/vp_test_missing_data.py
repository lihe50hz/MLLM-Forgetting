#!/usr/bin/env python3
"""
Script to download missing images from COCO dataset based on JSON file entries.
Reads a JSON file, extracts image filenames, and downloads them using wget.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def read_json_file(json_path: str) -> List[Dict[str, Any]]:
    """
    Read and parse a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of dictionaries from the JSON file
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected JSON structure in {json_path}")
            
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        sys.exit(1)


def extract_image_filenames(data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract image filenames from JSON data.
    
    Args:
        data: List of dictionaries containing image entries
        
    Returns:
        List of image filenames
    """
    image_filenames = []
    
    for entry in data:
        if 'image' in entry:
            image_filename = entry['image']
            if image_filename:  # Skip empty entries
                image_filenames.append(image_filename)
    
    return image_filenames


def create_output_directory(output_dir: str) -> None:
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def download_image(image_filename: str, base_url: str, output_dir: str) -> bool:
    """
    Download a single image using wget.
    
    Args:
        image_filename: Name of the image file
        base_url: Base URL for the image
        output_dir: Directory to save the image
        
    Returns:
        True if download successful, False otherwise
    """
    # Construct the full URL
    image_url = base_url + image_filename
    
    # Construct the output path
    output_path = os.path.join(output_dir, image_filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Image already exists: {image_filename}")
        return True
    
    # Use wget to download the image
    try:
        print(f"Downloading: {image_filename}")
        result = subprocess.run([
            'wget',
            '-O', output_path,
            image_url
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"Successfully downloaded: {image_filename}")
            return True
        else:
            print(f"Failed to download {image_filename}: {result.stderr}")
            # Remove partial file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout downloading {image_filename}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except Exception as e:
        print(f"Error downloading {image_filename}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def download_images(image_filenames: List[str], base_url_1: str, base_url_2: str, output_dir: str) -> None:
    """
    Download all images from the list.
    
    Args:
        image_filenames: List of image filenames to download
        base_url_1: Base URL for downloading images
        base_url_2: Base URL for downloading images
        output_dir: Directory to save images
    """
    successful_downloads = 0
    failed_downloads = 0
    
    print(f"Found {len(image_filenames)} images to download")
    
    for i, image_filename in enumerate(image_filenames, 1):
        print(f"[{i}/{len(image_filenames)}] Processing: {image_filename}")
        
        if download_image(image_filename, base_url_1, output_dir):
            successful_downloads += 1
        elif download_image(image_filename, base_url_2, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
            print(f"Failed to download {image_filename}")
    
    print(f"\nDownload summary:")
    print(f"  Successful: {successful_downloads}")
    print(f"  Failed: {failed_downloads}")
    print(f"  Total: {len(image_filenames)}")


def main():
    """
    Main function to process JSON file and download images.
    """
    # Configuration
    BASE_URL_1 = "http://images.cocodataset.org/train2017/"
    BASE_URL_2 = "http://images.cocodataset.org/val2017/"
    ROOT_DIR = "/pasteur2/u/lihe50hz/MLLM-CL/VP_test"
    output_dir = f"{ROOT_DIR}/images"
    json_file = f"{ROOT_DIR}/test.json"
    
    print(f"Processing JSON file: {json_file}")
    
    # Read JSON file
    data = read_json_file(json_file)
    print(f"Loaded {len(data)} entries from JSON file")
    
    # Extract image filenames
    image_filenames = extract_image_filenames(data)
    
    if not image_filenames:
        print("No image filenames found in the JSON file")
        sys.exit(0)
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Download images
    download_images(image_filenames, BASE_URL_1, BASE_URL_2, output_dir)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
