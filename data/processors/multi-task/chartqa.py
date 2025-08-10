import argparse
import io
import json
import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


def find_parquet_files(data_dir: str, split: str) -> List[str]:
    """
    Find all parquet files for a given split in the data directory.
    
    Args:
        data_dir (str): Path to the directory containing parquet files
        split (str): Dataset split (e.g., 'train', 'test', 'val')
    
    Returns:
        List[str]: List of parquet file paths for the given split
    """
    pattern = f"{split}-*-of-*-*.parquet"
    parquet_files = glob.glob(os.path.join(data_dir, pattern))
    parquet_files.sort()  # Ensure consistent ordering
    
    if not parquet_files:
        print(f"Warning: No parquet files found for split '{split}' in {data_dir}")
        print(f"Expected pattern: {pattern}")
    
    return parquet_files


def load_parquet_data(parquet_files: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple parquet files.
    
    Args:
        parquet_files (List[str]): List of parquet file paths
    
    Returns:
        pd.DataFrame: Combined dataframe from all parquet files
    """
    dataframes = []
    
    for file_path in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pd.read_parquet(file_path)
            dataframes.append(df)
            print(f"Loaded {len(df)} samples from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No parquet files could be loaded successfully")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total samples loaded: {len(combined_df)}")
    
    return combined_df


def save_image_from_bytes(image_bytes: bytes, image_path: str) -> bool:
    """
    Save image bytes to a file.
    
    Args:
        image_bytes (bytes): Image data as bytes
        image_path (str): Path where to save the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Load image from bytes and save
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image.save(image_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error saving image to {image_path}: {e}")
        return False


def process_chartqa_split(root_dir: str, split: str, output_format: str = "json") -> None:
    """
    Process ChartQA data for a specific split.
    
    Args:
        root_dir (str): Root directory containing the data
        split (str): Dataset split to process
        output_format (str): Output format, either "json" or "jsonl"
    """
    data_dir = os.path.join(root_dir, "data")
    images_dir = os.path.join(root_dir, "images", split)
    output_dir = os.path.join(root_dir, "processed")
    
    # Create output directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find and load parquet files
    print(f"Processing split: {split}")
    parquet_files = find_parquet_files(data_dir, split)
    
    if not parquet_files:
        print(f"No parquet files found for split '{split}'. Skipping.")
        return
    
    df = load_parquet_data(parquet_files)
    
    # Validate required columns
    required_columns = ["image", "query", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns} in dataframe. Available columns: {list(df.columns)}")
        # For now, let's assume we can work with whatever columns we have
        # You can hardcode the column names here if they're different
    
    vqa_samples = []
    failed_count = 0
    
    print(f"Processing {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} samples"):
        try:
            # Extract data from row
            # Note: Adjust these column names based on actual parquet file structure
            image_data = row.get("image")["bytes"]
            question = row.get("query")
            answer = row.get("label")[0]
            
            if image_data is None:
                print(f"No image data found for sample {idx}")
                failed_count += 1
                continue
            
            # Generate image filename
            image_filename = f"{split}_{idx:06d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # Save image
            if isinstance(image_data, bytes):
                success = save_image_from_bytes(image_data, image_path)
            else:
                print(f"Unexpected image data type for sample {idx}: {type(image_data)}")
                failed_count += 1
                continue
            
            if not success:
                failed_count += 1
                continue
            
            # Create VQA sample
            vqa_sample = {
                "image_path": image_path,
                "question": "<image>\n" + str(question),
                "answer": str(answer)
            }
            vqa_samples.append(vqa_sample)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed_count += 1
            continue
    
    # Save results
    if output_format == "jsonl":
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in vqa_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vqa_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete for split '{split}':")
    print(f"  - Successfully processed: {len(vqa_samples)} samples")
    print(f"  - Failed: {failed_count} samples")
    print(f"  - Images saved to: {images_dir}")
    print(f"  - JSON saved to: {output_file}")


def process_chartqa_dataset(root_dir: str, splits: List[str] = None, output_format: str = "json") -> None:
    """
    Process ChartQA dataset for multiple splits.
    
    Args:
        root_dir (str): Root directory containing the data
        splits (List[str]): List of splits to process. If None, will auto-detect
        output_format (str): Output format, either "json" or "jsonl"
    """
    data_dir = os.path.join(root_dir, "data")
    
    if splits is None:
        # Auto-detect splits from parquet files
        pattern = re.compile(r"^([^-]+)-\d+-of-\d+-[a-f0-9]+\.parquet$")
        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
        
        detected_splits = set()
        for file_path in parquet_files:
            filename = os.path.basename(file_path)
            match = pattern.match(filename)
            if match:
                detected_splits.add(match.group(1))
        
        splits = sorted(list(detected_splits))
        print(f"Auto-detected splits: {splits}")
    
    if not splits:
        print("No splits found or specified. Please check the data directory.")
        return
    
    # Process each split
    for split in splits:
        try:
            process_chartqa_split(root_dir, split, output_format)
        except Exception as e:
            print(f"Error processing split '{split}': {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Process ChartQA dataset from parquet files to JSON format.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing 'data' folder with parquet files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Dataset splits to process (e.g., 'train', 'test'). If not specified, will auto-detect.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format for the processed data.",
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory '{args.root_dir}' does not exist.")
        return
    
    data_dir = os.path.join(args.root_dir, "data")
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print(f"Expected structure: {args.root_dir}/data/{{split}}-*-of-*-*.parquet")
        return
    
    print(f"Processing ChartQA dataset from: {args.root_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output format: {args.output_format}")
    
    process_chartqa_dataset(args.root_dir, args.splits, args.output_format)
    
    print("\nChartQA processing complete!")
    print(f"Results saved under: {args.root_dir}")
    print("Directory structure:")
    print(f"  {args.root_dir}/images/{{split}}/  - Contains extracted images")
    print(f"  {args.root_dir}/processed/  - Contains JSON files for finetuning")


if __name__ == "__main__":
    main()
