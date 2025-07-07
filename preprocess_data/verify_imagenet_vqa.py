import argparse
import io
import random

import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Verify the generated ImageNet-VQA dataset.")
    parser.add_argument(
        "--parquet_file",
        type=str,
        default="/pasteur2/u/lihe50hz/imagenet-vqa/val/val-00002-of-00002.parquet",
        help="Path to the Parquet file to verify.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of random samples to display for verification.",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.parquet_file}...")
    try:
        df = pd.read_parquet(args.parquet_file)
    except Exception as e:
        print(f"Failed to load Parquet file: {e}")
        return

    num_rows = len(df)
    print(f"Dataset contains {num_rows} samples.")

    if num_rows == 0:
        print("Dataset is empty. Nothing to verify.")
        return

    # Adjust number of samples if the dataset is smaller than requested
    num_to_verify = min(args.num_samples, num_rows)
    print(f"Displaying {num_to_verify} random samples for verification...\n")

    # Get random indices
    random_indices = random.sample(range(num_rows), num_to_verify)

    for i, index in enumerate(random_indices):
        sample = df.iloc[index]

        image_bytes = sample["image"]
        question = sample["prompt"]
        response = sample["response"]

        # Decode and display image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            print(f"--- Sample #{i + 1} (Row {index}) ---")
            print(f"Question:\n{question}")
            print(f"\nAnswer: {response}")
            # Save image
            image.save(f"sample_{i + 1}.png")
            print("--------------------------------\n")
        except Exception as e:
            print(f"Could not process sample at row {index}: {e}\n")


if __name__ == "__main__":
    main() 