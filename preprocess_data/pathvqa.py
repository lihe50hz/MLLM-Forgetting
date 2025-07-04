import os
from datasets import load_dataset, Dataset

# --- Configuration Section ---
# Base directory where your Parquet files are located
BASE_DATA_DIR = "/pasteur/u/lihe50hz/VLMTrans/flaviagiammarino/path-vqa/data"

# Define the split you want to process (e.g., 'train', 'test', 'validation')
SPLIT_TO_PROCESS = "train"

# The name of the text column in your dataset
TEXT_COLUMN_NAME = "question" # Based on common VQA datasets, 'question' is more likely than 'text'
                              # Please double-check your dataset features if it's different.

# --- Step 1: Prepare file paths for the specific split ---
# Get all parquet files for the chosen split
parquet_files = [
    os.path.join(BASE_DATA_DIR, f)
    for f in os.listdir(BASE_DATA_DIR)
    if f.startswith(f"{SPLIT_TO_PROCESS}-") and f.endswith(".parquet")
]

if not parquet_files:
    print(f"Error: No Parquet files found for split '{SPLIT_TO_PROCESS}' in directory: {BASE_DATA_DIR}")
    exit()

# The output path for the processed split.
# It's good practice to rename it clearly, e.g., adding a "_processed" suffix.
# You could also consider making a new directory for processed splits.
OUTPUT_PARQUET_PATH = os.path.join(BASE_DATA_DIR, f"{SPLIT_TO_PROCESS}_processed.parquet")

print(f"Found {len(parquet_files)} files for the '{SPLIT_TO_PROCESS}' split.")
# print(f"Files to be loaded: {parquet_files}") # Uncomment to verify file list

# --- Step 2: Load the dataset for the specific split ---
dataset = None
try:
    print(f"Attempting to load {SPLIT_TO_PROCESS} split from: {BASE_DATA_DIR}")
    # Pass the list of parquet files for the specific split
    dataset = load_dataset("parquet", data_files=parquet_files, split="train") # 'split="train"' here is a convention, can be omitted or set to None
    print(f"Successfully loaded {SPLIT_TO_PROCESS} split, containing {len(dataset)} samples.")

except Exception as e:
    print(f"Failed to load Parquet dataset for {SPLIT_TO_PROCESS} split. Error: {e}")
    exit()

print("Dataset features (columns):", dataset.features)
if TEXT_COLUMN_NAME not in dataset.features:
    print(f"Warning: The specified text column '{TEXT_COLUMN_NAME}' was not found in the dataset.")
    print("Available columns:", list(dataset.features.keys()))
    print("Please update TEXT_COLUMN_NAME if necessary.")
    # Exit or handle as per your requirement if the column is critical

# --- Step 3: Define the prefix adding function ---
def add_image_prefix_to_text(example):
    """
    Adds the '<image>\n' prefix to the specified text column in a dataset example.
    """
    if TEXT_COLUMN_NAME in example and example[TEXT_COLUMN_NAME] is not None:
        example[TEXT_COLUMN_NAME] = '<image>\n' + str(example[TEXT_COLUMN_NAME])
    return example

# --- Step 4: Apply the modification using map function ---
print(f"\nStarting to process the '{SPLIT_TO_PROCESS}' dataset: adding '<image>\\n' prefix to '{TEXT_COLUMN_NAME}' column...")
# Use num_proc for parallel processing to speed things up
processed_dataset = dataset.map(add_image_prefix_to_text, num_proc=os.cpu_count())
print(f"Processing of '{SPLIT_TO_PROCESS}' dataset complete.")

# --- Step 5: Verify the modification ---
if TEXT_COLUMN_NAME in processed_dataset.features and len(processed_dataset) > 0:
    print(f"\nFirst sample of the processed '{SPLIT_TO_PROCESS}' dataset (text column):")
    print(processed_dataset[0][TEXT_COLUMN_NAME])
else:
    print(f"Could not verify the modification for '{SPLIT_TO_PROCESS}' dataset. Column '{TEXT_COLUMN_NAME}' not found or dataset is empty.")


# --- Step 6: Safely replace the original files for this split ---
# Define a temporary output path for this processed split
TEMP_OUTPUT_PARQUET_PATH = OUTPUT_PARQUET_PATH + ".temp"

try:
    print(f"\nSaving processed '{SPLIT_TO_PROCESS}' dataset to temporary file: {TEMP_OUTPUT_PARQUET_PATH}")
    # When saving a split that was originally multiple files, it will be saved as one file.
    # If you want to keep the multi-file structure, you'd need to manually re-shard the dataset
    # before saving, which is more complex.
    processed_dataset.to_parquet(TEMP_OUTPUT_PARQUET_PATH)
    print(f"Processed dataset successfully saved to temporary file.")

    # --- Replace original files with the new processed file ---
    print(f"Removing original '{SPLIT_TO_PROCESS}' parquet files...")
    for f in parquet_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed: {f}")
        else:
            print(f"  Warning: Original file {f} not found, skipping removal.")

    print(f"Renaming temporary processed file to final output path: {OUTPUT_PARQUET_PATH}")
    os.rename(TEMP_OUTPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH)
    print(f"'{SPLIT_TO_PROCESS}' dataset successfully updated to the latest version at: {OUTPUT_PARQUET_PATH}")

except Exception as e:
    print(f"\nDataset update for '{SPLIT_TO_PROCESS}' split failed. Error: {e}")
    # Clean up any leftover temporary file if an error occurred during replacement
    if os.path.exists(TEMP_OUTPUT_PARQUET_PATH):
        print(f"Cleaning up temporary file: {TEMP_OUTPUT_PARQUET_PATH}")
        os.remove(TEMP_OUTPUT_PARQUET_PATH)
    print("Please check the error message and manually restore your original data if you have backups.")