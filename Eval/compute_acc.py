import json
import argparse
from pathlib import Path


def compute_accuracy(jsonl_file_path):
    """
    Compute accuracy by checking if 'predict' starts with 'label' for each entry.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        
    Returns:
        float: Mean accuracy as a percentage
    """
    correct_predictions = 0
    total_predictions = 0
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line as JSON: {line[:50]}...")
                continue
            
            if 'predict' in entry and 'label' in entry:
                predict = str(entry['predict']).strip()
                label = str(entry['label']).strip()
                
                if predict.startswith(label):
                    correct_predictions += 1
                total_predictions += 1
            else:
                print(f"Warning: Entry missing 'predict' or 'label' field: {entry}")
    
    if total_predictions == 0:
        print("No valid predictions found in the JSON file.")
        return 0.0
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Compute accuracy from JSONL predictions')
    parser.add_argument('--file', type=str, help='Path to the JSONL file containing predictions and labels')
    args = parser.parse_args()
    
    jsonl_file_path = Path(args.file)
    
    if not jsonl_file_path.exists():
        print(f"Error: File {jsonl_file_path} does not exist.")
        return
    
    try:
        accuracy = compute_accuracy(jsonl_file_path)
        
        # Count total lines in JSONL file
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        
        print(f"Total entries processed: {total_lines}")
        print(f"Mean Accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
