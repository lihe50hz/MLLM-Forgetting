import json
import argparse
from pathlib import Path


def load_imagewikiqa_data(jsonl_path):
    """
    Load ImageWikiQA dataset from JSONL file.
    
    Args:
        jsonl_path (str): Path to the imagewikiqa.jsonl file
    
    Returns:
        list: List of dictionaries with image paths, questions, and answers
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data


def load_generated_results(jsonl_path):
    """
    Load generated results from JSONL file.
    
    Args:
        jsonl_path (str): Path to the generated results JSONL file
    
    Returns:
        list: List of dictionaries with predictions
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line as JSON: {line[:50]}...")
                continue
    return data


def compute_classname_misclassification(imagewikiqa_data, generated_results):
    """
    Check how many model predictions incorrectly chose the class name as the answer.
    
    Args:
        imagewikiqa_data: List of ImageWikiQA dataset entries
        generated_results: List of generated prediction entries
        
    Returns:
        tuple: (classname_misclassifications, total_samples, percentage)
    """
    classname_misclassifications = 0
    total_samples = 0
    correct_samples = 0
    for result_entry in generated_results:
        try:
            # Get the model prediction
            prompt = result_entry.get('prompt', '').strip().upper()
            choices = prompt.split("\n")[-6:-2]
            choices[-1] = choices[-1].replace('<|IM_END|>','')
            label = result_entry.get('label', '').strip().upper()

            for dataset_entry in imagewikiqa_data:
                # Get the class name from the dataset
                class_name = dataset_entry.get('classname', '').strip().upper()
                des_choices = dataset_entry.get('text', '').upper().split('\n')[1:-1]
                num_matches = 0
                # print(choices, des_choices)
                # import time
                # time.sleep(10)
                for choice, des_choice in zip(choices, des_choices):
                    if choice == des_choice:
                        num_matches += 1
                if num_matches == 3:
                    break


            if class_name in choices[0]:
                class_choice = 'A'
            elif class_name in choices[1]:
                class_choice = 'B'
            elif class_name in choices[2]:
                class_choice = 'C'
            elif class_name in choices[3]:
                class_choice = 'D'
            else:
                raise ValueError(f"Class name {class_name} not found in choices: {choices}")
            
            predict = str(result_entry.get('predict', '')).strip()
            if ":" in predict:
                predict = predict.split(":")[-1].strip()
            predict = predict.upper()
            
            # Clean up the prediction (same logic as result_matcher.py)
            
            
            # Check if the prediction matches the class name
            if predict == class_choice or predict.startswith(f"{class_choice}."):
                classname_misclassifications += 1

            if label == predict or predict.startswith(f"{label}."):
                correct_samples += 1
            
            total_samples += 1
            
        except Exception as e:
            print(f"Warning: Error processing entry {total_samples}: {e}")
            continue
    
    if total_samples == 0:
        print("No valid samples found.")
        return 0, 0, 0.0
    
    percentage = (classname_misclassifications / total_samples) * 100
    return classname_misclassifications, total_samples, percentage, correct_samples


def main():
    parser = argparse.ArgumentParser(description='Check how many predictions incorrectly chose class name as answer')
    parser.add_argument(
        '--imagewikiqa_jsonl', 
        type=str, 
        default="data/imagewikiqa.jsonl",
        help='Path to the ImageWikiQA JSONL file'
    )
    parser.add_argument(
        '--generated_results', 
        type=str, 
        required=True,
        help='Path to the generated results JSONL file'
    )
    args = parser.parse_args()
    
    # Check if files exist
    imagewikiqa_path = Path(args.imagewikiqa_jsonl)
    results_path = Path(args.generated_results)
    
    if not imagewikiqa_path.exists():
        print(f"Error: ImageWikiQA file {imagewikiqa_path} does not exist.")
        return
    
    if not results_path.exists():
        print(f"Error: Generated results file {results_path} does not exist.")
        return
    
    try:
        # Load the datasets
        print(f"Loading ImageWikiQA dataset from: {imagewikiqa_path}")
        imagewikiqa_data = load_imagewikiqa_data(imagewikiqa_path)
        print(f"Loaded {len(imagewikiqa_data)} ImageWikiQA samples")
        
        print(f"Loading generated results from: {results_path}")
        generated_results = load_generated_results(results_path)
        print(f"Loaded {len(generated_results)} generated results")
        
        # Check if lengths match
        if len(imagewikiqa_data) != len(generated_results):
            print(f"Warning: Dataset lengths don't match! ImageWikiQA: {len(imagewikiqa_data)}, Results: {len(generated_results)}")
            print("Using the minimum length for comparison...")
            min_length = min(len(imagewikiqa_data), len(generated_results))
            imagewikiqa_data = imagewikiqa_data[:min_length]
            generated_results = generated_results[:min_length]
        
        # Compute class name misclassifications
        misclassifications, total, percentage, correct_samples = compute_classname_misclassification(
            imagewikiqa_data, generated_results
        )
        
        print(f"\nResults:")
        print(f"Total samples analyzed: {total}")
        print(f"Predictions that incorrectly chose class name: {misclassifications}")
        print(f"Percentage of class name misclassifications: {percentage:.2f}%")
        print(f"Correct samples: {correct_samples}")
        print(f"Accuracy: {100*correct_samples/total:.2f}%")
        print(f"Percentage of class name misclassifications in wrong choices: {100*misclassifications/(total-correct_samples):.2f}%")
    except Exception as e:
        print(f"Error processing files: {e}")


if __name__ == "__main__":
    main()
