import json
from datasets import load_dataset
import sys

def main():
    local_path = "deepscaler/data/train/deepscaler.json"
    hf_dataset_name = "agentica-org/DeepScaleR-Preview-Dataset"
    
    print(f"Loading local dataset from {local_path}...")
    try:
        with open(local_path, 'r') as f:
            local_data = json.load(f)
    except Exception as e:
        print(f"Error loading local file: {e}")
        return

    print(f"Loading HF dataset {hf_dataset_name}...")
    try:
        # Load the dataset. We'll assume it's the 'train' split if not specified, 
        # or load the whole thing if it's small.
        hf_dataset = load_dataset(hf_dataset_name)
        # Often datasets have a 'train' split. Let's see what's available.
        if 'train' in hf_dataset:
            hf_data = hf_dataset['train']
        else:
            # If no train split, take the first available split
            split_name = list(hf_dataset.keys())[0]
            hf_data = hf_dataset[split_name]
            print(f"Using split: {split_name}")
    except Exception as e:
        print(f"Error loading HF dataset: {e}")
        return

    print(f"Local count: {len(local_data)}")
    print(f"HF count: {len(hf_data)}")

    local_problems = [item['problem'] for item in local_data]
    hf_problems = [item['problem'] for item in hf_data]

    local_set = set(local_problems)
    hf_set = set(hf_problems)

    print(f"Unique local problems: {len(local_set)}")
    print(f"Unique HF problems: {len(hf_set)}")

    # Check if they are identical sets
    if local_set == hf_set:
        print("Success: Every problem matches perfectly!")
    else:
        missing_in_hf = local_set - hf_set
        missing_in_local = hf_set - local_set

        if missing_in_hf:
            print(f"Found {len(missing_in_hf)} problems in local file that are NOT in HF dataset.")
            # Print first 2 for debugging
            for p in list(missing_in_hf)[:2]:
                print(f"  Missing in HF: {p[:100]}...")
        
        if missing_in_local:
            print(f"Found {len(missing_in_local)} problems in HF dataset that are NOT in local file.")
            # Print first 2 for debugging
            for p in list(missing_in_local)[:2]:
                print(f"  Missing in local: {p[:100]}...")

    # Also check ordering if desired, but set comparison is usually what's meant by "every problems match"
    if len(local_data) == len(hf_data):
        mismatches = 0
        for i in range(len(local_data)):
            if local_data[i]['problem'] != hf_data[i]['problem']:
                mismatches += 1
        if mismatches == 0:
            print("The order of problems also matches perfectly.")
        else:
            print(f"Note: Order differs or content differs at {mismatches} positions.")

if __name__ == "__main__":
    main()
