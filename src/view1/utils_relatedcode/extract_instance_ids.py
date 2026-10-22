import json
import argparse
import os
import sys

def process_dataset(input_file, output_file):
    """
    Reads a JSONL dataset, constructs an instance_id for each entry,
    and writes a new JSONL file with only instance_id and commit_sha.
    """
    print(f"Processing input file: {input_file}")
    print(f"Writing to output file: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    skipped = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_number, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_number}: {e}")
                    skipped += 1
                    continue

                owner = data.get('owner')
                repo = data.get('repo')
                commit_sha = data.get('commit_sha')

                if owner and repo and commit_sha:
                    # Construct instance_id as requested: owner_repo_commit_sha[:7]
                    instance_id = f"{owner}_{repo}_{commit_sha[:7]}"
                    
                    output_obj = {
                        "instance_id": instance_id,
                        "commit_sha": commit_sha
                    }
                    
                    f_out.write(json.dumps(output_obj) + '\n')
                    count += 1
                else:
                    print(f"Warning: Missing owner, repo, or commit_sha at line {line_number}. Skipping.")
                    skipped += 1
        
        print(f"\nProcessing complete.")
        print(f"Successfully processed: {count} items")
        print(f"Skipped items: {skipped}")
        print(f"Results saved to: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract instance_id and commit_sha from a JSONL dataset.")
    parser.add_argument("input_file", help="Path to the input JSONL file containing the dataset.")
    parser.add_argument("output_file", help="Path to the output JSONL file to be created.")
    
    args = parser.parse_args()

    process_dataset(args.input_file, args.output_file)
