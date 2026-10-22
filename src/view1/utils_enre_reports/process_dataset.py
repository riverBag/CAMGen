import json
import os
import subprocess
import argparse
from tqdm import tqdm


def process_dataset(jsonl_file, base_repo_path, log_file, max_items=5):
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    # Get total number of lines for tqdm progress bar
    with open(jsonl_file, "r", encoding="utf-8-sig") as f:
        total_lines = sum(1 for _ in f)

    if max_items is not None:
        total_lines = min(total_lines, max_items)

    processed_count = 0

    with open(jsonl_file, "r", encoding="utf-8-sig") as f_in, open(log_file, "w", encoding="utf-8") as f_log:
        for line_no, line in enumerate(tqdm(f_in, total=total_lines, desc="Processing dataset"), start=1):
            if max_items is not None and processed_count >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Handle nested structure if present
                if 'subset_entry' in data:
                    entry = data['subset_entry']
                    repo = entry.get('repo')
                    commit_sha = entry.get('commit_sha')
                else:
                    repo = data.get('repo')
                    commit_sha = data.get('commit_sha')

                if not repo or not commit_sha:
                    msg = f"Skipping line due to missing 'repo' or 'commit_sha': {line}"
                    f_log.write(msg + "\n")
                    tqdm.write(msg)
                    continue

                repo_path = os.path.join(base_repo_path, repo)

                if not os.path.isdir(repo_path):
                    msg = f"Repository directory not found for repo '{repo}': {repo_path}"
                    f_log.write(msg + "\n")
                    tqdm.write(msg)
                    continue

                command = ['git', 'fetch', 'origin', commit_sha]
                result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)

                if result.returncode == 0:
                    tqdm.write(f"Successfully fetched commit {commit_sha} for repo {repo}")
                else:
                    error_message = result.stderr.strip()
                    msg = f"Failed to fetch commit {commit_sha} for repo {repo}. Error: {error_message}"
                    f_log.write(msg + "\n")
                    tqdm.write(msg)

            except json.JSONDecodeError as e:
                msg = f"Skipping line {line_no} due to JSON decoding error: {e}. Raw line: {line}"
                f_log.write(msg + "\n")
                tqdm.write(msg)
            except Exception as e:
                msg = f"An unexpected error occurred for line {line}: {e}"
                f_log.write(msg + "\n")
                tqdm.write(msg)
            finally:
                processed_count += 1

    print(f"Processed {processed_count} record(s).")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch commits for repositories listed in a JSONL file.")
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parser.add_argument(
        "--jsonl_file",
        default=os.path.join(
            package_root, "dataset", "full", "commits", "cpp.jsonl",
        ),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--base_repo_path",
        default=os.path.join(package_root, "outputs", "view1", "repositories", "cpp"),
        help="Directory containing cloned repositories (each repo is a subdirectory).",
    )
    parser.add_argument(
        "--log_file",
        default=os.path.join(package_root, "outputs", "view1", "processing_errors_cpp.log"),
        help="Path to error log file.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=5,
        help="Only process the first N records (default: 5). Use 0 or negative to process all.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_items = args.max_items if args.max_items > 0 else None
    process_dataset(args.jsonl_file, args.base_repo_path, args.log_file, max_items=max_items)
    print("Dataset processing complete. Check the log file for any issues.")
