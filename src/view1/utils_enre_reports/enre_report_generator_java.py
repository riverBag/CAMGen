import os
import json
import subprocess
import argparse
from tqdm import tqdm

def run_command(command, working_dir):
    """Runs a command and returns a tuple of (success, output)."""
    # print(f"Running command: '{' '.join(command)}' in '{working_dir}'")
    try:
        result = subprocess.run(
            command,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Return code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        # print(f"Error running command: {' '.join(command)}\n{error_message}")
        return False, error_message,e.stderr
    except FileNotFoundError:
        error_message = f"Command '{command[0]}' not found. Make sure it's in your PATH."
        # print(f"Error: {error_message}")
        return False, error_message

def log_message(log_path, message):
    """Writes a message to the log file and prints it to the console."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + "\n")
    tqdm.write(message)

def write_error_log(error_log_path, instance_id, task, error_message):
    """Appends a formatted error message to the log file."""
    msg = f"Instance ID: {instance_id}\nTask: {task}\nError: {error_message}\n" + ("-" * 40)
    log_message(error_log_path, msg)

def process_dataset(dataset_path, repo_base_dir, enre_jar_path, output_dir, error_log_path, start_index=0, batch_size=None, repo_filter=None):
    """
    Processes the dataset to generate ENRE reports for each entry.
    """
    repo_base_dir = os.path.abspath(repo_base_dir)
    enre_jar_path = os.path.abspath(enre_jar_path)
    output_dir = os.path.abspath(output_dir)
    error_log_path = os.path.abspath(error_log_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine mode for log file
    mode = 'a' # Always append to be safe with parallel processes
    
    # If it's a new file and we are not appending to existing (and not parallel writing to same file risk? 
    # Actually multiple processes writing to same log file might interleave lines, but it's better than overwriting.
    # Ideally each repo filter has its own log, or we accept interleaved logs.
    
    if not os.path.exists(error_log_path):
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write("ENRE Report Generation Error Log\n" + "="*40 + "\n")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        if batch_size is not None:
            end_index = min(start_index + batch_size, total_lines)
            lines_to_process = lines[start_index:end_index]
            process_range_desc = f"lines {start_index} to {end_index}"
        else:
            end_index = total_lines
            lines_to_process = lines[start_index:]
            process_range_desc = f"lines {start_index} to end"
            
        print(f"Processing {process_range_desc} (Total lines in file: {total_lines})")
        if repo_filter:
            print(f"Filtering for repository: {repo_filter}")
            # Pre-filter lines to show correct progress bar
            filtered_lines = []
            for line in lines_to_process:
                try:
                    data = json.loads(line)
                    if 'subset_entry' in data:
                        entry = data['subset_entry']
                        repo_name = entry.get('repo')
                    else:
                        repo_name = data.get('repo')
                    
                    if repo_name == repo_filter:
                        filtered_lines.append(line)
                except:
                    continue # Skip malformed lines during pre-filter
            lines_to_process = filtered_lines

        for line in tqdm(lines_to_process, desc=f"Generating Reports ({repo_filter if repo_filter else 'All'})"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                write_error_log(error_log_path, "N/A", "JSON Parsing", f"Malformed JSON line: {line.strip()}")
                continue
            
            # Extract basic info, with fallback for nested 'subset_entry'
            if 'subset_entry' in data:
                entry = data['subset_entry']
                owner = entry.get('owner')
                repo_name = entry.get('repo')
                commit_sha = entry.get('commit_sha')
            else:
                owner = data.get('owner')
                repo_name = data.get('repo')
                commit_sha = data.get('commit_sha')
            
            # Filter by repo name if specified (Double check, though we pre-filtered)
            if repo_filter and repo_name != repo_filter:
                continue
            
            if not owner or not repo_name or not commit_sha:
                write_error_log(error_log_path, "Unknown", "Data Extraction", f"Missing required fields in line: {line.strip()[:100]}...")
                continue

            instance_id = f"{owner}_{repo_name}_{commit_sha[:7]}"
            
            repo_path = os.path.join(repo_base_dir, repo_name)
            output_report_path = os.path.join(output_dir, f"{instance_id}_enre_report.json")

            if os.path.exists(output_report_path):
                continue

            if not os.path.isdir(repo_path):
                error_msg = f"Repository path not found at {repo_path}"
                write_error_log(error_log_path, instance_id, "File Check", error_msg)
                continue

            checkout_command = ['git', 'checkout', commit_sha, '--force']
            success, error = run_command(checkout_command, working_dir=repo_path)
            if not success:
                write_error_log(error_log_path, instance_id, "Git Checkout", error)
                continue

            # Verify checkout by comparing commit hashes
            verify_command = ['git', 'rev-parse', 'HEAD']
            verify_success, verify_output = run_command(verify_command, working_dir=repo_path)
            
            if not verify_success:
                # This happens if 'git rev-parse HEAD' fails
                error_msg = f"Failed to verify checkout for commit {commit_sha[:7]}. `git rev-parse HEAD` failed. Error: {verify_output}"
                write_error_log(error_log_path, instance_id, "Git Checkout Verification", error_msg)
                continue

            current_commit = verify_output.strip()
            if current_commit == commit_sha:
                log_message(error_log_path, f"Successfully checked out commit {commit_sha[:7]} in {repo_name}.")
            else:
                # This happens if checkout resulted in a different commit
                error_msg = f"Failed to checkout commit {commit_sha[:7]}. Expected {commit_sha[:7]}, but HEAD is at {current_commit[:7]}."
                write_error_log(error_log_path, instance_id, "Git Checkout Verification", error_msg)
                continue

            # Per user's instruction, the CWD will be the final output directory.
            # The -o parameter will be just the filename, and the source path will be absolute.
            output_filename = f"{instance_id}_enre_report"

            enre_command = [
                'java',
                '-Dfile.encoding=UTF-8',
                '-Xmx16G',
                '-jar',
                enre_jar_path,
                'java',
                repo_path,          # Absolute path to the repo to be analyzed
                repo_name,
                '-o',
                output_filename     # Simple filename, as CWD will be output_dir
            ]

            log_message(error_log_path, f"Running ENRE command: {' '.join(enre_command)}")

            # Run the command from the output directory to ensure the report is created in the correct place.
            success, output = run_command(enre_command, working_dir=output_dir)

            if output:
                log_message(error_log_path, f"--- ENRE Command Output ---\n{output}\n---------------------------")

            if not success:
                write_error_log(error_log_path, instance_id, "ENRE Execution", output)
                continue

            # Check where the file was actually created
            # ENRE-java 2.0+ creates a subdirectory named <project_name>-enre-out
            generated_file_path = os.path.join(
                output_dir, 
                f"{repo_name}-enre-out", 
                f"{output_filename}.json"
            )
            
            # If the file exists in the subdirectory, move it to the expected main output directory
            # if os.path.exists(generated_file_path):
            #     import shutil
            #     try:
            #         shutil.move(generated_file_path, output_report_path)
            #         # output_report_path is the flat path defined earlier
            #         log_message(error_log_path, f"Moved report to: {output_report_path}")
            #     except Exception as e:
            #         error_msg = f"Failed to move report file: {e}"
            #         write_error_log(error_log_path, instance_id, "File Move", error_msg)
            #         continue
            
            # After successful execution (and potential move), verify that the output file/directory exists
            # if not os.path.exists(output_report_path):
            #     error_msg = f"ENRE command finished with exit code 0, but output path was not found at expected location: {output_report_path}\n(Checked generated path: {generated_file_path})"
            #     # print(error_msg)
            #     write_error_log(error_log_path, instance_id, "ENRE Post-Verification", error_msg)
            #     continue
            # 新逻辑：只检查 enre-out 目录下的文件是否存在
            if not os.path.exists(generated_file_path):
                error_msg = f"ENRE command finished with exit code 0, but output path was not found at expected location: {generated_file_path}"
                write_error_log(error_log_path, instance_id, "ENRE Post-Verification", error_msg)
                continue

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    except Exception as e:
        write_error_log(error_log_path, "GLOBAL", "CRITICAL", f"An unexpected error occurred: {e}")
        # print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate ENRE reports for a dataset.")
    parser.add_argument(
        '--dataset_file',
        required=True,
        help='Path to the input JSONL dataset file.'
    )
    parser.add_argument(
        '--repo_dir',
        required=True,
        help='Base directory containing the cloned git repositories.'
    )
    parser.add_argument(
        '--enre_jar',
        required=True,
        help='Path to the ENRE java jar file.'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory to save the generated ENRE reports.'
    )
    parser.add_argument(
        '--error_log',
        default=None,
        help='Path to save the error log file (default: enre_generator_errors.log).'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Line number to start processing from (0-based).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Number of items to process in this run.'
    )
    parser.add_argument(
        '--repo_filter',
        type=str,
        default=None,
        help='Only process entries for this specific repository usage (e.g., "beam").'
    )
    args = parser.parse_args()

    if args.error_log is None:
        args.error_log = os.path.join(args.output_dir, "enre_java_generator_errors.log")

    try:
        import tqdm
    except ImportError:
        print("'tqdm' library not found. Please install it using: pip install tqdm")
        return

    process_dataset(args.dataset_file, args.repo_dir, args.enre_jar, args.output_dir, args.error_log, args.start_index, args.batch_size, args.repo_filter)

if __name__ == '__main__':
    main()
