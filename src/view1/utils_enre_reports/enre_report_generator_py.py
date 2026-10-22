import os
import json
import shutil
import subprocess
import argparse
import sys
from tqdm import tqdm


SKIP_REPOS = {"tvm"}


def run_command(command, working_dir, env=None, timeout=None):
    try:
        result = subprocess.run(
            command,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
        return True, result.stdout
    except subprocess.TimeoutExpired as e:
        return False, (
            f"Command timed out after {timeout} seconds\n"
            f"STDOUT: {e.stdout or ''}\nSTDERR: {e.stderr or ''}"
        )
    except subprocess.CalledProcessError as e:
        return False, f"Return code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    except FileNotFoundError:
        return False, f"Command '{command[0]}' not found. Make sure it is in PATH."


def write_error_log(error_log_path, instance_id, task, error_message):
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Error: {error_message}\n")
        f.write("-" * 40 + "\n")


def build_enre_py_command(
    repo_path,
    python_executable=sys.executable,
    enable_cfg=False,
    enable_cg=False,
    compatible=False,
    builtins=None,
):
    command = [python_executable, "-m", "enre", repo_path]

    if enable_cfg:
        command.append("--cfg")
    if enable_cg:
        command.append("--cg")
    if compatible:
        command.append("--compatible")
    if builtins:
        command.extend(["--builtins", builtins])

    return command


def get_script_default_enre_py_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", "..", "..", "ENRE-tools", "ENRE-py"))


def resolve_enre_py_dir(enre_py_dir):
    candidates = [
        enre_py_dir,
        os.path.join(enre_py_dir, "ENRE-py"),
        os.path.join(enre_py_dir, "ENRE-tools", "ENRE-py"),
    ]

    for candidate in candidates:
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(os.path.join(candidate, "enre", "__main__.py")):
            return candidate

    return None


def build_enre_env(enre_py_dir):
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = enre_py_dir
    return run_env


def verify_enre_import(enre_py_dir, python_executable=sys.executable):
    run_env = build_enre_env(enre_py_dir)
    command = [
        python_executable,
        "-c",
        "import enre; import os; print(os.path.dirname(enre.__file__))",
    ]
    return run_command(command, working_dir=enre_py_dir, env=run_env)


def load_generated_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_dataset(
    dataset_path,
    repo_base_dir,
    enre_py_dir,
    output_dir,
    error_log_path,
    enable_cfg=False,
    enable_cg=False,
    compatible=False,
    builtins=None,
    enre_timeout=None,
    start_line=None,
    end_line=None,
):
    output_dir = os.path.abspath(output_dir)
    repo_base_dir = os.path.abspath(repo_base_dir)
    enre_py_dir = os.path.abspath(enre_py_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("ENRE-py Report Generation Error Log\n" + "=" * 40 + "\n")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line is not None or end_line is not None:
            start_index = max((start_line or 1) - 1, 0)
            end_index = end_line if end_line is not None else len(lines)
            selected_lines = lines[start_index:end_index]
            desc = f"Generating ENRE-py Reports ({start_line or 1}-{end_line or len(lines)})"
        else:
            selected_lines = lines
            desc = "Generating ENRE-py Reports"

        for line in tqdm(selected_lines, desc=desc):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                write_error_log(error_log_path, "N/A", "JSON Parsing", f"Malformed JSON line: {line.strip()}")
                continue

            entry = data.get("subset_entry") if isinstance(data.get("subset_entry"), dict) else data
            repo_name = entry.get("repo")
            commit_sha = entry.get("commit_sha")
            owner = entry.get("owner")

            if not (repo_name and commit_sha and owner):
                write_error_log(
                    error_log_path,
                    "N/A",
                    "Missing Fields",
                    f"Missing repo/commit_sha/owner in subset_entry: {entry}",
                )
                continue

            instance_id = data.get("instance_id") or f"{owner}_{repo_name}_{commit_sha[:7]}"

            if repo_name in SKIP_REPOS:
                write_error_log(error_log_path, instance_id, "Repo Skip", f"Skipped repository '{repo_name}'.")
                continue

            repo_path = os.path.abspath(os.path.join(repo_base_dir, repo_name))
            output_report_path = os.path.join(output_dir, f"{instance_id}_enre_report.json")

            if os.path.exists(output_report_path):
                try:
                    if os.path.getsize(output_report_path) > 0:
                        continue
                    os.remove(output_report_path)
                    write_error_log(
                        error_log_path,
                        instance_id,
                        "Existing Empty Report",
                        f"Removed existing empty report: {output_report_path}",
                    )
                except Exception as e:
                    try:
                        os.remove(output_report_path)
                    except Exception:
                        pass
                    write_error_log(
                        error_log_path,
                        instance_id,
                        "Existing Report Check",
                        f"Removed unreadable existing report: {e}",
                    )

            if not os.path.isdir(repo_path):
                write_error_log(error_log_path, instance_id, "File Check", f"Repository path not found at {repo_path}")
                continue

            success, output = run_command(["git", "checkout", commit_sha, "--force"], working_dir=repo_path)
            if not success:
                write_error_log(error_log_path, instance_id, "Git Checkout", output)
                continue

            verify_success, verify_output = run_command(["git", "rev-parse", "HEAD"], working_dir=repo_path)
            if not verify_success:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "Git Checkout Verification",
                    f"Failed to run git rev-parse HEAD: {verify_output}",
                )
                continue

            current_commit = verify_output.strip()
            if current_commit != commit_sha:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "Git Checkout Verification",
                    f"Expected {commit_sha[:7]}, but HEAD is at {current_commit[:7]}.",
                )
                continue

            generated_report_name = f"{repo_name}-report-enre.json"
            generated_report_path = os.path.join(output_dir, generated_report_name)

            if os.path.exists(generated_report_path):
                os.remove(generated_report_path)

            command = build_enre_py_command(
                repo_path=repo_path,
                enable_cfg=enable_cfg,
                enable_cg=enable_cg,
                compatible=compatible,
                builtins=builtins,
            )
            run_env = build_enre_env(enre_py_dir)

            print(f"RUN: {' '.join(command)}")
            print(f"CWD: {output_dir}")
            print(f"PYTHONPATH: {run_env['PYTHONPATH']}")

            success, output = run_command(
                command,
                working_dir=output_dir,
                env=run_env,
                timeout=enre_timeout,
            )

            if not success:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-py Execution",
                    (
                        f"{output}\n"
                        f"Python executable: {command[0]}\n"
                        f"ENRE-py directory: {enre_py_dir}\n"
                        f"PYTHONPATH: {run_env.get('PYTHONPATH', '')}"
                    ),
                )
                continue

            if not os.path.exists(generated_report_path):
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-py Post-Verification",
                    f"Command succeeded, but expected output file was not found: {generated_report_path}",
                )
                continue

            try:
                generated_data = load_generated_report(generated_report_path)
            except Exception as e:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-py Report Check",
                    f"Failed to inspect generated report: {e}",
                )
                continue

            if not generated_data.get("variables") and not generated_data.get("cells"):
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-py Empty Report",
                    f"Generated empty report for repo_path={repo_path}",
                )
                os.remove(generated_report_path)
                continue

            shutil.move(generated_report_path, output_report_path)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate ENRE-py reports for dataset entries.")
    parser.add_argument("--dataset_file", required=True, help="Path to the input JSONL dataset file.")
    parser.add_argument("--repo_dir", required=True, help="Base directory containing cloned git repositories.")
    parser.add_argument("--enre_py_dir", default=get_script_default_enre_py_dir())
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--error_log", default=None)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--cg", action="store_true")
    parser.add_argument("--compatible", action="store_true")
    parser.add_argument("--builtins", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--start_line", type=int, default=None)
    parser.add_argument("--end_line", type=int, default=None)

    args = parser.parse_args()

    resolved_enre_py_dir = resolve_enre_py_dir(args.enre_py_dir)
    if not resolved_enre_py_dir:
        print(
            "Error: ENRE-py directory not found or invalid. Expected a directory "
            f"containing enre/__main__.py, got: {args.enre_py_dir}"
        )
        return

    import_success, import_output = verify_enre_import(resolved_enre_py_dir)
    if not import_success:
        print(
            "Error: ENRE-py is not importable with the current Python executable.\n"
            f"Python executable: {sys.executable}\n"
            f"ENRE-py directory: {resolved_enre_py_dir}\n"
            f"{import_output}"
        )
        return

    error_log_path = args.error_log
    if not error_log_path:
        error_log_path = os.path.join(args.output_dir, "enre_py_generator_errors.log")

    process_dataset(
        dataset_path=args.dataset_file,
        repo_base_dir=args.repo_dir,
        enre_py_dir=resolved_enre_py_dir,
        output_dir=args.output_dir,
        error_log_path=error_log_path,
        enable_cfg=args.cfg,
        enable_cg=args.cg,
        compatible=args.compatible,
        builtins=args.builtins,
        enre_timeout=args.timeout,
        start_line=args.start_line,
        end_line=args.end_line,
    )


if __name__ == "__main__":
    main()
