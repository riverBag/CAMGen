import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from tqdm import tqdm

_STALL_OUT_MARKER = "ENRE_OUTPUT_STALL_ABORTED"


_INSTANCE_ID_LINE = re.compile(r"^Instance ID:\s*(.+?)\s*$")


def parse_instance_ids_from_error_log(path):
    """从既有 error log 中解析所有 `Instance ID:` 行（在清空日志前读取）。"""
    ids = set()
    if not path or not os.path.isfile(path):
        return ids
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _INSTANCE_ID_LINE.match(line.rstrip("\n"))
                if not m:
                    continue
                iid = m.group(1).strip()
                if iid and iid != "N/A":
                    ids.add(iid)
    except OSError:
        pass
    return ids


def run_command(command, working_dir):
    """Run a command and return (success, output_or_error)."""
    try:
        result = subprocess.run(
            command,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Return code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        )
        return False, error_message
    except FileNotFoundError:
        return False, f"Command '{command[0]}' not found. Make sure it is in PATH."


def _reader_thread_enqueue_lines(pipe, out_queue, sentinel):
    try:
        for line in iter(pipe.readline, ""):
            out_queue.put(line)
    finally:
        try:
            pipe.close()
        except OSError:
            pass
        out_queue.put(sentinel)


def run_enre_cpp_subprocess(
    command,
    working_dir,
    stream_parse_paths=True,
    stream_all_output=False,
    parse_stall_timeout_sec=360,
):
    """
    Run ENRE-CPP java. Collect full stdout+stderr for failure logs; optionally print live.
    If parse_stall_timeout_sec > 0: kill ENRE when no line is read from combined output for
    that many seconds (detects “stuck” parse with no log progress).
    """
    stall_enabled = (
        parse_stall_timeout_sec is not None and int(parse_stall_timeout_sec) > 0
    )
    stall_sec = int(parse_stall_timeout_sec) if stall_enabled else 0
    any_stream = stream_parse_paths or stream_all_output

    cmd = list(command)
    if shutil.which("stdbuf") and cmd and cmd[0] == "java":
        cmd = ["stdbuf", "-oL", "-eL"] + cmd

    if not stall_enabled and not any_stream:
        return run_command(cmd, working_dir)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return False, f"Command '{cmd[0]}' not found. Make sure it is in PATH."

    if proc.stdout is None:
        return False, "Popen had no stdout pipe."

    sentinel = object()
    out_q: queue.Queue = queue.Queue()
    reader = threading.Thread(
        target=_reader_thread_enqueue_lines,
        args=(proc.stdout, out_q, sentinel),
        daemon=True,
    )
    reader.start()

    chunks = []
    last_line_time = time.monotonic()

    while True:
        if stall_enabled and (time.monotonic() - last_line_time > stall_sec):
            proc.kill()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            partial = "".join(chunks)
            msg = (
                f"{_STALL_OUT_MARKER}\n"
                f"Aborted ENRE after {stall_sec}s with no stdout/stderr "
                "(no line read from JVM; likely stuck parsing).\nPartial output:\n"
                f"{partial}"
            )
            reader.join(timeout=5)
            return False, msg

        get_timeout = 1.0 if stall_enabled else None
        try:
            if get_timeout is not None:
                line = out_q.get(timeout=get_timeout)
            else:
                line = out_q.get()
        except queue.Empty:
            continue

        if line is sentinel:
            break

        last_line_time = time.monotonic()
        chunks.append(line)
        s = line.rstrip("\n\r")
        if stream_all_output:
            tqdm.write(s)
        elif stream_parse_paths and "parse file path:" in s.lower():
            tqdm.write(s)

    ret = proc.wait(timeout=None)
    reader.join(timeout=5)
    out = "".join(chunks)
    if ret == 0:
        return True, out
    return False, f"Return code: {ret}\nSTDOUT:\n{out}"


def write_error_log(error_log_path, instance_id, task, error_message):
    """Append a formatted error message to the log file."""
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Error: {error_message}\n")
        f.write("-" * 40 + "\n")


def build_enre_cpp_command(
    enre_jar_path,
    repo_path,
    project_name,
    java_xmx,
    java_xms=None,
    program_environments=None,
    extra_dirs=None,
):
    """Build command following ENRE-CPP usage: directory projectName [-p ...] [-d ...]."""
    command = [
        "java",
        "-Dfile.encoding=UTF-8",
        f"-Xmx{java_xmx}",
    ]

    if java_xms:
        command.append(f"-Xms{java_xms}")

    command.extend([
        "-jar",
        enre_jar_path,
        repo_path,
        project_name,
    ])

    for env in (program_environments or []):
        command.extend(["-p", env])

    for d in (extra_dirs or []):
        command.extend(["-d", d])

    return command


def find_generated_json(tmp_dir):
    """Locate generated json report(s) in ENRE-CPP temporary output dir."""
    candidates = []
    for root, _, files in os.walk(tmp_dir):
        for name in files:
            if name.lower().endswith(".json"):
                full_path = os.path.join(root, name)
                candidates.append((os.path.getmtime(full_path), full_path))

    if not candidates:
        return None

    # Prefer the latest json when multiple files are generated.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def process_dataset(
    dataset_path,
    repo_base_dir,
    enre_jar_path,
    output_dir,
    error_log_path,
    java_xmx,
    java_xms=None,
    program_environments=None,
    extra_dirs=None,
    skip_instance_ids=None,
    enre_stream_parse_paths=True,
    enre_stream_all=False,
    enre_parse_stall_timeout_sec=360,
):
    """Process dataset entries and generate ENRE-CPP report for each commit."""
    repo_base_dir = os.path.abspath(repo_base_dir)
    enre_jar_path = os.path.abspath(enre_jar_path)
    output_dir = os.path.abspath(output_dir)
    error_log_path = os.path.abspath(error_log_path)
    program_environments = [os.path.abspath(path) for path in (program_environments or [])]
    extra_dirs = [os.path.abspath(path) for path in (extra_dirs or [])]
    skip_instance_ids = frozenset(skip_instance_ids) if skip_instance_ids else frozenset()
    os.makedirs(output_dir, exist_ok=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("ENRE-CPP Report Generation Error Log\n" + "=" * 40 + "\n")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        pbar = tqdm(lines, desc="Generating ENRE-CPP Reports", total=len(lines))
        for i, line in enumerate(pbar, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                tqdm.write(f"[{i}/{len(lines)}] step: parse JSON | FAILED (malformed line)")
                write_error_log(
                    error_log_path,
                    "N/A",
                    "JSON Parsing",
                    f"Malformed JSON line: {line.strip()}",
                )
                continue

            instance_id = f"{data['owner']}_{data['repo']}_{data['commit_sha'][:7]}"
            repo_name = data["repo"]
            commit_sha = data["commit_sha"]

            if instance_id in skip_instance_ids:
                pbar.set_postfix(blocklist=instance_id[:40], refresh=False)
                continue

            repo_path = os.path.join(repo_base_dir, repo_name)
            output_report_path = os.path.join(output_dir, f"{instance_id}_enre_report.json")

            if os.path.exists(output_report_path):
                pbar.set_postfix(skip=instance_id[:40], refresh=False)
                continue

            tqdm.write(
                f"Starting run for instance: {instance_id} (dataset row {i} of {len(lines)})"
            )
            tqdm.write(f"[{i}/{len(lines)}] {instance_id} | step: start (report missing)")

            if not os.path.isdir(repo_path):
                tqdm.write(
                    f"[{i}/{len(lines)}] {instance_id} | step: repo dir missing → skip"
                )
                write_error_log(
                    error_log_path,
                    instance_id,
                    "File Check",
                    f"Repository path not found at {repo_path}",
                )
                continue

            tqdm.write(
                f"[{i}/{len(lines)}] {instance_id} | step: git checkout {commit_sha[:12]}…"
            )
            checkout_command = ["git", "checkout", commit_sha, "--force"]
            success, output = run_command(checkout_command, working_dir=repo_path)
            if not success:
                # On Windows, checkout may return non-zero because of unlink warnings
                # even when HEAD already moved to the target commit.
                verify_success, verify_output = run_command(["git", "rev-parse", "HEAD"], working_dir=repo_path)
                if verify_success and verify_output.strip() == commit_sha:
                    pass
                else:
                    write_error_log(error_log_path, instance_id, "Git Checkout", output)
                    tqdm.write(f"[{i}/{len(lines)}] {instance_id} | step: git checkout → ERROR (see log)")
                    continue

            verify_command = ["git", "rev-parse", "HEAD"]
            verify_success, verify_output = run_command(verify_command, working_dir=repo_path)
            if not verify_success:
                tqdm.write(
                    f"[{i}/{len(lines)}] {instance_id} | step: verify HEAD → ERROR (see log)"
                )
                write_error_log(
                    error_log_path,
                    instance_id,
                    "Git Checkout Verification",
                    f"Failed to run git rev-parse HEAD: {verify_output}",
                )
                continue

            current_commit = verify_output.strip()
            if current_commit != commit_sha:
                tqdm.write(
                    f"[{i}/{len(lines)}] {instance_id} | step: verify HEAD mismatch → ERROR (see log)"
                )
                write_error_log(
                    error_log_path,
                    instance_id,
                    "Git Checkout Verification",
                    (
                        f"Expected {commit_sha[:7]}, but HEAD is at {current_commit[:7]}."
                    ),
                )
                continue

            tqdm.write(
                f"[{i}/{len(lines)}] {instance_id} | step: checkout OK → ENRE-CPP java starting (may take a long time)"
            )

            tmp_output_dir = os.path.join(output_dir, f".tmp_enre_cpp_{instance_id}")
            os.makedirs(tmp_output_dir, exist_ok=True)

            command = build_enre_cpp_command(
                enre_jar_path=enre_jar_path,
                repo_path=repo_path,
                project_name=repo_name,
                java_xmx=java_xmx,
                java_xms=java_xms,
                program_environments=program_environments,
                extra_dirs=extra_dirs,
            )

            success, output = run_enre_cpp_subprocess(
                command,
                tmp_output_dir,
                stream_parse_paths=enre_stream_parse_paths,
                stream_all_output=enre_stream_all,
                parse_stall_timeout_sec=enre_parse_stall_timeout_sec,
            )
            if not success:
                is_stall = _STALL_OUT_MARKER in output
                if is_stall:
                    tqdm.write(
                        f"[{i}/{len(lines)}] {instance_id} | step: ENRE-CPP → skipped (stdout stall "
                        f">{enre_parse_stall_timeout_sec}s, see error log)"
                    )
                    write_error_log(
                        error_log_path,
                        instance_id,
                        "ENRE-CPP Parse Stall (no stdout line)",
                        output,
                    )
                else:
                    tqdm.write(
                        f"[{i}/{len(lines)}] {instance_id} | step: ENRE-CPP → FAILED (see log)"
                    )
                    write_error_log(error_log_path, instance_id, "ENRE-CPP Execution", output)
                shutil.rmtree(tmp_output_dir, ignore_errors=True)
                continue

            generated_json = find_generated_json(tmp_output_dir)
            if not generated_json:
                tqdm.write(
                    f"[{i}/{len(lines)}] {instance_id} | step: ENRE-CPP → no JSON output (see log)"
                )
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-CPP Post-Verification",
                    "Command succeeded, but no JSON report was found in temporary output directory.",
                )
                continue

            shutil.move(generated_json, output_report_path)
            shutil.rmtree(tmp_output_dir, ignore_errors=True)
            tqdm.write(
                f"[{i}/{len(lines)}] {instance_id} | step: done → {os.path.basename(output_report_path)}"
            )

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ENRE-CPP reports for dataset entries."
    )
    parser.add_argument(
        "--dataset_file",
        required=True,
        help="Path to the input JSONL dataset file.",
    )
    parser.add_argument(
        "--repo_dir",
        required=True,
        help="Base directory containing cloned git repositories.",
    )
    parser.add_argument(
        "--enre_jar",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ENRE-tools", "ENRE-cpp", "enre_cpp.jar")),
        help="Path to ENRE-CPP jar file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated reports will be saved.",
    )
    parser.add_argument(
        "--error_log",
        default=None,
        help="Path of error log file.",
    )
    parser.add_argument(
        "--java_xmx",
        default="8G",
        help="Java max heap, e.g. 4G or 64g.",
    )
    parser.add_argument(
        "--java_xms",
        default=None,
        help="Java initial heap, e.g. 8G (optional).",
    )
    parser.add_argument(
        "-p",
        "--program_environment",
        action="append",
        default=[],
        help="Program environment path, can be passed multiple times.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="extra_dirs",
        action="append",
        default=[],
        help="Additional directory for analysis, can be passed multiple times.",
    )
    parser.add_argument(
        "--skip_from_error_log",
        default=None,
        metavar="PATH",
        help="Skip Instance IDs from an explicitly selected error log, read before processing starts.",
    )
    parser.add_argument(
        "--skip_instance_id",
        action="append",
        default=[],
        metavar="INSTANCE_ID",
        help="Instance ID to skip; repeat this option to select multiple IDs.",
    )
    parser.add_argument(
        "--no_enre_stream_parse_paths",
        action="store_true",
        help="不在终端实时打印 ENRE 的 “Parse file path:” 行（仍完整写入失败时的 error log）。",
    )
    parser.add_argument(
        "--enre_stream_all",
        action="store_true",
        help="把 ENRE Java 的全部标准输出/错误都实时打到终端（刷屏，用于深度排查）。",
    )
    parser.add_argument(
        "--enre_parse_stall_timeout_sec",
        type=int,
        default=360,
        help=(
            "Kill ENRE and skip row if JVM writes no stdout/stderr line for this many seconds "
            "(0 disables)."
        ),
    )

    args = parser.parse_args()

    if args.error_log is None:
        args.error_log = os.path.join(args.output_dir, "enre_cpp_generator_errors.log")

    if not os.path.isfile(args.enre_jar):
        print(f"Error: ENRE-CPP jar not found at {args.enre_jar}")
        return

    skip_ids = set()
    if args.skip_from_error_log is not None:
        from_log_skip = parse_instance_ids_from_error_log(args.skip_from_error_log)
        skip_ids.update(from_log_skip)
        print(f"Skip list: +{len(from_log_skip)} instance_id(s) from {args.skip_from_error_log}")
    for s in args.skip_instance_id:
        if s and s.strip():
            skip_ids.add(s.strip())
    print(f"Skip list: {len(skip_ids)} explicitly selected instance_id(s).")

    process_dataset(
        dataset_path=args.dataset_file,
        repo_base_dir=args.repo_dir,
        enre_jar_path=args.enre_jar,
        output_dir=args.output_dir,
        error_log_path=args.error_log,
        java_xmx=args.java_xmx,
        java_xms=args.java_xms,
        program_environments=args.program_environment,
        extra_dirs=args.extra_dirs,
        skip_instance_ids=skip_ids,
        enre_stream_parse_paths=not args.no_enre_stream_parse_paths,
        enre_stream_all=args.enre_stream_all,
        enre_parse_stall_timeout_sec=args.enre_parse_stall_timeout_sec,
    )


if __name__ == "__main__":
    main()
