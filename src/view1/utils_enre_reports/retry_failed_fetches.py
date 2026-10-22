"""
Re-run `git fetch origin <sha>` only for failures listed in processing_errors_cpp.log
(produced by process_dataset.py).
"""
import argparse
import os
import re
import subprocess
import sys
import time

from tqdm import tqdm

# Matches process_dataset.py log lines like:
# Failed to fetch commit <sha> for repo <name>. Error: ...
_FETCH_FAIL_RE = re.compile(
    r"Failed to fetch commit (?P<sha>[0-9a-fA-F]{7,40}) for repo (?P<repo>[^\s.]+)\.",
)


def parse_error_log(log_path):
    """Unique (repo, sha_lower) preserving first-seen order."""
    seen = set()
    pairs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _FETCH_FAIL_RE.search(line)
            if not m:
                continue
            sha = m.group("sha").lower()
            repo = m.group("repo")
            key = (repo, sha)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((repo, sha))
    return pairs


def retry_fetches(items, base_repo_path, retries, backoff_sec, outfile):
    remaining = []

    def log_fail(repo, sha, stderr_text):
        line = f"Failed to fetch commit {sha} for repo {repo}. Error: {stderr_text.strip()}"
        outfile.write(line + "\n")

    total = len(items)
    stderr_last = ""

    for repo, sha in tqdm(items, desc="Retry failed fetches", unit="fetch"):
        repo_path = os.path.join(base_repo_path, repo)
        if not os.path.isdir(repo_path):
            tqdm.write(f"Skipping {repo}/{sha}: directory not found: {repo_path}")
            log_fail(repo, sha, f"repository directory not found: {repo_path}")
            remaining.append((repo, sha))
            continue

        succeeded = False
        for attempt in range(1, retries + 1):
            proc = subprocess.run(
                ["git", "fetch", "origin", sha],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                tqdm.write(f"OK {repo} {sha[:12]}… (attempt {attempt})")
                succeeded = True
                break

            stderr_last = proc.stderr or proc.stdout or ""
            tqdm.write(
                f"Fail {repo} {sha[:12]}… attempt {attempt}/{retries}: {stderr_last.strip()[:200]}"
            )
            if attempt < retries and backoff_sec > 0:
                time.sleep(backoff_sec)

        if not succeeded:
            log_fail(repo, sha, stderr_last or "(empty stderr)")
            remaining.append((repo, sha))

    return remaining


def parse_args():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    p = argparse.ArgumentParser(
        description="Retry git fetch only for commits listed in an error log from process_dataset.py."
    )
    p.add_argument(
        "--error_log",
        default=os.path.join(package_root, "outputs", "view1", "processing_errors_cpp.log"),
        help="processing_errors_cpp.log (or compatible) produced by process_dataset.",
    )
    p.add_argument(
        "--base_repo_path",
        default=os.path.join(package_root, "outputs", "view1", "repositories", "cpp"),
        help="Directory holding one folder per repo (same as process_dataset --base_repo_path).",
    )
    p.add_argument(
        "--out_log",
        default=os.path.join(package_root, "outputs", "view1", "processing_errors_cpp_retry_remainder.log"),
        help="Write any still-failing fetches here (same line format as the input error log).",
    )
    p.add_argument("--retries", type=int, default=5, help="Attempts per commit (default 3).")
    p.add_argument(
        "--backoff",
        type=float,
        default=10.0,
        help="Seconds to sleep between failed attempts for the same commit (default 10).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_log)), exist_ok=True)

    pairs = parse_error_log(args.error_log)
    if not pairs:
        print(f"No matching 'Failed to fetch commit …' lines in {args.error_log}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(pairs)} unique repo/commit pairs from {args.error_log}")

    with open(args.out_log, "w", encoding="utf-8") as out_f:
        remaining = retry_fetches(
            pairs,
            args.base_repo_path,
            retries=max(1, args.retries),
            backoff_sec=max(0.0, args.backoff),
            outfile=out_f,
        )

    if remaining:
        print(f"Done. {len(remaining)} fetch(es) still failed — see {args.out_log}")
        sys.exit(2)
    print(f"All {len(pairs)} retry fetch(es) succeeded.")
    sys.exit(0)


if __name__ == "__main__":
    main()
