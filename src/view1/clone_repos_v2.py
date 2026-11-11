import argparse
import json
import os
import shutil
import subprocess


def get_view1_dir():
    return os.path.dirname(os.path.abspath(__file__))


def default_dataset_file(language):
    package_root = os.path.dirname(os.path.dirname(get_view1_dir()))
    return os.path.join(package_root, "dataset", "full", "commits", f"{language}.jsonl")


def default_target_dir(language):
    package_root = os.path.dirname(os.path.dirname(get_view1_dir()))
    return os.path.join(package_root, "outputs", "view1", "repositories", language)


def iter_dataset_entries(dataset_file):
    with open(dataset_file, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, f"JSON decode error: {exc}"
                continue

            entry = data.get("subset_entry") if isinstance(data.get("subset_entry"), dict) else data
            yield line_no, entry, None


def collect_repositories(dataset_files, limit=None):
    repos = []
    seen = set()
    skipped = 0

    for dataset_file in dataset_files:
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        print(f"Processing dataset: {dataset_file}")
        for line_no, entry, error in iter_dataset_entries(dataset_file):
            if error:
                skipped += 1
                print(f"Warning: {dataset_file}:{line_no}: {error}")
                continue

            git_url = entry.get("git_url") if isinstance(entry, dict) else None
            repo_name = entry.get("repo") if isinstance(entry, dict) else None
            owner = entry.get("owner") if isinstance(entry, dict) else None

            if not git_url or not repo_name:
                skipped += 1
                print(f"Warning: {dataset_file}:{line_no}: missing git_url or repo")
                continue

            repo_key = (git_url, repo_name)
            if repo_key in seen:
                continue

            seen.add(repo_key)
            repos.append(
                {
                    "git_url": git_url,
                    "repo": repo_name,
                    "owner": owner,
                    "source": dataset_file,
                    "line": line_no,
                }
            )

            if limit is not None and len(repos) >= limit:
                return repos, skipped

    return repos, skipped


def build_clone_command(git_url, repo_path, clone_mode, depth):
    command = ["git", "clone"]
    if clone_mode == "partial":
        command.extend(["--filter=blob:none"])
    elif clone_mode == "shallow":
        command.extend(["--depth", str(depth)])
    command.extend([git_url, repo_path])
    return command


def clone_repositories(
    dataset_files,
    target_dir,
    clone_mode="partial",
    depth=1,
    dry_run=False,
    reclone_incomplete=False,
    limit=None,
    fail_fast=False,
):
    os.makedirs(target_dir, exist_ok=True)
    repos, skipped = collect_repositories(dataset_files, limit=limit)

    print(f"Found {len(repos)} unique repos. Skipped malformed entries: {skipped}")
    print(f"Target directory: {target_dir}")
    print(f"Clone mode: {clone_mode}")

    cloned = 0
    existing = 0
    failed = 0
    incomplete = 0

    for repo in repos:
        repo_name = repo["repo"]
        git_url = repo["git_url"]
        repo_path = os.path.join(target_dir, repo_name)

        if os.path.isdir(os.path.join(repo_path, ".git")):
            existing += 1
            print(f"Skipping {repo_name}: already exists")
            continue

        if os.path.exists(repo_path):
            incomplete += 1
            if not reclone_incomplete:
                print(
                    f"Warning: {repo_path} exists but is not a git repo. "
                    "Use --reclone-incomplete to replace it."
                )
                continue
            print(f"Removing incomplete directory before clone: {repo_path}")
            if not dry_run:
                shutil.rmtree(repo_path)

        command = build_clone_command(git_url, repo_path, clone_mode, depth)
        print(f"Cloning {repo_name} from {git_url}")
        print("RUN:", " ".join(command))

        if dry_run:
            continue

        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            cloned += 1
        else:
            failed += 1
            print(f"Error: clone failed for {repo_name} with exit code {result.returncode}")
            if fail_fast:
                raise SystemExit(result.returncode)

    print(
        "Summary: "
        f"cloned={cloned}, existing={existing}, failed={failed}, "
        f"incomplete_skipped_or_recloned={incomplete}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone repositories listed in a view1 JSONL dataset."
    )
    parser.add_argument(
        "--language",
        choices=["python", "cpp", "java"],
        default="python",
        help="Language-specific default paths to use when dataset/target are omitted.",
    )
    parser.add_argument(
        "--dataset_file",
        action="append",
        default=None,
        help="Input JSONL dataset file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--target_dir",
        default=None,
        help="Directory where repositories will be cloned.",
    )
    parser.add_argument(
        "--clone_mode",
        choices=["partial", "full", "shallow"],
        default="partial",
        help=(
            "partial uses --filter=blob:none, full does a normal clone, "
            "shallow uses --depth. partial is recommended before fetching specific commits."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Depth used only when --clone_mode shallow.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Clone only the first N unique repositories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print clone commands without running them.",
    )
    parser.add_argument(
        "--reclone-incomplete",
        action="store_true",
        help="Replace an existing non-git directory at the target repo path.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first clone failure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_files = args.dataset_file or [default_dataset_file(args.language)]
    target_dir = args.target_dir or default_target_dir(args.language)
    clone_repositories(
        dataset_files=dataset_files,
        target_dir=target_dir,
        clone_mode=args.clone_mode,
        depth=args.depth,
        dry_run=args.dry_run,
        reclone_incomplete=args.reclone_incomplete,
        limit=args.limit,
        fail_fast=args.fail_fast,
    )
