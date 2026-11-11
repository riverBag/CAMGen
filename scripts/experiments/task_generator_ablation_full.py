"""
Generate commit-message generation tasks for the ablation and full experiments.

The script fills one prompt template with:
- query diff from dataset/full/commits
- contexts from either the 1000-sample ablation dataset or the complete
  dataset/full/contexts collection

It writes one JSONL task file per language and an all.jsonl file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_DATASET_DIR = DEFAULT_ROOT / "dataset/full"
DEFAULT_ABLATION_DATASET_DIR = DEFAULT_ROOT / "dataset/ablation"
DEFAULT_PROMPT_DIR = DEFAULT_ROOT / "prompts"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs/tasks/ablation"
DEFAULT_FULL_OUTPUT_DIR = DEFAULT_ROOT / "outputs/tasks/full"
DEFAULT_LANGUAGES = ("cpp", "java", "python")


PLACEHOLDERS = {
    "query_diff": "{QUERY_DIFF}",
    "core_entities": "{CORE_ENTITIES}",
    "structural_dependency_context": "{STRUCTURAL_DEPENDENCY_CONTEXT}",
    "functional_semantic_cluster_context": "{FUNCTIONAL_SEMANTIC_CLUSTER_CONTEXT}",
    "similar_diff": "{SIMILAR_DIFF}",
    "similar_message": "{SIMILAR_MESSAGE}",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            records.append(item)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def common_id_from_original(item: Mapping[str, Any]) -> str:
    repo = item.get("repo")
    commit_sha = item.get("commit_sha")
    if not isinstance(repo, str) or not isinstance(commit_sha, str) or len(commit_sha) < 7:
        raise ValueError(f"Cannot build common_id from original record: {item}")
    return f"{repo}_{commit_sha[:7]}"


def load_original_by_exact_commit(original_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in load_jsonl(original_path):
        repo = item.get("repo")
        commit_sha = item.get("commit_sha")
        if not isinstance(repo, str) or not isinstance(commit_sha, str):
            continue
        key = (repo, commit_sha)
        if key in mapping:
            raise ValueError(f"Duplicate original data key {key} in {original_path}")
        mapping[key] = item
    return mapping


def load_by_common_id(path: Path) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in load_jsonl(path):
        common_id = item.get("common_id")
        if not isinstance(common_id, str):
            raise ValueError(f"Missing common_id in {path}: {item}")
        if common_id in mapping:
            raise ValueError(f"Duplicate common_id {common_id} in {path}")
        mapping[common_id] = item
    return mapping


def load_by_unique_string_field(path: Path, field: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in load_jsonl(path):
        value = item.get(field) or item.get(field.replace("_", "-"))
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing {field} in {path}: {item}")
        if value in mapping:
            raise ValueError(f"Duplicate {field}={value} in {path}")
        mapping[value] = item
    return mapping


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def load_core_entity_names(core_entities_path: Path) -> List[str]:
    names: List[str] = []
    for item in load_jsonl(core_entities_path):
        core_entity = item.get("core_entity")
        if not isinstance(core_entity, dict):
            continue
        name = core_entity.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return unique_preserve_order(names)


def load_functional_cluster_context(core_clusters_path: Path) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for item in load_jsonl(core_clusters_path):
        clusters.append(
            {
                "cluster_label": item.get("cluster_label"),
                "entities": item.get("entities") or [],
            }
        )
    return clusters


def replace_required(template: str, replacements: Mapping[str, str]) -> str:
    prompt = template
    for placeholder in PLACEHOLDERS.values():
        if placeholder in prompt and placeholder not in replacements:
            raise ValueError(f"Missing replacement for required placeholder {placeholder}")
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def needs_view2_context(prompt_template: str) -> bool:
    return any(placeholder in prompt_template for placeholder in (
        "{CORE_ENTITIES}", "{FUNCTIONAL_SEMANTIC_CLUSTER_CONTEXT}",
    ))


def build_task(
    *,
    prompt_template: str,
    prompt_name: str,
    language: str,
    sample: Mapping[str, Any],
    original_item: Mapping[str, Any],
    view1_item: Mapping[str, Any] | None,
    view2_dir: Path,
    view3_item: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    sample_id = sample.get("sample_id")
    common_id = sample.get("common_id")
    commit_sha = sample.get("commit_sha")
    if not isinstance(sample_id, str) or not isinstance(common_id, str) or not isinstance(commit_sha, str):
        raise ValueError(f"Bad sample row: {sample}")

    query_diff = original_item.get("diff")
    if not isinstance(query_diff, str) or not query_diff:
        raise ValueError(f"Missing query diff for {sample_id}")

    replacements: Dict[str, str] = {
        "{QUERY_DIFF}": query_diff,
    }

    if "{CORE_ENTITIES}" in prompt_template:
        core_entities_path = view2_dir / "core_entities.jsonl"
        names = load_core_entity_names(core_entities_path)
        if not names:
            raise ValueError(f"No core entity names found for {sample_id}: {core_entities_path}")
        replacements["{CORE_ENTITIES}"] = json_block(names)

    if "{STRUCTURAL_DEPENDENCY_CONTEXT}" in prompt_template:
        if view1_item is None:
            raise ValueError(f"Missing structural dependency context for {sample_id}")
        retrieved_code = view1_item.get("retrieved_code")
        if not retrieved_code:
            raise ValueError(f"Empty structural dependency context for {sample_id}")
        replacements["{STRUCTURAL_DEPENDENCY_CONTEXT}"] = json_block(retrieved_code)

    if "{FUNCTIONAL_SEMANTIC_CLUSTER_CONTEXT}" in prompt_template:
        core_clusters_path = view2_dir / "core_clusters_simplified.jsonl"
        clusters = load_functional_cluster_context(core_clusters_path)
        if not clusters:
            raise ValueError(f"Empty functional semantic cluster context for {sample_id}: {core_clusters_path}")
        replacements["{FUNCTIONAL_SEMANTIC_CLUSTER_CONTEXT}"] = json_block(clusters)

    if "{SIMILAR_DIFF}" in prompt_template or "{SIMILAR_MESSAGE}" in prompt_template:
        if view3_item is None:
            raise ValueError(f"Missing evolutionary exemplar context for {sample_id}")
        similar_diff = view3_item.get("retrieve_diff") or view3_item.get("retrieve-diff")
        similar_message = view3_item.get("retrieve_message") or view3_item.get("retrieve-message")
        if not isinstance(similar_diff, str) or not similar_diff:
            raise ValueError(f"Missing retrieve_diff for {sample_id}")
        if not isinstance(similar_message, str) or not similar_message:
            raise ValueError(f"Missing retrieve_message for {sample_id}")
        replacements["{SIMILAR_DIFF}"] = similar_diff
        replacements["{SIMILAR_MESSAGE}"] = similar_message

    user_prompt = replace_required(prompt_template, replacements)

    return {
        "task_id": sample_id,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "label": original_item.get("message"),
        "meta": {
            "language": language,
            "common_id": common_id,
            "owner": original_item.get("owner"),
            "repo": original_item.get("repo"),
            "commit_sha": commit_sha,
            "prompt_name": prompt_name,
            "view2_source_dir": sample.get("view2_source_dir"),
        },
    }


def create_tasks(
    *,
    prompt_path: Path,
    experiment_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    final_dataset_dir: Path = DEFAULT_FINAL_DATASET_DIR,
    ablation_dataset_dir: Path = DEFAULT_ABLATION_DATASET_DIR,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    overwrite: bool = False,
) -> Dict[str, int]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    target_dir = output_dir / experiment_name
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {target_dir}. Use --overwrite.")
    target_dir.mkdir(parents=True, exist_ok=True)

    all_tasks: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    seen_task_ids: set[str] = set()

    for language in languages:
        original_path = ablation_dataset_dir / f"commits/{language}.jsonl"
        view1_path = ablation_dataset_dir / f"contexts/view1/{language}.jsonl"
        view3_path = ablation_dataset_dir / f"contexts/view3/{language}.jsonl"
        view2_lang_dir = ablation_dataset_dir / f"contexts/view2/downloaded_repositories_{language}_common"

        originals = load_jsonl(original_path)
        view1_by_common_id = load_by_common_id(view1_path) if "{STRUCTURAL_DEPENDENCY_CONTEXT}" in prompt_template else {}
        view3_by_common_id = load_by_common_id(view3_path) if "{SIMILAR_DIFF}" in prompt_template or "{SIMILAR_MESSAGE}" in prompt_template else {}

        language_tasks: List[Dict[str, Any]] = []
        for original_item in originals:
            owner = original_item.get("owner")
            repo = original_item.get("repo")
            commit_sha = original_item.get("commit_sha")
            if not all(isinstance(value, str) and value for value in (owner, repo, commit_sha)):
                raise ValueError(f"Bad commit row in {original_path}: {original_item}")
            if len(commit_sha) < 8:
                raise ValueError(f"Commit SHA is too short in {original_path}: {commit_sha}")
            common_id = common_id_from_original(original_item)
            sample_id = f"{language}_{common_id}"
            if sample_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id in ablation dataset: {sample_id}")
            seen_task_ids.add(sample_id)
            view2_source_dir = f"{repo}_{commit_sha[:8]}"
            sample = {
                "sample_id": sample_id,
                "common_id": common_id,
                "owner": owner,
                "repo": repo,
                "commit_sha": commit_sha,
                "view2_source_dir": view2_source_dir,
            }

            view2_dir = view2_lang_dir / view2_source_dir
            if needs_view2_context(prompt_template) and not view2_dir.is_dir():
                raise FileNotFoundError(f"Missing view2 context directory for {sample.get('sample_id')}: {view2_dir}")

            task = build_task(
                prompt_template=prompt_template,
                prompt_name=prompt_path.name,
                language=language,
                sample=sample,
                original_item=original_item,
                view1_item=view1_by_common_id.get(common_id),
                view2_dir=view2_dir,
                view3_item=view3_by_common_id.get(common_id),
            )
            language_tasks.append(task)

        output_path = target_dir / f"{language}.jsonl"
        counts[language] = write_jsonl(output_path, language_tasks)
        all_tasks.extend(language_tasks)

    counts["all"] = write_jsonl(target_dir / "all.jsonl", all_tasks)

    manifest = {
        "experiment_name": experiment_name,
        "prompt_path": str(prompt_path),
        "output_dir": str(target_dir),
        "languages": list(languages),
        "counts": counts,
        "final_dataset_dir": str(final_dataset_dir),
        "ablation_dataset_dir": str(ablation_dataset_dir),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def create_full_dataset_tasks(
    *,
    prompt_path: Path,
    experiment_name: str,
    output_dir: Path = DEFAULT_FULL_OUTPUT_DIR,
    final_dataset_dir: Path = DEFAULT_FINAL_DATASET_DIR,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Generate tasks for every record in dataset/full/commits."""
    prompt_template = prompt_path.read_text(encoding="utf-8")
    target_dir = output_dir / experiment_name
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {target_dir}. Use --overwrite.")
    target_dir.mkdir(parents=True, exist_ok=True)

    all_output_path = target_dir / "all.jsonl"
    with all_output_path.open("w", encoding="utf-8"):
        pass

    counts: Dict[str, int] = {}
    seen_task_ids: set[str] = set()

    for language in languages:
        original_path = final_dataset_dir / f"commits/{language}.jsonl"
        view1_path = final_dataset_dir / f"contexts/view1/{language}.jsonl"
        view3_path = final_dataset_dir / f"contexts/view3/{language}.jsonl"
        view2_lang_dir = final_dataset_dir / f"contexts/view2/downloaded_repositories_{language}_common"

        originals = load_jsonl(original_path)
        view1_by_instance_id = (
            load_by_unique_string_field(view1_path, "instance_id")
            if "{STRUCTURAL_DEPENDENCY_CONTEXT}" in prompt_template
            else {}
        )
        view3_by_query_sha = (
            load_by_unique_string_field(view3_path, "query_sha")
            if "{SIMILAR_DIFF}" in prompt_template or "{SIMILAR_MESSAGE}" in prompt_template
            else {}
        )

        language_tasks: List[Dict[str, Any]] = []
        for original_item in originals:
            owner = original_item.get("owner")
            repo = original_item.get("repo")
            commit_sha = original_item.get("commit_sha")
            if not all(isinstance(value, str) and value for value in (owner, repo, commit_sha)):
                raise ValueError(f"Bad original row in {original_path}: {original_item}")
            if len(commit_sha) < 8:
                raise ValueError(f"Commit SHA is too short in {original_path}: {commit_sha}")

            instance_id = f"{owner}_{repo}_{commit_sha[:7]}"
            common_id = f"{repo}_{commit_sha[:7]}"
            view2_source_dir = f"{repo}_{commit_sha[:8]}"
            sample_id = f"{language}_{common_id}"
            if sample_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id in full dataset: {sample_id}")
            seen_task_ids.add(sample_id)

            view2_dir = view2_lang_dir / view2_source_dir
            if needs_view2_context(prompt_template) and not view2_dir.is_dir():
                raise FileNotFoundError(f"Missing view2 context for {sample_id}: {view2_dir}")

            sample = {
                "sample_id": sample_id,
                "common_id": common_id,
                "repo": repo,
                "commit_sha": commit_sha,
                "view2_source_dir": view2_source_dir,
            }
            task = build_task(
                prompt_template=prompt_template,
                prompt_name=prompt_path.name,
                language=language,
                sample=sample,
                original_item=original_item,
                view1_item=view1_by_instance_id.get(instance_id),
                view2_dir=view2_dir,
                view3_item=view3_by_query_sha.get(commit_sha),
            )
            language_tasks.append(task)

        output_path = target_dir / f"{language}.jsonl"
        counts[language] = write_jsonl(output_path, language_tasks)
        with all_output_path.open("a", encoding="utf-8") as f:
            for task in language_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    counts["all"] = sum(counts.values())
    manifest = {
        "dataset_scope": "full",
        "experiment_name": experiment_name,
        "prompt_path": str(prompt_path),
        "output_dir": str(target_dir),
        "languages": list(languages),
        "counts": counts,
        "final_dataset_dir": str(final_dataset_dir),
        "view1_source": str(final_dataset_dir / "contexts/view1"),
        "view1_variant": "original_unoptimized",
        "view2_source": str(final_dataset_dir / "contexts/view2"),
        "view3_source": str(final_dataset_dir / "contexts/view3"),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_DIR / "ablation_direct_prompt.txt",
        help="Prompt template path.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Output experiment name. Defaults to the prompt file stem without '_prompt'.",
    )
    parser.add_argument(
        "--dataset-scope",
        choices=("ablation", "full"),
        default="ablation",
        help="Generate the 1000-sample ablation tasks or all final-dataset tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root. Defaults to the scope-specific tasks directory.",
    )
    parser.add_argument("--final-dataset-dir", type=Path, default=DEFAULT_FINAL_DATASET_DIR)
    parser.add_argument("--ablation-dataset-dir", type=Path, default=DEFAULT_ABLATION_DATASET_DIR)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = args.prompt_path.stem.removesuffix("_prompt")
    if args.dataset_scope == "full":
        output_dir = args.output_dir or DEFAULT_FULL_OUTPUT_DIR
        counts = create_full_dataset_tasks(
            prompt_path=args.prompt_path,
            experiment_name=experiment_name,
            output_dir=output_dir,
            final_dataset_dir=args.final_dataset_dir,
            languages=args.languages,
            overwrite=args.overwrite,
        )
    else:
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        counts = create_tasks(
            prompt_path=args.prompt_path,
            experiment_name=experiment_name,
            output_dir=output_dir,
            final_dataset_dir=args.final_dataset_dir,
            ablation_dataset_dir=args.ablation_dataset_dir,
            languages=args.languages,
            overwrite=args.overwrite,
        )
    print("Task generation finished.")
    print(f"Prompt: {args.prompt_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset scope: {args.dataset_scope}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
