import os
import re
import sys
import json
import argparse
import subprocess
from tqdm import tqdm

# Allow importing utils modules from both workspace root and utils_relatedcode.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, "utils_relatedcode")
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from cpp_utils.relation_analyzer import analyze_cpp_enre_report
from cpp_utils.code_retriever import retrieve_code_context


CPP_TYPE_TO_CATEGORY = {
    "function": "Function",
    "class": "Class",
    "struct": "Struct",
    "union": "Union",
    "template": "Template",
    "file": "File",
    "namespace": "Namespace",
}


def _normalize_rel_path(path: str | None) -> str:
    if not isinstance(path, str):
        return ""
    return path.replace("\\", "/").strip("/")


def _normalize_cpp_variables(raw_data) -> list[dict]:
    data = raw_data
    if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
        data = raw_data[0]

    if not isinstance(data, dict):
        return []

    vars_raw = data.get("variables", [])
    if not isinstance(vars_raw, list):
        return []

    normalized = []
    for var in vars_raw:
        if not isinstance(var, dict):
            continue
        entity_file = var.get("entityFile") or var.get("File")
        category = var.get("entityType") or var.get("category")
        parent_id = var.get("parentID") if var.get("parentID") is not None else var.get("parentId")
        normalized.append(
            {
                "id": var.get("id"),
                "qualifiedName": var.get("qualifiedName"),
                "category": category,
                "File": entity_file,
                "parentId": parent_id,
            }
        )
    return normalized


def _path_matches(entity_path: str | None, enre_file_path: str | None) -> bool:
    normalized_entity = _normalize_rel_path(entity_path)
    normalized_enre = _normalize_rel_path(enre_file_path)
    if not normalized_entity or not normalized_enre:
        return False
    if normalized_entity == normalized_enre:
        return True
    return normalized_enre.endswith("/" + normalized_entity)


def _strip_template_suffix(name: str) -> str:
    return re.sub(r"<.*>$", "", name)


def _extract_cpp_tail_name(qualified_name: str | None) -> str:
    if not qualified_name:
        return ""
    tail = re.split(r"::|\.", qualified_name)[-1]
    return _strip_template_suffix(tail)


def _extract_entity_name_tail(name: str | None) -> str:
    if not name:
        return ""
    tail = re.split(r"::|\.", str(name))[-1]
    return _strip_template_suffix(tail)


def _canonical_cpp_symbol_name(name: str | None) -> str:
    if not name:
        return ""
    token = _strip_template_suffix(str(name)).strip()
    token = token.lstrip("_")
    if token.startswith("arrow_"):
        token = token[len("arrow_") :]
    return token.lower()


def _function_name_matches_qn(target_name: str, qualified_name: str) -> bool:
    target_tail = _extract_entity_name_tail(target_name)
    qn_tail = _extract_cpp_tail_name(qualified_name)
    if not target_tail or not qn_tail:
        return False

    # Direct and common wrapper forms.
    if qn_tail in {target_tail, f"_arrow_{target_tail}", f"_{target_tail}"}:
        return True

    # Canonical match ignores _arrow_ and leading underscores.
    target_canon = _canonical_cpp_symbol_name(target_tail)
    qn_canon = _canonical_cpp_symbol_name(qn_tail)
    if target_canon and qn_canon == target_canon:
        return True

    # Allow wrapped tails like "compute__CallFunction" to match "CallFunction".
    return bool(target_canon and qn_canon.endswith(target_canon))


def _extract_scope_tokens(qualified_name: str | None) -> list[str]:
    if not qualified_name:
        return []
    tokens = [_strip_template_suffix(t) for t in re.split(r"::|\.", qualified_name) if t]
    return tokens


def _class_anchor_matches_qn(class_name: str | None, qualified_name: str) -> bool:
    if not class_name:
        return True
    anchor_tokens = _extract_scope_tokens(class_name)
    qn_tokens = _extract_scope_tokens(qualified_name)
    if not anchor_tokens or len(qn_tokens) < 2:
        return False
    # Exclude function tail token, only match in scope chain.
    scope_tokens = qn_tokens[:-1]
    scope_canon = {_canonical_cpp_symbol_name(t) for t in scope_tokens}
    anchor_canon = {_canonical_cpp_symbol_name(t) for t in anchor_tokens}
    return any(a and a in scope_canon for a in anchor_canon)


def generate_cpp_qualified_names(entity: dict, variables: list[dict]) -> tuple[list[str], str | None]:
    entity_type = str(entity.get("type", "")).lower()
    path = entity.get("path")
    name = entity.get("name")

    if entity_type not in CPP_TYPE_TO_CATEGORY:
        warning_msg = f"Unsupported entity type '{entity_type}'. Entity: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    if entity_type in {"function", "class", "struct", "union", "template", "namespace"} and not all([path, name]):
        warning_msg = f"Skipping entity due to missing 'path' or 'name'. Entity: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    expected_category = CPP_TYPE_TO_CATEGORY[entity_type]
    class_name = entity.get("class_name")

    # File entity: strict path-based matching.
    if entity_type == "file":
        file_candidates = []
        for var in variables:
            if var.get("category") != expected_category:
                continue
            if _path_matches(path, var.get("File")):
                qn = var.get("qualifiedName")
                if qn:
                    file_candidates.append(qn)
        file_candidates = list(dict.fromkeys(file_candidates))
        if len(file_candidates) == 1:
            return file_candidates, None
        if len(file_candidates) > 1:
            warning_msg = f"Ambiguous file mapping for entity (multiple exact matches): {entity}"
            print(f"Warning: {warning_msg}")
            return file_candidates, warning_msg
        warning_msg = f"Could not find file entity in ENRE report: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    # Function entity: strict file+name matching, optional class_name disambiguation.
    if entity_type == "function":
        target_tail_name = _extract_entity_name_tail(name)
        if not target_tail_name:
            warning_msg = f"Could not infer function tail name from entity: {entity}"
            print(f"Warning: {warning_msg}")
            return [], warning_msg

        anchor_matched = []
        tail_and_path_matched = []
        for var in variables:
            if var.get("category") != expected_category:
                continue
            if not _path_matches(path, var.get("File")):
                continue
            qn = var.get("qualifiedName")
            if not qn:
                continue
            if not _function_name_matches_qn(target_tail_name, qn):
                continue

            tail_and_path_matched.append(qn)
            if _class_anchor_matches_qn(class_name, qn):
                anchor_matched.append(qn)

        # Prefer candidates satisfying class anchor (e.g. FileMetaData).
        preferred = list(dict.fromkeys(anchor_matched)) if class_name else []
        fallback = list(dict.fromkeys(tail_and_path_matched))

        if class_name and len(preferred) == 1:
            return preferred, None
        if class_name and len(preferred) > 1:
            warning_msg = f"Ambiguous function mapping for entity (overloads or duplicates): {entity}"
            print(f"Warning: {warning_msg}")
            return preferred, warning_msg
        if len(fallback) == 1:
            return fallback, None
        if len(fallback) > 1:
            warning_msg = f"Ambiguous function mapping for entity (tail+path matched multiple candidates): {entity}"
            print(f"Warning: {warning_msg}")
            return fallback, warning_msg
        warning_msg = f"Could not find function entity in ENRE report: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    # Class-like / namespace entity: strict file+simple-name matching.
    target_tail_name = _extract_entity_name_tail(name)
    type_candidates = []
    for var in variables:
        if var.get("category") != expected_category:
            continue
        if not _path_matches(path, var.get("File")):
            continue
        qn = var.get("qualifiedName")
        if not qn:
            continue
        if _extract_cpp_tail_name(qn) == target_tail_name:
            type_candidates.append(qn)

    type_candidates = list(dict.fromkeys(type_candidates))
    if len(type_candidates) == 1:
        return type_candidates, None
    if len(type_candidates) > 1:
        warning_msg = f"Ambiguous {entity_type} mapping for entity (multiple exact matches): {entity}"
        print(f"Warning: {warning_msg}")
        return type_candidates, warning_msg
    warning_msg = f"Could not find {entity_type} entity in ENRE report: {entity}"
    print(f"Warning: {warning_msg}")
    return [], warning_msg


def simplify_cpp_results(results: list) -> list:
    simplified = []
    for item in results:
        entity_type = str(item.get("core_entity", {}).get("type", "")).lower()
        out = {"qualified_name": item.get("qualified_name")}

        ctx = item.get("retrieved_context") or {}
        slim_ctx = {}

        if entity_type == "function":
            for rel_key in ["calls", "called_by", "overrides", "overridden_by"]:
                if rel_key in ctx:
                    rels = [
                        {
                            "qualifiedName": rel.get("qualifiedName"),
                            "code_snippet": rel.get("code_snippet"),
                        }
                        for rel in ctx[rel_key]
                    ]
                    rels = [r for r in rels if r.get("qualifiedName")]
                    if rels:
                        slim_ctx[rel_key] = rels

        if entity_type in {"class", "struct", "union", "template"}:
            if "inherits_from" in ctx:
                inherits = []
                for rel in ctx["inherits_from"]:
                    e = {"qualifiedName": rel.get("qualifiedName")}
                    snippets = [
                        {"qualifiedName": s.get("qualifiedName"), "code_snippet": s.get("code_snippet")}
                        for s in rel.get("contained_functions_snippets", [])
                        if s.get("code_snippet")
                    ]
                    if snippets:
                        e["contained_functions_snippets"] = snippets
                    inherits.append(e)
                inherits = [x for x in inherits if x.get("qualifiedName")]
                if inherits:
                    slim_ctx["inherits_from"] = inherits

            if "instantiated_in" in ctx:
                inst = [
                    {"File": rel.get("File"), "code_snippet": rel.get("code_snippet")}
                    for rel in ctx["instantiated_in"]
                    if rel.get("code_snippet")
                ]
                if inst:
                    slim_ctx["instantiated_in"] = inst

            if ctx.get("subclasses"):
                slim_ctx["subclasses"] = ctx["subclasses"]

        if entity_type == "file":
            for rel_key in ["includes", "included_by", "defines"]:
                if rel_key in ctx:
                    rels = [
                        {
                            "qualifiedName": rel.get("qualifiedName"),
                            "code_snippet": rel.get("code_snippet"),
                        }
                        for rel in ctx[rel_key]
                    ]
                    rels = [r for r in rels if r.get("qualifiedName")]
                    if rels:
                        slim_ctx[rel_key] = rels

        if entity_type == "namespace" and ctx.get("defines"):
            rels = [
                {
                    "qualifiedName": rel.get("qualifiedName"),
                    "code_snippet": rel.get("code_snippet"),
                }
                for rel in ctx["defines"]
            ]
            rels = [r for r in rels if r.get("qualifiedName")]
            if rels:
                slim_ctx["defines"] = rels

        if "total_relations" in ctx:
            slim_ctx["total_relations"] = ctx["total_relations"]

        if slim_ctx:
            out["retrieved_context"] = slim_ctx
        simplified.append(out)

    return simplified


def load_commit_map(commit_map_file: str) -> dict:
    commit_map = {}
    try:
        with open(commit_map_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in commit map file: {line.strip()}")
                    continue
                instance_id = data.get("instance_id")
                commit_sha = data.get("commit_sha")
                if not commit_sha and "subset_entry" in data:
                    subset_entry = data.get("subset_entry")
                    if isinstance(subset_entry, dict):
                        commit_sha = subset_entry.get("commit_sha")
                if instance_id and commit_sha:
                    commit_map[instance_id] = commit_sha
    except FileNotFoundError:
        print(f"Error: Commit map file not found at '{commit_map_file}'.")
        return {}
    except Exception as e:
        print(f"Error reading commit map file: {e}")
        return {}
    return commit_map


def checkout_repo(repo_path: str, commit_sha: str) -> bool:
    try:
        if not os.path.exists(repo_path):
            print(f"Error: Repository path not found: {repo_path}")
            return False
        subprocess.run(["git", "checkout", "-f", commit_sha], cwd=repo_path, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        # On Windows, checkout may return non-zero because of unlink warnings
        # even when HEAD already moved to the target commit.
        try:
            verify = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            current_head = verify.stdout.strip()
            if current_head == commit_sha:
                print(
                    f"Warning: checkout returned non-zero, but HEAD already at target commit "
                    f"{commit_sha[:12]} in {repo_path}. Continuing."
                )
                return True
        except Exception:
            pass

        print(f"Error checking out commit {commit_sha} in {repo_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during git checkout: {e}")
        return False


def _find_enre_report(all_enre_report_dir: str, instance_id: str) -> str | None:
    candidates = [
        os.path.join(all_enre_report_dir, f"{instance_id}.json"),
        os.path.join(all_enre_report_dir, f"{instance_id}_enre_report.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_json_with_fallback(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return json.load(f)


def process_workflow(core_entities_file: str, commit_map_file: str, all_enre_report_dir: str, all_repo_dir: str, output_file: str, error_report_file: str):
    print(f"Starting workflow with input file: {core_entities_file}")
    print(f"Loading commit map from: {commit_map_file}")

    commit_map = load_commit_map(commit_map_file)
    if not commit_map:
        print("Error: Failed to load commit map or map is empty. Aborting workflow.")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    error_dir = os.path.dirname(error_report_file)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    total_lines = 0
    try:
        with open(core_entities_file, "r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
    except Exception:
        pass

    try:
        with open(output_file, "w", encoding="utf-8") as f_out, \
             open(error_report_file, "w", encoding="utf-8") as f_err, \
             open(core_entities_file, "r", encoding="utf-8") as f_in:

            def write_instance_error(instance_id: str, condition: str, detail: str | None = None) -> None:
                payload = {
                    "instance_id": instance_id,
                    "condition": condition,
                }
                if condition == "all_core_entities_not_retrieved":
                    payload["error"] = "所有核心实体均未检索到"
                elif condition == "retrieved_entities_without_related_code":
                    payload["error"] = "存在核心实体被检索到，但是检索到的实体没有相关代码"
                if detail:
                    payload["detail"] = detail
                f_err.write(json.dumps(payload, ensure_ascii=False) + "\n")

            for line in tqdm(f_in, total=total_lines, desc="Processing C++ Entities", unit="line"):
                try:
                    instance = json.loads(line)
                except json.JSONDecodeError:
                    error_msg = f"Skipping malformed JSON line in {core_entities_file}: {line.strip()}"
                    print(f"Warning: {error_msg}")
                    continue

                instance_id = instance.get("instance_id")
                core_entities = instance.get("core_entities")
                instance_results = []

                if not instance_id:
                    error_msg = f"Skipping instance due to missing 'instance_id'. Instance: {instance}"
                    print(f"Warning: {error_msg}")
                    continue

                if core_entities is None or not isinstance(core_entities, list):
                    error_msg = "Missing 'core_entities' field."
                    print(f"Warning: {error_msg} Instance: {instance_id}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                if len(core_entities) == 0:
                    error_msg = "'core_entities' is empty."
                    print(f"Warning: {error_msg} Instance: {instance_id}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                commit_sha = commit_map.get(instance_id)
                if not commit_sha:
                    error_msg = f"Missing 'commit_sha' for instance_id: {instance_id} in commit map file."
                    print(f"Warning: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                parts = instance_id.split("_")
                if len(parts) < 3:
                    error_msg = f"Invalid instance_id format '{instance_id}'. Cannot derive project root."
                    print(f"Warning: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                repo_name = "_".join(parts[1:-1])
                project_root = os.path.join(all_repo_dir, repo_name)

                if not checkout_repo(project_root, commit_sha):
                    error_msg = f"Failed to checkout commit {commit_sha} in {project_root}"
                    print(f"Error: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                enre_report_file = _find_enre_report(all_enre_report_dir, instance_id)
                if not enre_report_file:
                    error_msg = f"ENRE report for instance '{instance_id}' not found."
                    print(f"Warning: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                try:
                    enre_data = _load_json_with_fallback(enre_report_file)
                except Exception as e:
                    error_msg = f"Failed to load ENRE report: {e}"
                    print(f"Error: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                variables = _normalize_cpp_variables(enre_data)
                if not variables:
                    print(f"Warning: No variables found in ENRE report for instance '{instance_id}'.")

                supported_entities_count = 0
                resolved_entities_count = 0
                resolved_entities_with_code_count = 0

                for entity in core_entities:
                    entity_type = str(entity.get("type", "")).lower()
                    if entity_type not in {"function", "class", "struct", "union", "template", "file", "namespace"}:
                        continue
                    supported_entities_count += 1

                    qualified_names, warning_msg = generate_cpp_qualified_names(entity, variables)

                    if not qualified_names:
                        instance_results.append(
                            {
                                "instance_id": instance_id,
                                "core_entity": entity,
                                "qualified_name": None,
                                "retrieved_context": {},
                            }
                        )
                        continue

                    resolved_entities_count += 1
                    entity_has_related_code = False

                    for qualified_name in qualified_names:
                        relations = analyze_cpp_enre_report(
                            report_path=enre_report_file,
                            target_qualified_name=qualified_name,
                            enre_data=enre_data,
                        )

                        if not relations:
                            instance_results.append(
                                {
                                    "instance_id": instance_id,
                                    "core_entity": entity,
                                    "qualified_name": qualified_name,
                                    "retrieved_context": {},
                                }
                            )
                            continue

                        augmented_relations = retrieve_code_context(
                            entity_relations=relations[0],
                            enre_report_path=enre_report_file,
                            project_root=project_root,
                        )

                        if augmented_relations:
                            entity_has_related_code = True
                            instance_results.append(
                                {
                                    "instance_id": instance_id,
                                    "core_entity": entity,
                                    "qualified_name": qualified_name,
                                    "retrieved_context": augmented_relations,
                                }
                            )
                        else:
                            instance_results.append(
                                {
                                    "instance_id": instance_id,
                                    "core_entity": entity,
                                    "qualified_name": qualified_name,
                                    "retrieved_context": {},
                                }
                            )

                    if entity_has_related_code:
                        resolved_entities_with_code_count += 1

                if instance_results:
                    output_line = {
                        "instance_id": instance_id,
                        "retrieved_code": simplify_cpp_results(instance_results),
                    }
                    f_out.write(json.dumps(output_line, ensure_ascii=False) + "\n")

                if resolved_entities_count == 0:
                    detail = f"supported_core_entities={supported_entities_count}, resolved_entities=0"
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", detail)
                elif resolved_entities_with_code_count == 0:
                    detail = f"resolved_entities={resolved_entities_count}, resolved_entities_with_code=0"
                    write_instance_error(instance_id, "retrieved_entities_without_related_code", detail)

        print(f"Workflow complete. Results saved in: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{core_entities_file}'.")
    except IOError as e:
        print(f"Error during file operations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process C/C++ core entities, analyze ENRE relations, and retrieve related code contexts."
    )
    parser.add_argument("--core_entities_file", required=True, help="Path to the JSONL output from entity extraction.")
    parser.add_argument("--commit_map_file", required=True, help="Path to JSONL mapping instance_id -> commit_sha.")
    parser.add_argument("--all_enre_report_dir", required=True, help="Directory containing ENRE-C/C++ reports.")
    parser.add_argument("--all_repo_dir", required=True, help="Directory containing all C/C++ repositories.")
    parser.add_argument("--output_file", required=True, help="Path to save final related code JSONL.")
    parser.add_argument("--error_report_file", required=True, help="Path to save workflow error report JSONL.")

    args = parser.parse_args()

    process_workflow(
        core_entities_file=args.core_entities_file,
        commit_map_file=args.commit_map_file,
        all_enre_report_dir=args.all_enre_report_dir,
        all_repo_dir=args.all_repo_dir,
        output_file=args.output_file,
        error_report_file=args.error_report_file,
    )


if __name__ == "__main__":
    main()
