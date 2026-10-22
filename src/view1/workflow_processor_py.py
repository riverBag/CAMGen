import os
import sys
import json
import argparse
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, "utils_relatedcode")
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from py_utils.relation_analyzer import analyze_python_enre_report
from py_utils.code_retriever import retrieve_code_context


PY_TYPE_TO_CATEGORY = {
    "function": "Function",
    "class": "Class",
    "module": "Module",
    "package": "Package",
}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def _normalize_rel_path(path: str | None) -> str:
    if not isinstance(path, str):
        return ""
    normalized = path.replace("\\", "/").strip()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _path_matches(entity_path: str | None, enre_file_path: str | None) -> bool:
    normalized_entity = _normalize_rel_path(entity_path)
    normalized_enre = _normalize_rel_path(enre_file_path)
    if not normalized_entity or not normalized_enre:
        return False
    if normalized_entity == normalized_enre:
        return True
    return normalized_enre.endswith("/" + normalized_entity)


def _strip_diff_path_prefix(path: str | None) -> str:
    normalized = _normalize_rel_path(path)
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


def _parse_unified_diff_changed_lines(diff_text: str | None) -> dict[str, set[int]]:
    if not isinstance(diff_text, str) or not diff_text:
        return {}

    changed_lines: dict[str, set[int]] = {}
    current_file = ""
    new_line = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            path = raw_line[4:].strip()
            current_file = "" if path == "/dev/null" else _strip_diff_path_prefix(path)
            if current_file:
                changed_lines.setdefault(current_file, set())
            new_line = None
            continue

        if raw_line.startswith("@@"):
            header_parts = raw_line.split("@@", 2)
            header = header_parts[1] if len(header_parts) > 1 else ""
            plus_part = None
            for part in header.strip().split():
                if part.startswith("+"):
                    plus_part = part
                    break
            if plus_part:
                start = plus_part[1:].split(",", 1)[0]
                try:
                    new_line = int(start)
                except ValueError:
                    new_line = None
            continue

        if new_line is None or not current_file:
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            changed_lines.setdefault(current_file, set()).add(new_line)
            new_line += 1
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            continue
        elif raw_line.startswith("\\"):
            continue
        else:
            new_line += 1

    return {path: lines for path, lines in changed_lines.items() if lines}


def _get_entity_changed_lines(entity: dict, changed_lines_by_file: dict[str, set[int]] | None) -> set[int]:
    if not changed_lines_by_file:
        return set()
    entity_path = entity.get("path")
    lines: set[int] = set()
    for file_path, file_lines in changed_lines_by_file.items():
        if _path_matches(entity_path, file_path):
            lines.update(file_lines)
    return lines


def _python_module_from_path(path: str | None) -> str:
    rel = _normalize_rel_path(path)
    if not rel:
        return ""
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return rel.replace("/", ".")


def _extract_python_name_from_qn(qualified_name: str | None) -> str:
    if not qualified_name:
        return ""
    return qualified_name.rsplit(".", 1)[-1]


def _class_name_matches(core_class_name: str | None, class_node: dict | None) -> bool:
    if not core_class_name or not class_node:
        return False
    class_qn = class_node.get("qualifiedName", "")
    class_simple_name = _extract_python_name_from_qn(class_qn)
    return core_class_name == class_simple_name or core_class_name == class_qn or class_qn.endswith("." + core_class_name)


def _build_python_parent_maps(variables: list[dict]) -> tuple[dict, dict]:
    id_to_node = {var.get("id"): var for var in variables if var.get("id") is not None}
    class_qname_to_node = {
        var.get("qualifiedName"): var
        for var in variables
        if var.get("category") == "Class" and var.get("qualifiedName")
    }
    return id_to_node, class_qname_to_node


def _infer_enclosing_class(function_node: dict, id_to_node: dict, class_qname_to_node: dict) -> dict | None:
    parent_id = function_node.get("parentId")
    parent_node = id_to_node.get(parent_id)
    if parent_node and parent_node.get("category") == "Class":
        return parent_node

    function_qn = function_node.get("qualifiedName", "")
    best_class_qn = ""
    best_class_node = None
    for class_qn, class_node in class_qname_to_node.items():
        if function_qn.startswith(class_qn + ".") and len(class_qn) > len(best_class_qn):
            best_class_qn = class_qn
            best_class_node = class_node
    return best_class_node


def _module_name_matches(entity: dict, variable: dict) -> bool:
    qn = variable.get("qualifiedName", "")
    name = entity.get("name")
    module_qn = _python_module_from_path(entity.get("path"))
    explicit_qn = entity.get("qualified_name") or entity.get("qualifiedName")
    if explicit_qn and qn == explicit_qn:
        return True
    if module_qn and qn == module_qn:
        return True
    return bool(name and (qn == name or qn.endswith("." + name)))


def _line_from_location(location: dict | None, *keys: str) -> int | None:
    if not isinstance(location, dict):
        return None
    for key in keys:
        value = location.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return None


def _candidate_start_line(candidate: dict) -> int | None:
    return _line_from_location(candidate.get("location"), "startLine", "start_line", "line")


def _candidate_end_line(candidate: dict) -> int | None:
    return _line_from_location(candidate.get("location"), "endLine", "end_line")


def _infer_candidate_span(candidate: dict, variables: list[dict], changed_lines: set[int]) -> tuple[int, int] | None:
    start_line = _candidate_start_line(candidate)
    if start_line is None:
        return None

    end_line = _candidate_end_line(candidate)
    if end_line is not None and end_line >= start_line:
        return start_line, end_line

    candidate_file = candidate.get("File")
    next_starts = []
    for var in variables:
        if var is candidate:
            continue
        if var.get("category") not in {"Function", "Class"}:
            continue
        if not _path_matches(candidate_file, var.get("File")):
            continue
        other_start = _candidate_start_line(var)
        if other_start is not None and other_start > start_line:
            next_starts.append(other_start)

    if next_starts:
        return start_line, min(next_starts) - 1

    if changed_lines:
        return start_line, max(max(changed_lines), start_line)

    return start_line, start_line


def _narrow_matches_by_changed_lines(
    entity: dict,
    matches: list[dict],
    variables: list[dict],
    changed_lines_by_file: dict[str, set[int]] | None,
) -> tuple[list[dict], str | None]:
    if len(matches) <= 1:
        return matches, None

    changed_lines = _get_entity_changed_lines(entity, changed_lines_by_file)
    if not changed_lines:
        return matches, None

    hits = []
    for candidate in matches:
        span = _infer_candidate_span(candidate, variables, changed_lines)
        if not span:
            continue
        start_line, end_line = span
        matched_lines = [line for line in changed_lines if start_line <= line <= end_line]
        if matched_lines:
            hits.append(
                {
                    "candidate": candidate,
                    "line_hits": len(matched_lines),
                    "span_size": end_line - start_line + 1,
                    "start_line": start_line,
                }
            )

    if not hits:
        return matches, None

    hits.sort(key=lambda item: (-item["line_hits"], item["span_size"], item["start_line"]))
    best = hits[0]
    tied = [
        item for item in hits
        if item["line_hits"] == best["line_hits"] and item["span_size"] == best["span_size"]
    ]
    if len(tied) == 1:
        qn = best["candidate"].get("qualifiedName")
        return [best["candidate"]], f"Narrowed ambiguous mapping by diff changed lines to: {qn}"

    return matches, None


def generate_python_qualified_names(
    entity: dict,
    variables: list[dict],
    changed_lines_by_file: dict[str, set[int]] | None = None,
) -> tuple[list[str], str | None]:
    entity_type = str(entity.get("type", "")).lower()
    expected_category = PY_TYPE_TO_CATEGORY.get(entity_type)
    target_name = entity.get("name")
    class_name = entity.get("class_name")

    if entity_type not in PY_TYPE_TO_CATEGORY:
        warning_msg = f"Unsupported Python entity type '{entity_type}'. Entity: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    if entity_type in {"function", "class"} and not all([entity.get("path"), target_name, entity_type]):
        warning_msg = f"Skipping entity due to missing 'path', 'name', or 'type'. Entity: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    id_to_node, class_qname_to_node = _build_python_parent_maps(variables)

    matches = []
    for var in variables:
        if var.get("category") != expected_category:
            continue
        if not var.get("qualifiedName"):
            continue

        if entity_type in {"function", "class", "module"} and entity.get("path"):
            if not _path_matches(entity.get("path"), var.get("File")):
                continue

        qn = var.get("qualifiedName", "")
        qn_name = _extract_python_name_from_qn(qn)

        if entity_type in {"function", "class"} and qn_name != target_name:
            continue

        if entity_type == "function" and class_name:
            parent_class = _infer_enclosing_class(var, id_to_node, class_qname_to_node)
            if not _class_name_matches(class_name, parent_class):
                continue

        if entity_type in {"module", "package"} and not _module_name_matches(entity, var):
            continue

        matches.append(var)

    if len(matches) == 1:
        return [matches[0].get("qualifiedName")], None

    if len(matches) > 1:
        narrowed_matches, narrowed_msg = _narrow_matches_by_changed_lines(
            entity=entity,
            matches=matches,
            variables=variables,
            changed_lines_by_file=changed_lines_by_file,
        )
        if len(narrowed_matches) == 1:
            return [narrowed_matches[0].get("qualifiedName")], narrowed_msg

        if entity_type == "function" and not class_name:
            warning_msg = f"Ambiguous function mapping for entity (missing class_name and multiple file-level matches): {entity}"
        else:
            warning_msg = f"Ambiguous {entity_type} mapping for entity (multiple exact matches): {entity}"

        print(f"Warning: {warning_msg}")
        return [], warning_msg

    warning_msg = f"Could not find {entity_type} entity in ENRE report: {entity}"
    print(f"Warning: {warning_msg}")
    return [], warning_msg


def resolve_python_qualified_name(entity: dict, variables: list[dict]) -> str | None:
    qualified_names, _ = generate_python_qualified_names(entity, variables)
    return qualified_names[0] if qualified_names else None


def _is_valid_code_snippet(snippet) -> bool:
    if not isinstance(snippet, str):
        return False
    stripped = snippet.strip()
    if not stripped:
        return False
    return not stripped.startswith("Error:")


def _dedupe_dicts(items: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    deduped = []
    seen = set()
    for item in items:
        key = tuple(item.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _count_output_relations(ctx: dict) -> int:
    total = 0
    for rel_key in ["calls", "called_by", "instantiated_in", "imports", "imported_by", "contains", "subclasses"]:
        total += len(ctx.get(rel_key, []))
    for rel in ctx.get("inherits_from", []):
        total += len(rel.get("contained_functions_snippets", []))
    return total


def simplify_python_results(results: list) -> list:
    simplified = []
    for item in results:
        entity_type = str(item.get("core_entity", {}).get("type", "")).lower()
        out = {"qualified_name": item.get("qualified_name")}

        ctx = item.get("retrieved_context") or {}
        slim_ctx = {}

        if entity_type == "function":
            for rel_key in ["calls", "called_by"]:
                rels = [
                    {
                        "qualifiedName": rel.get("qualifiedName"),
                        "code_snippet": rel.get("code_snippet"),
                    }
                    for rel in ctx.get(rel_key, [])
                ]
                rels = [
                    rel for rel in rels
                    if rel.get("qualifiedName") and _is_valid_code_snippet(rel.get("code_snippet"))
                ]
                rels = _dedupe_dicts(rels, ("qualifiedName", "code_snippet"))
                if rels:
                    slim_ctx[rel_key] = rels

        if entity_type == "class":
            inherits = []
            for rel in ctx.get("inherits_from", []):
                item_out = {"qualifiedName": rel.get("qualifiedName")}
                snippets = [
                    {
                        "qualifiedName": snippet.get("qualifiedName"),
                        "code_snippet": snippet.get("code_snippet"),
                    }
                    for snippet in rel.get("contained_functions_snippets", [])
                    if snippet.get("qualifiedName") and _is_valid_code_snippet(snippet.get("code_snippet"))
                ]
                snippets = _dedupe_dicts(snippets, ("qualifiedName", "code_snippet"))
                if snippets:
                    item_out["contained_functions_snippets"] = snippets
                if item_out.get("qualifiedName"):
                    inherits.append(item_out)
            inherits = _dedupe_dicts(inherits, ("qualifiedName",))
            if inherits:
                slim_ctx["inherits_from"] = inherits

            inst = [
                {
                    "File": rel.get("File"),
                    "code_snippet": rel.get("code_snippet"),
                }
                for rel in ctx.get("instantiated_in", [])
                if rel.get("File") and _is_valid_code_snippet(rel.get("code_snippet"))
            ]
            inst = _dedupe_dicts(inst, ("File", "code_snippet"))
            if inst:
                slim_ctx["instantiated_in"] = inst

            if ctx.get("subclasses"):
                slim_ctx["subclasses"] = _dedupe_dicts(ctx["subclasses"], ("qualifiedName", "File"))

        if entity_type in {"module", "package"}:
            for rel_key in ["imports", "imported_by", "contains"]:
                rels = [
                    {
                        "qualifiedName": rel.get("qualifiedName"),
                        "code_snippet": rel.get("code_snippet"),
                    }
                    for rel in ctx.get(rel_key, [])
                ]
                rels = [
                    rel for rel in rels
                    if rel.get("qualifiedName") and _is_valid_code_snippet(rel.get("code_snippet"))
                ]
                rels = _dedupe_dicts(rels, ("qualifiedName", "code_snippet"))
                if rels:
                    slim_ctx[rel_key] = rels

        if slim_ctx:
            slim_ctx["total_relations"] = _count_output_relations(slim_ctx)
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
                    continue
                instance_id = data.get("instance_id")
                commit_sha = data.get("commit_sha")

                if not commit_sha and isinstance(data.get("subset_entry"), dict):
                    commit_sha = data["subset_entry"].get("commit_sha")

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
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_json_with_fallback(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return json.load(f)


def _write_jsonl_record(handle, payload: dict) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_processed_instance_ids(*jsonl_files: str) -> set[str]:
    processed = set()
    for jsonl_file in jsonl_files:
        if not jsonl_file or not os.path.exists(jsonl_file):
            continue
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    instance_id = item.get("instance_id")
                    if instance_id:
                        processed.add(instance_id)
        except OSError as e:
            log_message(f"Warning: failed to load processed ids from {jsonl_file}: {e}")
    return processed


def process_workflow(
    core_entities_file: str,
    commit_map_file: str,
    all_enre_report_dir: str,
    all_repo_dir: str,
    output_file: str,
    error_report_file: str,
    max_instances: int | None = None,
    resume: bool = False,
):
    log_message(f"Starting workflow with input file: {core_entities_file}")
    if max_instances is not None:
        log_message(f"Processing only the first {max_instances} input lines.")
    if resume:
        log_message("Resume mode enabled: existing output/error instance_ids will be skipped.")
    log_message(f"Loading commit map from: {commit_map_file}")

    commit_map = load_commit_map(commit_map_file)
    if not commit_map:
        log_message("Error: Failed to load commit map or map is empty. Aborting workflow.")
        return
    log_message(f"Loaded commit map entries: {len(commit_map)}")

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
    if max_instances is not None and total_lines:
        total_lines = min(total_lines, max_instances)

    processed_instance_ids = _load_processed_instance_ids(output_file, error_report_file) if resume else set()
    if processed_instance_ids:
        log_message(f"Resume mode loaded processed instance_ids: {len(processed_instance_ids)}")

    file_mode = "a" if resume else "w"

    try:
        with open(output_file, file_mode, encoding="utf-8", buffering=1) as f_out, \
             open(error_report_file, file_mode, encoding="utf-8", buffering=1) as f_err, \
             open(core_entities_file, "r", encoding="utf-8") as f_in:

            for line_no, line in enumerate(tqdm(f_in, total=total_lines, desc="Processing Python Entities", unit="line"), start=1):
                if max_instances is not None and line_no > max_instances:
                    break

                try:
                    instance = json.loads(line)
                except json.JSONDecodeError:
                    _write_jsonl_record(f_err, {"error": "Malformed JSON line.", "line": line.strip()})
                    log_message("Warning: skipped malformed JSON line.")
                    continue

                instance_id = instance.get("instance_id")
                core_entities = instance.get("core_entities")

                if not instance_id:
                    _write_jsonl_record(f_err, {"instance_id": "unknown", "error": "Missing 'instance_id'."})
                    log_message("Warning: skipped entry with missing instance_id.")
                    continue

                if instance_id in processed_instance_ids:
                    continue

                if core_entities is None or not isinstance(core_entities, list):
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": "Missing 'core_entities' field."})
                    log_message(f"[{instance_id}] Error: missing or invalid core_entities.")
                    continue

                if len(core_entities) == 0:
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": "'core_entities' is empty."})
                    log_message(f"[{instance_id}] Error: core_entities is empty.")
                    continue

                commit_sha = commit_map.get(instance_id)
                if not commit_sha:
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": "Missing commit sha in commit map."})
                    log_message(f"[{instance_id}] Error: missing commit sha in commit map.")
                    continue

                parts = instance_id.split("_")
                if len(parts) < 3:
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": "Invalid instance_id format."})
                    log_message(f"[{instance_id}] Error: invalid instance_id format.")
                    continue

                repo_name = "_".join(parts[1:-1])
                project_root = os.path.join(all_repo_dir, repo_name)
                log_message(
                    f"[{instance_id}] Processing repo={repo_name}, commit={commit_sha[:12]}, "
                    f"core_entities={len(core_entities)}"
                )

                if not checkout_repo(project_root, commit_sha):
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": f"Failed to checkout commit {commit_sha}."})
                    log_message(f"[{instance_id}] Error: failed to checkout commit {commit_sha}.")
                    continue

                enre_report_file = _find_enre_report(all_enre_report_dir, instance_id)
                if not enre_report_file:
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": "ENRE report not found."})
                    log_message(f"[{instance_id}] Error: ENRE report not found.")
                    continue
                log_message(f"[{instance_id}] Using ENRE report: {enre_report_file}")

                try:
                    enre_data = _load_json_with_fallback(enre_report_file)
                except Exception as e:
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": f"Failed to load ENRE report: {e}"})
                    log_message(f"[{instance_id}] Error: failed to load ENRE report: {e}")
                    continue

                variables = enre_data.get("variables", []) if isinstance(enre_data, dict) else []
                log_message(f"[{instance_id}] ENRE variables: {len(variables)}")

                subset_entry = instance.get("subset_entry") if isinstance(instance.get("subset_entry"), dict) else {}
                diff_text = subset_entry.get("diff") or instance.get("diff")
                changed_lines_by_file = _parse_unified_diff_changed_lines(diff_text)

                instance_results = []
                valid_entity_found = False
                supported_entities_count = 0
                resolved_entities_count = 0

                for entity in core_entities:
                    entity_type = str(entity.get("type", "")).lower()
                    if entity_type not in {"function", "class", "module", "package"}:
                        log_message(f"[{instance_id}] Skipping unsupported entity type: {entity.get('type')}")
                        continue

                    supported_entities_count += 1
                    qualified_names, warning_msg = generate_python_qualified_names(
                        entity,
                        variables,
                        changed_lines_by_file=changed_lines_by_file,
                    )
                    if warning_msg:
                        log_message(f"[{instance_id}] Warning: {warning_msg}")

                    if not qualified_names:
                        continue

                    valid_entity_found = True
                    resolved_entities_count += 1
                    log_message(f"[{instance_id}] Resolved {entity} -> {qualified_names}")

                    for qualified_name in qualified_names:
                        relations = analyze_python_enre_report(
                            report_path=enre_report_file,
                            target_qualified_name=qualified_name,
                            enre_data=enre_data,
                        )

                        if not relations:
                            log_message(f"[{instance_id}] No relations found for {qualified_name}.")
                            continue

                        augmented_relations = retrieve_code_context(
                            entity_relations=relations[0],
                            enre_report_path=enre_report_file,
                            project_root=project_root,
                        )

                        if augmented_relations:
                            instance_results.append(
                                {
                                    "instance_id": instance_id,
                                    "core_entity": entity,
                                    "qualified_name": qualified_name,
                                    "retrieved_context": augmented_relations,
                                }
                            )
                            log_message(f"[{instance_id}] Retrieved related code for {qualified_name}.")

                simplified_results = simplify_python_results(instance_results)
                if simplified_results:
                    output_line = {
                        "instance_id": instance_id,
                        "retrieved_code": simplified_results,
                    }
                    _write_jsonl_record(f_out, output_line)
                    log_message(
                        f"[{instance_id}] Done: supported={supported_entities_count}, "
                        f"resolved={resolved_entities_count}, output_entities={len(simplified_results)}"
                    )
                else:
                    if not valid_entity_found:
                        msg = "No valid entities found or resolved."
                    else:
                        msg = "Entities found but no valid code context retrieved."
                    _write_jsonl_record(f_err, {"instance_id": instance_id, "error": msg})
                    log_message(
                        f"[{instance_id}] Error: {msg} "
                        f"supported={supported_entities_count}, resolved={resolved_entities_count}"
                    )

        log_message(f"Workflow complete. Results saved in: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        log_message(f"Error: Input file not found at '{core_entities_file}'.")
    except IOError as e:
        log_message(f"Error during file operations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process Python core entities, analyze ENRE relations, and retrieve related code contexts."
    )
    parser.add_argument("--core_entities_file", required=True, help="Path to the JSONL output from entity extraction.")
    parser.add_argument("--commit_map_file", required=True, help="Path to JSONL mapping instance_id -> commit_sha.")
    parser.add_argument("--all_enre_report_dir", required=True, help="Directory containing ENRE-Python reports.")
    parser.add_argument("--all_repo_dir", required=True, help="Directory containing all Python repositories.")
    parser.add_argument("--output_file", required=True, help="Path to save final related code JSONL.")
    parser.add_argument("--error_report_file", required=True, help="Path to save workflow error report JSONL.")
    parser.add_argument("--log_file", default=None, help="Optional path to save the full runtime log while still printing to console.")
    parser.add_argument("--max_instances", type=int, default=None, help="Only process the first N input lines from core_entities_file.")
    parser.add_argument("--resume", action="store_true", help="Append outputs and skip instance_ids already present in output/error files.")

    args = parser.parse_args()

    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_file, "w", encoding="utf-8") as f_log:
            stdout_tee = TeeStream(sys.stdout, f_log)
            stderr_tee = TeeStream(sys.stderr, f_log)
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                log_message(f"Runtime log file: {os.path.abspath(args.log_file)}")
                process_workflow(
                    core_entities_file=args.core_entities_file,
                    commit_map_file=args.commit_map_file,
                    all_enre_report_dir=args.all_enre_report_dir,
                    all_repo_dir=args.all_repo_dir,
                    output_file=args.output_file,
                    error_report_file=args.error_report_file,
                    max_instances=args.max_instances,
                    resume=args.resume,
                )
    else:
        process_workflow(
            core_entities_file=args.core_entities_file,
            commit_map_file=args.commit_map_file,
            all_enre_report_dir=args.all_enre_report_dir,
            all_repo_dir=args.all_repo_dir,
            output_file=args.output_file,
            error_report_file=args.error_report_file,
            max_instances=args.max_instances,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
