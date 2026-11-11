
import os
import sys
import json
import argparse
import subprocess
from itertools import chain
from tqdm import tqdm

# Add the parent directory to sys.path to allow imports from java_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils_relatedcode'))

from java_utils.relation_analyzer import analyze_java_enre_report
from java_utils.code_retriever import retrieve_code_context, get_code_snippet

def _normalize_rel_path(path: str | None) -> str:
    if not path:
        return ""
    normalized = path.replace('\\', '/').strip()
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    if normalized.startswith('./'):
        normalized = normalized[2:]
    return normalized

def _path_matches(entity_path: str | None, enre_file_path: str | None) -> bool:
    """
    Strict path matching with format normalization.
    Supports exact match and segment-safe suffix match to tolerate different roots.
    """
    normalized_entity = _normalize_rel_path(entity_path)
    normalized_enre = _normalize_rel_path(enre_file_path)
    if not normalized_entity or not normalized_enre:
        return False
    if normalized_entity == normalized_enre:
        return True
    return normalized_enre.endswith('/' + normalized_entity)

def _extract_method_name_from_qn(qualified_name: str | None) -> str:
    if not qualified_name:
        return ""
    tail = qualified_name.rsplit('.', 1)[-1]
    return tail.split('(', 1)[0]

def _extract_type_name_from_qn(qualified_name: str | None) -> str:
    if not qualified_name:
        return ""
    tail = qualified_name.rsplit('.', 1)[-1]
    tail = tail.split('(', 1)[0]
    return tail.split('$')[-1]

def _class_name_matches(core_class_name: str, parent_node: dict) -> bool:
    """
    Strict-equivalence class name matching (not fuzzy similarity).
    """
    if not core_class_name:
        return False
    parent_qn = parent_node.get('qualifiedName', '')
    parent_simple_name = _extract_type_name_from_qn(parent_qn)
    return (
        core_class_name == parent_simple_name
        or core_class_name == parent_qn
        or parent_qn.endswith('.' + core_class_name)
    )

def generate_qualified_names(entity: dict, enre_data: dict) -> tuple[list[str], str | None]:
    """
    Resolves the exact ENRE qualifiedName for a core entity using strict structural
    matching on category, file path, and declared names.
    No fuzzy/similarity matching is used.

    Returns:
        tuple[list[str], str | None]:
        - one or more qualified names when resolved
        - warning message when resolution fails or is ambiguous, else None
    """
    entity_type = entity.get("type")
    path = entity.get("path")
    name = entity.get("name")

    if not all([path, name, entity_type]):
        warning_msg = f"Skipping entity due to missing 'path', 'name', or 'type'. Entity: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    variables = enre_data.get('variables', [])
    id_to_node = {var.get('id'): var for var in variables if var.get('id') is not None}

    if entity_type == 'class':
        class_candidates = []
        for var in variables:
            if var.get('category') != 'Class':
                continue
            if not _path_matches(path, var.get('File')):
                continue
            candidate_name = _extract_type_name_from_qn(var.get('qualifiedName'))
            if candidate_name == name:
                class_candidates.append(var)

        if len(class_candidates) == 1:
            return [class_candidates[0].get('qualifiedName')], None

        if len(class_candidates) > 1:
            warning_msg = f"Ambiguous class mapping for entity (multiple exact matches): {entity}"
            candidate_qns = [c.get('qualifiedName') for c in class_candidates if c.get('qualifiedName')]
            print(f"Warning: {warning_msg}")
            return candidate_qns, warning_msg
        else:
            warning_msg = f"Could not find class entity in ENRE report: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    if entity_type == 'function':
        class_name = entity.get("class_name")

        method_candidates = []
        for var in variables:
            if var.get('category') != 'Method':
                continue
            if not _path_matches(path, var.get('File')):
                continue
            method_name = _extract_method_name_from_qn(var.get('qualifiedName'))
            if method_name != name:
                continue

            parent_node = id_to_node.get(var.get('parentId'))
            if not parent_node:
                continue
            if parent_node.get('category') not in ['Class', 'Interface']:
                continue
            if class_name and not _class_name_matches(class_name, parent_node):
                continue

            method_candidates.append(var)

        # class_name is missing in some core entities. In this case we keep strict
        # resolution by requiring a unique match in the same file+method scope.
        if not class_name:
            anonymous_candidates = []
            for candidate in method_candidates:
                parent_node = id_to_node.get(candidate.get('parentId'), {})
                parent_qn = parent_node.get('qualifiedName', '')
                parent_simple_name = _extract_type_name_from_qn(parent_qn)
                if (
                    parent_simple_name == 'Anonymous_Class'
                    or parent_qn.endswith('.Anonymous_Class')
                    or '$' in parent_qn
                ):
                    anonymous_candidates.append(candidate)

            if len(anonymous_candidates) == 1:
                return [anonymous_candidates[0].get('qualifiedName')], None

            if len(anonymous_candidates) > 1:
                warning_msg = f"Ambiguous function mapping for entity (multiple anonymous class matches): {entity}"
                candidate_qns = [c.get('qualifiedName') for c in anonymous_candidates if c.get('qualifiedName')]
                print(f"Warning: {warning_msg}")
                return candidate_qns, warning_msg

            if len(method_candidates) == 1:
                return [method_candidates[0].get('qualifiedName')], None

            if len(method_candidates) > 1:
                warning_msg = f"Ambiguous function mapping for entity (missing class_name and multiple file-level method matches): {entity}"
                candidate_qns = [c.get('qualifiedName') for c in method_candidates if c.get('qualifiedName')]
                print(f"Warning: {warning_msg}")
                return candidate_qns, warning_msg
            else:
                warning_msg = f"Could not find function entity in ENRE report (missing class_name): {entity}"
            print(f"Warning: {warning_msg}")
            return [], warning_msg

        if len(method_candidates) == 1:
            return [method_candidates[0].get('qualifiedName')], None

        if len(method_candidates) > 1:
            warning_msg = f"Ambiguous function mapping for entity (overloads or duplicates): {entity}"
            candidate_qns = [c.get('qualifiedName') for c in method_candidates if c.get('qualifiedName')]
            print(f"Warning: {warning_msg}")
            return candidate_qns, warning_msg
        else:
            warning_msg = f"Could not find function entity in ENRE report: {entity}"
        print(f"Warning: {warning_msg}")
        return [], warning_msg

    warning_msg = f"Unsupported entity type '{entity_type}'. Entity: {entity}"
    print(f"Warning: {warning_msg}")
    return [], warning_msg

def get_overridden_method_context(function_qualified_name: str, enre_data: dict, project_root: str) -> list:
    """
    For a given function, finds the corresponding method in its parent class(es)
    and retrieves a code snippet. This now receives the correct function QN directly.
    """
    overridden_contexts = []
    
    # 1. Derive the class QN and function name from the function's qualified name.
    # e.g., "com.example.MyClass.myMethod" -> "com.example.MyClass", "myMethod"
    parts = function_qualified_name.split('.')
    if len(parts) < 2:
        return []
    class_qn = ".".join(parts[:-1])
    print(f"Class QN: {class_qn}")
    function_name = parts[-1]

    # 2. Analyze the class to find its parents.
    # Note: This still calls analyze_java_enre_report, which re-parses the file.
    # A future optimization could be to pass the pre-analyzed relations.
    class_relations_list = analyze_java_enre_report(
        report_path=None, # Passing data directly
        target_qualified_name=class_qn,
        enre_data=enre_data
    )
    if not class_relations_list:
        return []
    
    class_relations = class_relations_list[0]
    #print(f"Class relations for {class_qn}: {class_relations}")

    # 3. Create lookup maps from the pre-loaded ENRE data.
    variables = enre_data.get('variables', [])
    enre_node_map = {var['qualifiedName']: var for var in variables if var.get('qualifiedName')}
    parent_id_to_methods = {}
    for var in variables:
        if var.get('category') != 'Method':
            continue
        parent_id = var.get('parentId')
        if parent_id is None:
            continue
        if parent_id not in parent_id_to_methods:
            parent_id_to_methods[parent_id] = []
        parent_id_to_methods[parent_id].append(var)

    # 4. Check each parent class or implemented interface for an overridden method.
    all_parents = chain(class_relations.get("inherits_from", []), class_relations.get("implements", []))
    for parent_info in all_parents:
        parent_qn = parent_info.get("qualifiedName")
        if not parent_qn:
            continue
        
        parent_node = enre_node_map.get(parent_qn)
        if not parent_node:
            continue
        parent_id = parent_node.get('id')
        if parent_id is None:
            continue

        candidate_methods = parent_id_to_methods.get(parent_id, [])
        for target_node in candidate_methods:
            method_qn = target_node.get('qualifiedName', '')
            if _extract_method_name_from_qn(method_qn) != function_name:
                continue

            file_path = os.path.join(project_root, target_node.get('File', ''))
            location = target_node.get('location', {})
            start_line = location.get('startLine')
            
            if start_line:
                # Get 3 lines of context around the start of the method definition
                snippet = get_code_snippet(file_path, start_line, context_lines=3)
                if snippet:
                    overridden_contexts.append({
                        "parent_qualifiedName": parent_qn,
                        "method_qualifiedName": method_qn,
                        "code_snippet": snippet,
                        "file_path": target_node.get('File', ''),
                        "location": location
                    })

    return overridden_contexts

def simplify_results(results: list) -> list:
    """
    Simplifies the structure of the final results to remove redundant information
    before saving to a file.
    """
    simplified_list = []
    for item in results:
        # Determine entity type from the original core_entity, which will be discarded.
        entity_type = item.get("core_entity", {}).get("type")

        simplified_item = {
            "qualified_name": item.get("qualified_name")
        }

        # 1. Simplify 'retrieved_context'
        retrieved_context = item.get("retrieved_context", {})
        simplified_context = {}

        if retrieved_context:
            if entity_type == 'function':
                # For functions, simplify 'calls' and 'called_by' lists.
                for relation_type in ["calls", "called_by"]:
                    if relation_type in retrieved_context:
                        simplified_relations = [
                            {
                                "qualifiedName": rel.get("qualifiedName"),
                                "code_snippet": rel.get("code_snippet")
                            }
                            for rel in retrieved_context[relation_type]
                        ]
                        if simplified_relations:
                            simplified_context[relation_type] = simplified_relations
                
                if "total_relations" in retrieved_context:
                    simplified_context["total_relations"] = retrieved_context["total_relations"]

            elif entity_type == 'class':
                # Simplify 'inherits_from' and 'implements' relations
                for rel_type in ["inherits_from", "implements"]:
                    if rel_type in retrieved_context:
                        simplified_rel_list = []
                        for rel_item in retrieved_context[rel_type]:
                            new_rel_item = {"qualifiedName": rel_item.get("qualifiedName")}
                            if "contained_methods_snippets" in rel_item:
                                snippets = [
                                    {"code_snippet": s.get("code_snippet")}
                                    for s in rel_item["contained_methods_snippets"] if s.get("code_snippet")
                                ]
                                if snippets:
                                    new_rel_item["contained_methods_snippets"] = snippets
                            simplified_rel_list.append(new_rel_item)
                        if simplified_rel_list:
                            simplified_context[rel_type] = simplified_rel_list

                # Simplify 'instantiated_in' relation
                if "instantiated_in" in retrieved_context:
                    simplified_inst_list = [
                        {
                            "File": inst.get("File"),
                            "code_snippet": inst.get("code_snippet")
                        }
                        for inst in retrieved_context["instantiated_in"]
                    ]
                    if simplified_inst_list:
                        simplified_context["instantiated_in"] = simplified_inst_list
                
                # Copy other specified relations like 'subclasses' directly
                for rel_type in ["subclasses"]:
                    if rel_type in retrieved_context and retrieved_context[rel_type]:
                        simplified_context[rel_type] = retrieved_context[rel_type]

        if simplified_context:
            simplified_item["retrieved_context"] = simplified_context

        # 2. Simplify 'overridden_method_context' for functions
        if "overridden_method_context" in item:
            simplified_overridden = [
                {
                    "method_qualifiedName": ov.get("method_qualifiedName"),
                    "code_snippet": ov.get("code_snippet")
                }
                for ov in item["overridden_method_context"]
            ]
            if simplified_overridden:
                simplified_item["overridden_method_context"] = simplified_overridden

        simplified_list.append(simplified_item)

    return simplified_list


def load_commit_map(commit_map_file: str) -> dict:
    """
    Loads the commit SHA mapping from a JSONL file.
    Returns a dictionary mapping instance_id to commit_sha.
    """
    commit_map = {}
    try:
        with open(commit_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    commit_sha = data.get("commit_sha")

                    # Handle cases where commit_sha is nested inside subset_entry (user-specified format)
                    if not commit_sha and "subset_entry" in data:
                        subset_entry = data.get("subset_entry")
                        if isinstance(subset_entry, dict):
                            commit_sha = subset_entry.get("commit_sha")

                    if instance_id and commit_sha:
                        commit_map[instance_id] = commit_sha
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in commit map file: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Commit map file not found at '{commit_map_file}'.")
        return {}
    except Exception as e:
        print(f"Error reading commit map file: {e}")
        return {}
    return commit_map

def checkout_repo(repo_path: str, commit_sha: str) -> bool:
    """
    Checkouts the repository to the specified commit SHA.
    """
    try:
        # Check if the repo path exists
        if not os.path.exists(repo_path):
            print(f"Error: Repository path not found: {repo_path}")
            return False

        # Checkout the commit
        subprocess.run(['git', 'checkout', '-f', commit_sha], cwd=repo_path, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit {commit_sha} in {repo_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during git checkout: {e}")
        return False

def process_workflow(core_entities_file: str, commit_map_file: str, all_enre_report_dir: str, all_repo_dir: str, output_file: str, error_report_file: str):
    """
    Orchestrates the full workflow and saves all results to a single file.
    """
    print(f"Starting workflow with input file: {core_entities_file}")
    
    # Load commit map
    print(f"Loading commit map from: {commit_map_file}")
    commit_map = load_commit_map(commit_map_file)
    if not commit_map:
        print("Error: Failed to load commit map or map is empty. Aborting workflow.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Ensure the error report directory exists
    error_dir = os.path.dirname(error_report_file)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    # Open the output file for writing and the input JSONL file for reading.
    try:
        # First, count total lines for progress bar
        total_lines = 0
        try:
            with open(core_entities_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
        except Exception:
            pass # If counting fails, we just won't have a total for tqdm

        with open(output_file, 'w', encoding='utf-8') as f_out, \
             open(error_report_file, 'w', encoding='utf-8') as f_err, \
             open(core_entities_file, 'r', encoding='utf-8') as f_in:

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
                f_err.write(json.dumps(payload, ensure_ascii=False) + '\n')

            # Loop through each line in the JSONL file with progress bar
            for line in tqdm(f_in, total=total_lines, desc="Processing Entities", unit="line"):
                try:
                    instance = json.loads(line)
                except json.JSONDecodeError:
                    error_msg = f"Skipping malformed JSON line in {core_entities_file}: {line.strip()}"
                    print(f"Warning: {error_msg}")
                    continue

                instance_id = instance.get('instance_id')
                core_entities = instance.get('core_entities')
                
                # Get commit_sha from the loaded map
                commit_sha = commit_map.get(instance_id)
                
                instance_results = []

                if not instance_id:
                     error_msg = f"Skipping instance due to missing 'instance_id'. Instance: {instance}"
                     print(f"Warning: {error_msg}")
                     continue
                
                if core_entities is None: # core_entities can be empty list, but not None if we want to process it properly (or maybe empty list is also invalid? user said empty list is error)
                    # User requirement: "core_entities": [] is an error.
                    pass 

                if not core_entities and not isinstance(core_entities, list):
                     # Missing core_entities key or None
                     error_msg = "Missing 'core_entities' field."
                     print(f"Warning: {error_msg} Instance: {instance_id}")
                     write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                     continue
                
                if len(core_entities) == 0:
                     error_msg = "'core_entities' is empty."
                     print(f"Warning: {error_msg} Instance: {instance_id}")
                     write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                     continue

                if not commit_sha:
                     error_msg = f"Missing 'commit_sha' for instance_id: {instance_id} in commit map file."
                     print(f"Warning: {error_msg}")
                     write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                     continue

                # Derive project_root from instance_id
                parts = instance_id.split('_')
                if len(parts) < 3:
                    error_msg = f"Invalid instance_id format '{instance_id}'. Cannot derive project root."
                    print(f"Warning: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue
                
                repo_name = "_".join(parts[1:-1])
                project_root = os.path.join(all_repo_dir, repo_name)

                # print(f"--- Processing Instance: {instance_id} (Repo: {repo_name}, Commit: {commit_sha}) ---")

                # Checkout the repository to the specific commit
                if checkout_repo(project_root, commit_sha):
                    # print(f"Successfully checked out {commit_sha} in {project_root}")
                    pass
                else:
                    error_msg = f"Failed to checkout commit {commit_sha} in {project_root}"
                    print(f"Error: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue
                
                # Try to find the report file, first with the simple name, then with the old suffix.
                report_candidates = [
                    os.path.join(all_enre_report_dir, f"{instance_id}.json"),
                    os.path.join(all_enre_report_dir, f"{instance_id}_enre_report.json"),
                    os.path.join(all_enre_report_dir, f"{repo_name}-enre-out", f"{instance_id}_enre_report.json"),
                ]
                enre_report_file = next((path for path in report_candidates if os.path.isfile(path)), None)
                if enre_report_file is None:
                    error_msg = f"ENRE report for instance '{instance_id}' not found. Checked: {report_candidates}. Skipping."
                    print(f"Warning: {error_msg}")
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue

                # Load the ENRE report once per instance for efficiency, with encoding fallback.
                try:
                    with open(enre_report_file, 'r', encoding='utf-8') as f:
                        enre_data = json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(enre_report_file, 'r', encoding='latin-1') as f:
                            enre_data = json.load(f)
                    except (IOError, json.JSONDecodeError) as e:
                        error_msg = f"Error: Could not read or parse ENRE report '{enre_report_file}' (latin-1 fallback): {e}"
                        print(error_msg)
                        write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                        continue
                except (IOError, json.JSONDecodeError) as e:
                    error_msg = f"Error: Could not read or parse ENRE report '{enre_report_file}': {e}"
                    print(error_msg)
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", error_msg)
                    continue
                
                enre_variables = enre_data.get('variables', [])
                if not enre_variables:
                     print(f"Warning: No variables found in ENRE report for instance '{instance_id}'.")

                supported_entities_count = 0
                resolved_entities_count = 0
                resolved_entities_with_code_count = 0
                for entity in core_entities:
                    if entity.get('type') not in ['function', 'class']:
                        print(f"Warning: Skipping unsupported core entity type '{entity.get('type')}' for instance '{instance_id}'.")
                        continue

                    supported_entities_count += 1

                    qualified_names, warning_msg = generate_qualified_names(entity, enre_data)
                    if warning_msg:
                        print(f"Warning: {warning_msg}")
                    
                    # Log this specific entity failure? Maybe too verbose. Since we process it anyway, no need.
                    
                    # print(f"  - Analyzing entity candidates: {qualified_names}")
                    if not qualified_names:
                        instance_results.append({
                            "instance_id": instance_id,
                            "core_entity": entity,
                            "qualified_name": None,
                            "retrieved_context": {},
                            "overridden_method_context": []
                        })
                        continue

                    resolved_entities_count += 1
                    entity_has_related_code = False

                    for qualified_name in qualified_names:
                        overridden_method_context = []
                        if entity.get('type') == 'function':
                            # print(f"    - Checking for overridden method for '{qualified_name}'")
                            overridden_method_context = get_overridden_method_context(
                                function_qualified_name=qualified_name,
                                enre_data=enre_data,
                                project_root=project_root
                            )

                        relations = analyze_java_enre_report(
                            report_path=enre_report_file,
                            target_qualified_name=qualified_name,
                            enre_data=enre_data
                        )

                        if not relations:
                            # Even if no relations, record this resolved candidate.
                            instance_results.append({
                                "instance_id": instance_id,
                                "core_entity": entity,
                                "qualified_name": qualified_name,
                                "retrieved_context": {},
                                "overridden_method_context": overridden_method_context
                            })
                            continue

                        # print(f"    - Retrieving code context for '{qualified_name}'")
                        augmented_relations = retrieve_code_context(
                            entity_relations=relations[0],
                            enre_report_path=enre_report_file,
                            project_root=project_root
                        )

                        if augmented_relations:
                            entity_has_related_code = True
                            result_item = {
                                "instance_id": instance_id,
                                "core_entity": entity,
                                "qualified_name": qualified_name,
                                "retrieved_context": augmented_relations
                            }
                            if overridden_method_context:
                                result_item["overridden_method_context"] = overridden_method_context
                            instance_results.append(result_item)
                        else:
                            instance_results.append({
                                "instance_id": instance_id,
                                "core_entity": entity,
                                "qualified_name": qualified_name,
                                "retrieved_context": {},
                                "overridden_method_context": overridden_method_context
                            })

                    if entity_has_related_code:
                        resolved_entities_with_code_count += 1
                
                if instance_results:
                    simplified_data = simplify_results(instance_results)
                    output_line = {
                        "instance_id": instance_id,
                        "retrieved_code": simplified_data
                    }
                    f_out.write(json.dumps(output_line, ensure_ascii=False) + '\n')
                    # print(f"  -> Processed and wrote results for instance '{instance_id}'")

                if resolved_entities_count == 0:
                    detail = f"supported_core_entities={supported_entities_count}, resolved_entities=0"
                    write_instance_error(instance_id, "all_core_entities_not_retrieved", detail)
                elif resolved_entities_with_code_count == 0:
                    detail = f"resolved_entities={resolved_entities_count}, resolved_entities_with_code=0"
                    write_instance_error(instance_id, "retrieved_entities_without_related_code", detail)


            print(f"\nWorkflow complete. All results saved in: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{core_entities_file}'.")
    except IOError as e:
        print(f"\nError during file operations: {e}")

def main():
    parser = argparse.ArgumentParser(description='A workflow to process core entities, analyze relations, and retrieve code context.')
    parser.add_argument('--core_entities_file', required=True, help='Path to the JSON output file from extract_entities.py.')
    parser.add_argument('--commit_map_file', required=True, help='Path to the JSONL file containing instance_id and commit_sha mapping.')
    parser.add_argument('--all_enre_report_dir', required=True, help='Directory containing all ENRE-Java JSON reports for the project.')
    parser.add_argument('--all_repo_dir', required=True, help='The directory containing all Java repositories.')
    parser.add_argument('--output_file', required=True, help='Path to save the final combined JSON output file.')
    parser.add_argument('--error_report_file', required=True, help='Path to save the error report JSONL file.')
    
    args = parser.parse_args()

    process_workflow(args.core_entities_file, args.commit_map_file, args.all_enre_report_dir, args.all_repo_dir, args.output_file, args.error_report_file)

if __name__ == '__main__':
    main()
