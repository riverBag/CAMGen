import argparse
import json
import os
from pathlib import Path


def has_useful_context(entity: dict) -> bool:
    ctx = entity.get("retrieved_context")
    if not isinstance(ctx, dict) or not ctx:
        return False

    for key, value in ctx.items():
        if key == "total_relations":
            continue
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True

    total_relations = ctx.get("total_relations")
    return isinstance(total_relations, (int, float)) and total_relations > 0


def classify_entity(entity) -> tuple[bool, str | None]:
    if not isinstance(entity, dict):
        return False, "not_a_dict"
    if not entity.get("qualified_name"):
        return False, "null_qualified_name"
    if not has_useful_context(entity):
        return False, "no_useful_related_code"
    return True, None


def default_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.with_name(path.stem + ".cleaned" + path.suffix))


def clean_file(input_path: str, output_path: str, report_path: str | None = None) -> dict:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = {
        "input_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "original_records": 0,
        "cleaned_records": 0,
        "dropped_records_all_entities_no_result": 0,
        "records_with_some_entities_removed_but_kept": 0,
        "malformed_records": 0,
        "original_entities": 0,
        "kept_entities": 0,
        "removed_entities": 0,
        "removed_null_qualified_name": 0,
        "removed_no_useful_related_code": 0,
        "removed_not_a_dict": 0,
    }

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_records"] += 1
                continue

            stats["original_records"] += 1
            original_entities = record.get("retrieved_code") or []
            if not isinstance(original_entities, list):
                original_entities = []

            kept_entities = []
            removed_reasons = []

            for entity in original_entities:
                stats["original_entities"] += 1
                keep, reason = classify_entity(entity)
                if keep:
                    kept_entities.append(entity)
                    stats["kept_entities"] += 1
                else:
                    removed_reasons.append(reason)
                    stats["removed_entities"] += 1
                    if reason == "null_qualified_name":
                        stats["removed_null_qualified_name"] += 1
                    elif reason == "no_useful_related_code":
                        stats["removed_no_useful_related_code"] += 1
                    elif reason == "not_a_dict":
                        stats["removed_not_a_dict"] += 1

            if kept_entities:
                if removed_reasons and len(original_entities) > 1:
                    stats["records_with_some_entities_removed_but_kept"] += 1
                record["retrieved_code"] = kept_entities
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["cleaned_records"] += 1
            else:
                stats["dropped_records_all_entities_no_result"] += 1

    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean C++ related_code JSONL by removing unresolved entities "
            "and entities without useful related code."
        )
    )
    parser.add_argument("--input", required=True, help="Input related_code_cpp JSONL path.")
    parser.add_argument("--output", default=None, help="Cleaned output JSONL path.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.input)
    report_path = args.report or output_path + ".report.json"
    stats = clean_file(args.input, output_path, report_path)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
