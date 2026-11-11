"""
解析 ENRE 报告 JSON：过滤出 Class/Function 实体，并从源码中提取代码块写入 code 字段。

用法示例：
  # 解析 cluster_demo 的报告（报告里 File 为 "cluster_demo.py" 等，base_dir 为项目根目录）
  python parse_enre_report.py cluster_demo.py-report-enre.json -o filtered.json -b .

  # 解析带 repo-commit 前缀的报告（File 形如 "airflow-ccd2b88a.../path/to/file.py"）
  python parse_enre_report.py airflow-xxx-report-enre.json -o filtered.json -b ./repos/

  # 不传 -b 时，默认用当前目录 . 作为 base_dir（与 -b . 等价）
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PACKAGE_ROOT / "outputs" / "view2" / "filtered_entities.json"
DEFAULT_BASE_DIR = Path("./repos/")
KEEP_CATEGORIES = {"Class", "Function"}


def load_report(report_path: Path) -> Mapping:
    """加载 ENRE 报告文件"""
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_variables(variables: Iterable[Mapping]) -> list[Mapping]:
    return [v for v in variables if v.get("category") in KEEP_CATEGORIES]


def extract_code(file_path: Path, start_line: int) -> Tuple[str, int]:
    """
    根据起始行提取同级缩进的代码块，返回代码文本和结束行号（1-based）。
    若无法确定，返回空串与原值 -1。
    """
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return "", -1

    if start_line <= 0 or start_line > len(lines):
        return "", -1

    start_idx = start_line - 1
    indent_len = len(lines[start_idx]) - len(lines[start_idx].lstrip(" "))
    end_idx = start_idx

    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if stripped == "":
            end_idx = i
            continue

        current_indent = len(line) - len(line.lstrip(" "))
        # 遇到同级或更外层的非空行，视为下一个代码块的开始，停止
        if current_indent <= indent_len:
            break

        end_idx = i

    code_lines = lines[start_idx : end_idx + 1]
    # 去除尾部纯空格/缩进行
    while code_lines and code_lines[-1].strip() == "":
        code_lines.pop()

    code_block = "\n".join(code_lines)
    return code_block, end_idx + 1


def enrich_entities(entities: Iterable[Mapping], repos_base_dir: Path) -> list[Mapping]:
    """
    Enrich 实体，添加代码内容
    
    Args:
        entities: 实体列表
        repos_base_dir: REPOS_BASE_DIR（如 ./repos/），File 字段已经包含 {repo}-{commit_sha} 前缀
    """
    enriched = []
    for entity in entities:
        file_rel = entity.get("File")
        start_line = entity.get("location", {}).get("startLine", -1)
        code_text, end_line = ("", -1)
        if file_rel and start_line != -1:
            # File 字段已经包含 {repo}-{commit_sha} 前缀，直接使用 REPOS_BASE_DIR
            file_path = repos_base_dir / file_rel
            code_text, end_line = extract_code(file_path, start_line)

        new_entity = dict(entity)
        location = dict(new_entity.get("location", {}))
        location["endLine"] = end_line
        new_entity["location"] = location
        new_entity["code"] = code_text
        enriched.append(new_entity)
    return enriched


def save_report(path: Path, report: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="解析 ENRE 报告：过滤 Class/Function 并提取代码块")
    parser.add_argument("input", help="ENRE 报告 JSON 路径")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 路径（默认: 脚本内 OUTPUT_PATH）")
    parser.add_argument("-b", "--base-dir", default=None, help="源码根目录，用于拼接 File 路径（默认: . ）")
    args = parser.parse_args()

    report_path = Path(args.input)
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    base_dir = Path(args.base_dir).resolve() if args.base_dir is not None else Path(".").resolve()

    if not report_path.exists():
        print(f"错误: 输入文件不存在: {report_path}")
        return

    report = load_report(report_path)
    filtered_variables = filter_variables(report.get("variables", []))
    enriched_entities = enrich_entities(filtered_variables, base_dir)
    filtered_report = {k: v for k, v in report.items() if k not in ("cells", "variables")}
    filtered_report["entities"] = enriched_entities
    save_report(output_path, filtered_report)
    print(f"过滤完成，输出文件: {output_path}")


if __name__ == "__main__":
    main()
