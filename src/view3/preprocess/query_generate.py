import argparse
import json
import random
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parents[3] / "outputs/view3/query.jsonl"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42  # 可选：保证结果可复现

def sample_jsonl(input_file, output_file, sample_size, seed=None):
    if seed is not None:
        random.seed(seed)

    # 读取所有数据
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    total = len(lines)
    if sample_size > total:
        raise ValueError(f"样本数 {sample_size} 大于数据总量 {total}")

    # 随机抽样
    sampled = random.sample(lines, sample_size)

    # 写入新 jsonl 文件
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已从 {total} 条数据中随机抽取 {sample_size} 条，输出至 {output_file}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Sample query commits reproducibly from a JSONL file.")
    parser.add_argument("input", help="Query commit JSONL to sample.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error("--sample-size must be nonnegative")
    sample_jsonl(args.input, args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
