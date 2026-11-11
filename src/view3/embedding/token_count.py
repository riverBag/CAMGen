import argparse
import json
import random
from tqdm import tqdm

SAMPLE_SIZE = 2000     # 抽样条数，1000~3000 都可以
SEED = 42

def count_tokens(text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate diff token counts in a commit JSONL file.")
    parser.add_argument("input", help="Commit JSONL containing diff fields.")
    args = parser.parse_args(argv)
    diffs = []

    print("[INFO] Loading diffs...")
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            diff = item.get("diff", "")
            if diff:
                diffs.append(diff)

    total = len(diffs)
    print(f"[INFO] Total diffs: {total}")
    if not diffs:
        raise ValueError("No nonempty diffs were found in the input file")

    random.seed(SEED)
    sample = diffs if total <= SAMPLE_SIZE else random.sample(diffs, SAMPLE_SIZE)

    print(f"[INFO] Sampling {len(sample)} diffs for token estimation...")

    token_counts = []
    for d in tqdm(sample):
        token_counts.append(count_tokens(d))

    token_counts.sort()

    avg = sum(token_counts) / len(token_counts)
    p95 = token_counts[int(len(token_counts) * 0.95)]
    max_v = token_counts[-1]

    estimated_total = avg * total

    print("\n===== Token Estimation Result =====")
    print(f"Average tokens / diff : {avg:.2f}")
    print(f"P95 tokens / diff     : {p95}")
    print(f"Max tokens / diff     : {max_v}")
    print(f"Estimated TOTAL tokens: {int(estimated_total):,}")
    print("==================================")


if __name__ == "__main__":
    main()
