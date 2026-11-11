# -*- coding: utf-8 -*-
"""
Build diff embedding index using local jina-embeddings-v2-base-code
Only uses `diff` field from full.jsonl
"""

import argparse
import json
import sys
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm


#
# Config
#
MODEL_PATH = "jinaai/jina-embeddings-v2-base-code"
OUTPUT_PATH = Path(__file__).resolve().parents[3] / "outputs/view3/jina_diff_index.pkl"
BATCH_SIZE = 2          # 本地模型建议稍小
MAX_SEQ_LENGTH = 4096     # 强烈建议裁剪，避免你之前看到的 token 爆炸

#
# Load JSONL
#
def load_diffs(jsonl_path):
    diffs = []
    raw_items = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error at line {line_no}: {e}")

            diff = item.get("diff", "")
            diffs.append(diff)
            raw_items.append(item)

    return diffs, raw_items


#
# Main (Index Builder)
#
def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a Jina index over historical commit diffs.")
    parser.add_argument("corpus", help="Historical commit JSONL, in the same row order as the BM25 corpus.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL_PATH, help="Hugging Face model ID or local model directory.")
    args = parser.parse_args(argv)
    from sentence_transformers import SentenceTransformer

    jsonl_path = args.corpus
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading diffs...")
    diffs, raw_items = load_diffs(jsonl_path)
    print(f"[INFO] Loaded {len(diffs)} diffs")

    print(f"[INFO] Loading embedding model from: {args.model}")

    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    print("[INFO] Encoding diffs (local model)...")
    all_vectors = []

    for i in tqdm(range(0, len(diffs), BATCH_SIZE)):
        batch = diffs[i : i + BATCH_SIZE]

        vectors = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=True,   # 内置 L2 normalize
        )

        all_vectors.append(vectors)

    embeddings = np.vstack(all_vectors)
    print(f"[INFO] Total embeddings shape: {embeddings.shape}")

    print(f"[INFO] Saving index to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "raw_items": raw_items,
                "model_path": args.model,
                "max_seq_length": MAX_SEQ_LENGTH,
            },
            f,
        )

    print("[INFO] Index saved successfully.")


if __name__ == "__main__":
    main()
