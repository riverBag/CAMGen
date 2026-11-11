# -*- coding: utf-8 -*-
"""
Fix NaN embeddings in jina_diff_index.pkl by re-embedding entries
from jina_nan_items.jsonl and replacing by commit_sha.
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
RESOURCE_DIR = Path(__file__).resolve().parents[3] / "outputs/view3"
OLD_INDEX_PATH = RESOURCE_DIR / "jina_diff_index.pkl"
OUTPUT_PATH = RESOURCE_DIR / "jina_diff_index_fixed.pkl"

BATCH_SIZE = 2
MAX_SEQ_LENGTH = 4096


#
# Load JSONL (NaN items)
#
def load_nan_items(jsonl_path):
    diffs = []
    items = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error at line {line_no}: {e}")

            if "commit_sha" not in item:
                raise RuntimeError(f"Missing commit_sha at line {line_no}")

            diff = item.get("diff", "")
            diffs.append(diff)
            items.append(item)

    return diffs, items


#
# Main
#
def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-embed specified NaN records in a Jina index.")
    parser.add_argument("nan_items", help="JSONL containing the affected corpus records.")
    parser.add_argument("--input-index", type=Path, default=OLD_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args(argv)
    from sentence_transformers import SentenceTransformer

    nan_jsonl_path = args.nan_items
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load old index ----------
    print("[INFO] Loading original jina_diff_index.pkl...")
    with open(args.input_index, "rb") as f:
        index_data = pickle.load(f)

    embeddings = index_data["embeddings"]
    raw_items = index_data["raw_items"]

    emb_dim = embeddings.shape[1]

    # commit_sha -> index
    sha2idx = {}
    for idx, item in enumerate(raw_items):
        sha = item.get("commit_sha")
        if sha:
            sha2idx[sha] = idx

    # ---------- 2. Load NaN items ----------
    print("[INFO] Loading NaN items...")
    diffs, nan_items = load_nan_items(nan_jsonl_path)
    print(f"[INFO] Loaded {len(diffs)} NaN diffs")

    # ---------- 3. Load model ----------
    print(f"[INFO] Loading embedding model from: {args.model}")
    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    # ---------- 4. Re-embed ----------
    print("[INFO] Re-encoding NaN diffs...")
    new_vectors = {}

    for i in tqdm(range(0, len(diffs), BATCH_SIZE)):
        batch_diffs = diffs[i : i + BATCH_SIZE]
        batch_items = nan_items[i : i + BATCH_SIZE]

        vectors = model.encode(
            batch_diffs,
            batch_size=len(batch_diffs),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        for item, vec in zip(batch_items, vectors):
            sha = item["commit_sha"]

            if np.isnan(vec).any():
                raise RuntimeError(f"Re-embedded vector still NaN: {sha}")

            if vec.shape[0] != emb_dim:
                raise RuntimeError("Embedding dimension mismatch")

            if sha not in sha2idx:
                raise RuntimeError(f"commit_sha not found in original index: {sha}")

            new_vectors[sha] = vec

    # ---------- 5. Replace ----------
    print("[INFO] Replacing embeddings in original index...")
    for sha, vec in new_vectors.items():
        idx = sha2idx[sha]
        embeddings[idx] = vec

    print(f"[INFO] Replaced {len(new_vectors)} embeddings")

    # ---------- 6. Save ----------
    print(f"[INFO] Saving fixed index to {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(index_data, f)

    print("[INFO] Done. NaN embeddings fixed.")


if __name__ == "__main__":
    main()
