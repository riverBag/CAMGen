# -*- coding: utf-8 -*-
import argparse
import json
import pickle
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.codebert_build_index import CodeBERTEncoder
from preprocess.bm25_indexing import tokenize_diff, BM25
from retriever.candidate_filter import HistoricalCandidateIndex

# =========================
# 工具函数
# =========================
def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_v, max_v = scores.min(), scores.max()
    if min_v == max_v:
        return np.zeros_like(scores)
    scores = (scores - min_v) / (max_v - min_v)
    return scores

def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            for k, v in r.items():
                if isinstance(v, (np.float32, np.float64, np.int32)):
                    r[k] = float(v)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# =========================
# 主逻辑
# =========================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrieve strictly earlier same-repository exemplars with BM25 and CodeBERT.")
    parser.add_argument("--queries", required=True, type=Path, help="Query commit JSONL.")
    parser.add_argument("--resource-dir", type=Path, default=Path(__file__).resolve().parents[3] / "outputs/view3", help="Directory containing the BM25 and dense indexes.")
    parser.add_argument("--output-dir", type=Path, help="Result directory; defaults to --resource-dir.")
    parser.add_argument("--model", default="microsoft/codebert-base", help="Use the same model as the CodeBERT index builder.")
    args = parser.parse_args(argv)
    import torch

    RESOURCE_DIR = args.resource_dir
    output_dir = args.output_dir or RESOURCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    BM25_INDEX_PATH = os.path.join(RESOURCE_DIR, "bm25_diff_index.pkl")
    CODEBERT_INDEX_PATH = os.path.join(RESOURCE_DIR, "codebert_diff_index.pkl")
    QUERY_JSONL_PATH = args.queries

    WEIGHTS = [(0,10), (3,7), (5,5), (7,3), (10,0)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing CodeBERT on {device}")
    cb_encoder = CodeBERTEncoder(args.model, device)
    cb_encoder.max_seq_length = 512

    print("[INFO] Loading indices...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25_model: BM25 = bm25_data["bm25"]
        bm25_items = bm25_data["raw_items"]

    with open(CODEBERT_INDEX_PATH, "rb") as f:
        cb_data = pickle.load(f)
        cb_embeddings = cb_data["embeddings"]

    if len(bm25_items) != len(cb_embeddings):
        raise ValueError("BM25 and CodeBERT indexes must have matching corpus positions")
    candidate_index = HistoricalCandidateIndex(bm25_items)

    # 加载查询
    queries = []
    with open(QUERY_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # =========================
    # 预计算 BM25 & CodeBERT score
    # =========================
    query_cache = {}
    print("[INFO] Precomputing BM25 & CodeBERT scores...")
    skipped_queries = 0
    for q_idx, q in enumerate(tqdm(queries)):
        q_sha = q["commit_sha"]
        q_diff = q.get("diff", "")

        # Restrict the candidate pool before scoring and normalization.
        candidates = candidate_index.candidate_indices(q)
        if not candidates:
            skipped_queries += 1
            continue

        # BM25 score
        q_tokens = tokenize_diff(q_diff)
        bm25_scores = np.array([bm25_model.score(q_tokens, idx) for idx in candidates])
        bm25_scores = min_max_normalize(bm25_scores)

        # CodeBERT dense score (余弦相似度)
        q_emb = cb_encoder.encode(q_diff).flatten()
        q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        cand_embs_subset = cb_embeddings[candidates]
        cand_embs_norm = cand_embs_subset / (np.linalg.norm(cand_embs_subset, axis=1, keepdims=True) + 1e-10)

        dense_scores = np.dot(cand_embs_norm, q_emb_norm)  # [-1,1]
        dense_scores = (dense_scores + 1.0) / 2.0          # [0,1]
        dense_scores = min_max_normalize(dense_scores)

        query_cache[q_idx] = {
            "indices": candidates,
            "bm25": bm25_scores,
            "dense": dense_scores
        }

    print(f"[INFO] Queries without strictly earlier same-repository candidates: {skipped_queries}")

    # =========================
    # 融合权重 + Top-1
    # =========================
    for w_bm25, w_dense in WEIGHTS:
        results = []
        for q_idx, q in enumerate(tqdm(queries, desc=f"BM25:{w_bm25} Dense:{w_dense}")):
            q_sha = q["commit_sha"]
            q_diff = q.get("diff", "")

            cache = query_cache.get(q_idx)
            if cache is None:
                continue

            # 权重融合
            final_scores = w_bm25 * cache["bm25"] + w_dense * cache["dense"]
            best_pos = int(np.argmax(final_scores))
            best_idx = cache["indices"][best_pos]
            best_item = bm25_items[best_idx]

            results.append({
                "query-owner": q["owner"],
                "query-repo": q["repo"],
                "query-date": q["date"],
                "query-sha": q_sha,
                "query-diff": q_diff,
                "retrieve-sha": best_item["commit_sha"],
                "retrieve-diff": best_item["diff"],
                "retrieve-message": best_item.get("message",""),
                "retrieve-owner": best_item["owner"],
                "retrieve-repo": best_item["repo"],
                "retrieve-date": best_item["date"],
                "bm25-score": float(cache["bm25"][best_pos]),
                "dense-score": float(cache["dense"][best_pos]),
                "score": float(final_scores[best_pos]),
                "weight": f"BM25:{w_bm25} Dense:{w_dense}"
            })

        out_file = os.path.join(output_dir, f"results_w_BM25_dense_{w_bm25}_{w_dense}_CodeBERT.jsonl")
        save_results(results, out_file)
        print(f"[INFO] Saved results to {out_file}")


if __name__ == "__main__":
    main()
