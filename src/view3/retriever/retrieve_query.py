import argparse
import json
import pickle
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
# Add the project root to sys.path to allow imports from sibling directories 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.codebert_build_index import CodeBERTEncoder
from preprocess.bm25_indexing import tokenize_diff, BM25
from retriever.candidate_filter import HistoricalCandidateIndex

# ==========================================
# Main Retrieval Logic
# ==========================================

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize the scores using min-max normalization.
    """
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def convert_numpy_types(result):
    """
    Convert any numpy types (like np.float32, np.int32) to standard Python types (like float, int).
    """
    return {key: (value.item() if isinstance(value, (np.float32, np.int32, np.float64)) else value) for key, value in result.items()}

def save_results(results, output_file):
    """
    Save the results to a JSONL file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            result = convert_numpy_types(result)  # Ensure numpy types are converted
            f.write(json.dumps(result) + "\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare CodeBERT/Jina hybrid retrieval over strictly earlier same-repository history.")
    parser.add_argument("--queries", required=True, type=Path, help="Query commit JSONL.")
    parser.add_argument("--resource-dir", type=Path, default=Path(__file__).resolve().parents[3] / "outputs/view3", help="Directory containing all three indexes.")
    parser.add_argument("--output-dir", type=Path, help="Result directory; defaults to --resource-dir.")
    parser.add_argument("--codebert-model", default="microsoft/codebert-base")
    parser.add_argument("--jina-model", default="jinaai/jina-embeddings-v2-base-code")
    args = parser.parse_args(argv)
    import torch
    from sentence_transformers import SentenceTransformer

    RESOURCE_DIR = args.resource_dir
    output_dir = args.output_dir or RESOURCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    BM25_INDEX_PATH = os.path.join(RESOURCE_DIR, "bm25_diff_index.pkl")
    CODEBERT_INDEX_PATH = os.path.join(RESOURCE_DIR, "codebert_diff_index.pkl")
    JINA_INDEX_PATH = os.path.join(RESOURCE_DIR, "jina_diff_index.pkl")
    QUERY_JSONL_PATH = args.queries

    # Weights configuration (BM25 : Dense)
    WEIGHTS = [
        (0, 10),
        (3, 7),
        (5, 5),
        (7, 3),
        (10, 0)
    ]



    # Initialize Models for Query Encoding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing CodeBERT model on {device}...")
    cb_encoder = CodeBERTEncoder(args.codebert_model, device)

    print(f"[INFO] Initializing Jina model on {device}...")
    jina_encoder = SentenceTransformer(args.jina_model, trust_remote_code=True)
    jina_encoder.max_seq_length = 4096

    print("[INFO] Loading indices...")

    # Load BM25 Index
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25_model: BM25 = bm25_data["bm25"]
        bm25_items = bm25_data["raw_items"]

    # Load CodeBERT Index
    with open(CODEBERT_INDEX_PATH, "rb") as f:
        cb_data = pickle.load(f)
        cb_embeddings = cb_data["embeddings"]

    # Load Jina Index
    with open(JINA_INDEX_PATH, "rb") as f:
        jina_data = pickle.load(f)
        jina_embeddings = jina_data["embeddings"]

    # Verify alignment
    if not len(bm25_items) == len(cb_embeddings) == len(jina_embeddings):
        raise ValueError("BM25 and dense indexes must have matching corpus positions")
    candidate_index = HistoricalCandidateIndex(bm25_items)

    print(f"[INFO] Loaded indices. Total documents: {len(bm25_items)}")

    # Load Queries
    queries = []
    with open(QUERY_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"[INFO] Loaded {len(queries)} queries.")

    # Initialize the empty candidate counter
    empty_candidate_counter = 0

    # Iterate over queries
    results_log = {
        (model, w_bm25, w_dense): []
        for model in ("CodeBERT", "Jina")
        for w_bm25, w_dense in WEIGHTS
    }

    for q_idx, query in enumerate(tqdm(queries, desc="Processing Queries")):
        q_repo = query.get("repo")
        q_sha = query.get("commit_sha")
        q_diff = query.get("diff", "")

        # 1. Restrict the pool before scoring and normalization.
        candidate_indices = candidate_index.candidate_indices(query)
        if not candidate_indices:
            empty_candidate_counter += 1
            print(f"[INFO] Query {q_idx}: no strictly earlier same-repository candidates")
            continue
        print(f"[INFO] Query {q_idx}: Repo={q_repo}, SHA={q_sha}, Candidates={len(candidate_indices)}")
        # 2. Compute BM25 Scores for Candidates
        q_tokens = tokenize_diff(q_diff)
        bm25_scores = []
        for idx in candidate_indices:
            bm25_scores.append(bm25_model.score(q_tokens, idx))
        bm25_scores = np.array(bm25_scores)

        # Normalize BM25 Scores (Min-Max)
        bm25_scores_norm = min_max_normalize(bm25_scores)

        # 3. Compute Dense Scores & Hybrid Fusion
        cb_q_emb = cb_encoder.encode(q_diff)  # Shape (1, H)
        cb_cand_embs = cb_embeddings[candidate_indices]  # (M, H)
        cb_scores = np.dot(cb_cand_embs, cb_q_emb.T).flatten()
        cb_scores_norm = min_max_normalize(cb_scores)

        jina_q_emb = jina_encoder.encode(q_diff, normalize_embeddings=True)
        jina_cand_embs = jina_embeddings[candidate_indices]
        jina_scores = np.dot(jina_cand_embs, jina_q_emb.T).flatten()
        jina_scores_norm = min_max_normalize(jina_scores)

        # 4. Apply Weights and Find Best Match
        def process_fusion(dense_name, dense_scores_norm):
            print(f"[INFO] Query Model={dense_name}")
            for (w_bm25, w_dense) in WEIGHTS:
                final_scores = (w_bm25 * bm25_scores_norm) + (w_dense * dense_scores_norm)
                best_pos = int(np.argmax(final_scores))
                best_item = bm25_items[candidate_indices[best_pos]]
                results_log[(dense_name, w_bm25, w_dense)].append({
                    "query-owner": query["owner"],
                    "query-repo": q_repo,
                    "query-date": query["date"],
                    "query-diff": q_diff,
                    "query-sha": q_sha,
                    "retrieve-owner": best_item["owner"],
                    "retrieve-repo": best_item["repo"],
                    "retrieve-date": best_item["date"],
                    "retrieve-diff": best_item["diff"],
                    "retrieve-message": best_item.get("message", ""),
                    "retrieve-sha": best_item["commit_sha"],
                    "bm25-score": bm25_scores_norm[best_pos],
                    "module-score": dense_scores_norm[best_pos],
                    "score": final_scores[best_pos],
                    "model": dense_name,
                    "weight": f"BM25:{w_bm25} Dense:{w_dense}",
                })

        # Run for CodeBERT
        process_fusion("CodeBERT", cb_scores_norm)

        # Run for Jina
        process_fusion("Jina", jina_scores_norm)

        print("-" * 40)

    # After processing all queries, save the results
    print(f"[INFO] Total processed queries: {len(queries) - empty_candidate_counter}")
    print(f"[INFO] Total skipped entries (empty candidates): {empty_candidate_counter}")
    print("[INFO] Saving results...")

    # Save results for each weight combination
    for (w_bm25, w_dense) in WEIGHTS:
        codebert_results = results_log[("CodeBERT", w_bm25, w_dense)]
        jina_results = results_log[("Jina", w_bm25, w_dense)]

        # Save CodeBERT results
        codebert_output_file = os.path.join(output_dir, f"results_w_BM25_dense_{w_bm25}_{w_dense}_CodeBERT.jsonl")
        save_results(codebert_results, codebert_output_file)
        print(f"[INFO] Saved CodeBERT results to {codebert_output_file}")

        # Save Jina results
        jina_output_file = os.path.join(output_dir, f"results_w_BM25_dense_{w_bm25}_{w_dense}_Jina.jsonl")
        save_results(jina_results, jina_output_file)
        print(f"[INFO] Saved Jina results to {jina_output_file}")


if __name__ == "__main__":
    main()
