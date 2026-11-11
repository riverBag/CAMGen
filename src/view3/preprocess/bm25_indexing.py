# -*- coding: utf-8 -*-
"""
BM25-based diff retrieval
- Only uses `diff` field from jsonl
- k1 = 1.2, b = 0.75
- Input: full.jsonl
- Output: Top-K similar entries by diff
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple


#
# Configuration
#
K1 = 1.2
B = 0.75
TOP_K = 5


#
# Diff Tokenizer
#
def tokenize_diff(diff: str) -> List[str]:
    """
    Tokenize diff text for BM25.
    - Remove diff metadata
    - Keep identifiers / code tokens
    """
    if not diff:
        return []

    diff = diff.lower()

    # Remove diff headers / metadata
    diff = re.sub(r'diff --git.*', ' ', diff)
    diff = re.sub(r'index .*', ' ', diff)
    diff = re.sub(r'@@.*@@', ' ', diff)
    diff = re.sub(r'\+\+\+|---', ' ', diff)

    # Extract identifier-like tokens
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', diff)
    return tokens


#
# Load JSONL Corpus
#
def load_corpus(jsonl_path: str):
    """
    Load jsonl file and build diff token corpus.
    Returns:
        corpus_tokens: List[List[str]]
        raw_items: List[dict]
    """
    corpus_tokens = []
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
            tokens = tokenize_diff(diff)

            corpus_tokens.append(tokens)
            raw_items.append(item)

    return corpus_tokens, raw_items


#
# BM25 Implementation
#
class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)

        if self.N == 0:
            raise ValueError("Corpus is empty")

        # Document lengths
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.N

        # Term frequencies per document
        self.tf = []
        # Document frequency
        self.df = defaultdict(int)

        for doc in corpus:
            freq = Counter(doc)
            self.tf.append(freq)
            for term in freq:
                self.df[term] += 1

        # Inverse document frequency
        self.idf = {
            term: math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def score(self, query_tokens: List[str], doc_index: int) -> float:
        """
        Compute BM25 score for a single document.
        """
        score = 0.0
        doc_tf = self.tf[doc_index]
        dl = self.doc_len[doc_index]

        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]
            idf = self.idf.get(term, 0.0)

            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1.0)) / denom

        return score

    def retrieve(
        self, query_tokens: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents by BM25 score.
        Returns: [(doc_index, score), ...]
        """
        scored = []

        for i in range(self.N):
            s = self.score(query_tokens, i)
            if s > 0.0:
                scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]



import pickle

def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a BM25 index over a historical commit corpus.")
    parser.add_argument("corpus", help="Historical commit JSONL; preserve row order across all indexes.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "outputs/view3/bm25_diff_index.pkl")
    args = parser.parse_args(argv)
    jsonl_path = args.corpus
    index_path = args.output
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading corpus...")
    corpus, raw_items = load_corpus(jsonl_path)
    print(f"[INFO] Loaded {len(corpus)} entries")

    print("[INFO] Building BM25 index...")
    bm25 = BM25(corpus, k1=K1, b=B)
    print("[INFO] BM25 index built")

    print("[INFO] Saving index...")
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "raw_items": raw_items
            },
            f
        )

    print(f"[INFO] Index saved to {index_path}")


if __name__ == "__main__":
    main()

    # python bm25_indexing.py full.jsonl

