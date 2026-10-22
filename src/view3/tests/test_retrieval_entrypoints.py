"""Exercise every retrieval entry point without loading or downloading models."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


VIEW3 = Path(__file__).resolve().parents[1]
if str(VIEW3) not in sys.path:
    sys.path.insert(0, str(VIEW3))

from preprocess.bm25_indexing import BM25, tokenize_diff


class FakeCodeBERTEncoder:
    """A deterministic two-dimensional encoder; no model libraries required."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text):
        return np.array([[1.0, 0.0]])


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.tokenizer = self.tokenize

    @staticmethod
    def tokenize(*args, **kwargs):
        return {"input_ids": np.array([[1, 0]])}

    def encode(self, text, **kwargs):
        vector = np.array([1.0, 0.0])
        return vector if isinstance(text, str) else vector.reshape(1, -1)


@contextlib.contextmanager
def fake_model_dependencies():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    transformers.AutoModel = object
    codebert = types.ModuleType("embedding.codebert_build_index")
    codebert.CodeBERTEncoder = FakeCodeBERTEncoder
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, *args, **kwargs: iterable
    with mock.patch.dict(sys.modules, {
        "torch": torch,
        "sentence_transformers": sentence_transformers,
        "transformers": transformers,
        "embedding.codebert_build_index": codebert,
        "tqdm": tqdm,
    }):
        yield


def commit(owner, repo, sha, date, diff="target match"):
    return {
        "owner": owner,
        "repo": repo,
        "commit_sha": sha,
        "date": date,
        "diff": diff,
        "message": f"Message for {sha}",
    }


class RetrievalEntryPointTests(unittest.TestCase):
    def run_entrypoint(self, module_name, expected_file_count):
        # The primary and secondary queries deliberately share a SHA. A cache
        # indexed by SHA alone would use the secondary repository's candidates
        # when producing the primary query's result.
        items = [
            commit("apache", "primary", "future", "2024-01-03T00:00:00Z"),
            commit("apache", "primary", "equal", "2024-01-02T00:00:00Z"),
            commit("another-owner", "primary", "wrong-owner", "2024-01-01T00:00:00Z"),
            commit("apache", "primary", "shared-query-sha", "2024-01-01T00:00:00Z"),
            commit("apache", "primary", "prior-primary", "2024-01-01T00:00:00Z", "older code"),
            commit("apache", "secondary", "prior-secondary", "2024-01-01T00:00:00Z", "older code"),
            commit("apache", "secondary", "future-secondary", "2024-01-03T00:00:00Z"),
            commit("apache", "no-history", "future-only", "2024-01-03T00:00:00Z"),
        ]
        queries = [
            commit("apache", "primary", "shared-query-sha", "2024-01-02T00:00:00Z"),
            commit("apache", "no-history", "empty-query", "2024-01-02T00:00:00Z"),
            commit("apache", "secondary", "shared-query-sha", "2024-01-02T00:00:00Z"),
        ]
        embeddings = np.array([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0],
        ])
        bm25 = BM25([tokenize_diff(item["diff"]) for item in items])
        scored_indices = []
        original_score = BM25.score

        def record_score(model, query_tokens, document_index):
            scored_indices.append(document_index)
            return original_score(model, query_tokens, document_index)

        with tempfile.TemporaryDirectory(prefix="camgen-retrieval-test-") as temp:
            base = Path(temp)
            resource = base / "resource"
            (resource / "apachecm").mkdir(parents=True)
            working_directory = base / "retriever"
            working_directory.mkdir()
            for relative in ("query.jsonl", "apachecm/test.jsonl"):
                (resource / relative).write_text(
                    "".join(json.dumps(query) + "\n" for query in queries),
                    encoding="utf-8",
                )
            with (resource / "bm25_diff_index.pkl").open("wb") as stream:
                pickle.dump({"bm25": bm25, "raw_items": items}, stream)
            for name in ("codebert_diff_index.pkl", "jina_diff_index.pkl", "jina_diff_index_fixed.pkl"):
                with (resource / name).open("wb") as stream:
                    pickle.dump({"embeddings": embeddings, "raw_items": items}, stream)

            old_directory = Path.cwd()
            old_module = sys.modules.pop(module_name, None)
            try:
                os.chdir(working_directory)
                with fake_model_dependencies(), mock.patch.object(BM25, "score", record_score):
                    module = importlib.import_module(module_name)
                    with contextlib.redirect_stdout(io.StringIO()):
                        module.main(["--resource-dir", str(resource), "--queries", str(resource / "query.jsonl")])
            finally:
                os.chdir(old_directory)
                sys.modules.pop(module_name, None)
                if old_module is not None:
                    sys.modules[module_name] = old_module

            # This checks the restriction before scoring, rather than accepting
            # a post-ranking filter that still scores or normalizes future data.
            self.assertEqual(scored_indices, [4, 5])
            result_files = sorted(resource.glob("results_w_BM25_dense_*.jsonl"))
            self.assertEqual(len(result_files), expected_file_count)
            for result_file in result_files:
                with self.subTest(entrypoint=module_name, output=result_file.name):
                    rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(rows), 2)
                    self.assertEqual({row["query-repo"] for row in rows}, {"primary", "secondary"})
                    for row in rows:
                        self.assertEqual(row["query-sha"], "shared-query-sha")
                        self.assertEqual(row["retrieve-sha"], f"prior-{row['query-repo']}")
                        self.assertEqual(row["query-owner"], "apache")
                        self.assertEqual(row["retrieve-owner"], row["query-owner"])
                        self.assertEqual(row["retrieve-repo"], row["query-repo"])
                        self.assertEqual(row["query-date"], "2024-01-02T00:00:00Z")
                        self.assertEqual(row["retrieve-date"], "2024-01-01T00:00:00Z")

    def test_jina_filters_before_scoring_and_isolates_query_cache(self):
        self.run_entrypoint("retriever.bm25_jina", expected_file_count=1)

    def test_codebert_filters_before_scoring_and_isolates_query_cache(self):
        self.run_entrypoint("retriever.bm25_codebert", expected_file_count=5)

    def test_combined_entrypoint_filters_both_models(self):
        self.run_entrypoint("retriever.retrieve_query", expected_file_count=10)


if __name__ == "__main__":
    unittest.main()
