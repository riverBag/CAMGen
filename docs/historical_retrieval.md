# 🕰️ Historical Exemplar View

The Historical Exemplar View (`view3`) combines BM25 similarity with a dense diff encoder. Retrieval selects an exemplar from the same `owner` and `repo`, excludes the query SHA, and considers only commits strictly earlier than the query. This restriction is applied before scoring and normalization.

## 🧰 Environment

From the replication package root:

```bash
conda env create -f environments/historical_retrieval.yml
conda activate camgen-history
```

Dense indexing and retrieval require PyTorch, Transformers, Sentence Transformers, and downloaded model weights. The defaults are `microsoft/codebert-base` and `jinaai/jina-embeddings-v2-base-code`. Use `--model` (or the combined retriever's `--codebert-model` and `--jina-model`) to select a local model directory, consistently for indexing and querying. Jina is loaded with `trust_remote_code=True`. The optional token estimation utility uses `tiktoken`.

## 📚 Input Data

Historical commit corpora, pretrained model weights, and generated retrieval indexes are not bundled. Obtain the historical corpus separately and place each language's records at, for example, `dataset/history/java.jsonl`. The supplied evaluation commits in `dataset/full/commits/` and `dataset/ablation/commits/` can be used as queries; they do not replace the historical corpus.

Each historical record and query must contain:

```json
{"owner":"apache","repo":"example","commit_sha":"a-full-commit-sha","date":"2024-01-01T00:00:00Z","diff":"diff --git ...","message":"Describe the change"}
```

`message` is used as the retrieved exemplar message; query reference messages are not used for ranking. Dates must be ISO 8601 strings with a timezone. Before indexing, exclude all 7,522 evaluation commits in `dataset/full/commits/` from the historical corpus, as required by the paper's evaluation protocol. The retrievers apply the earlier/same-repository restriction at query time; evaluation-set exclusion is a corpus preparation step. Build all indexes from the **same corpus file in exactly the same row order**: dense vectors are matched to BM25 records by position. Use a separate resource directory for each language to prevent one language's indexes from overwriting another's.

## ⚙️ Build Indexes and Retrieve Exemplars

The following Java example runs from the replication package root. Replace `java` with `cpp` or `python` for the other languages.

```bash
python src/view3/preprocess/bm25_indexing.py dataset/history/java.jsonl --output outputs/view3/java/bm25_diff_index.pkl
python src/view3/embedding/jina_build_diff_index.py dataset/history/java.jsonl --output outputs/view3/java/jina_diff_index.pkl
python src/view3/retriever/bm25_jina.py --queries dataset/full/commits/java.jsonl --resource-dir outputs/view3/java
```

The Jina retriever uses equal BM25 and dense weights `(5, 5)` and writes `results_w_BM25_dense_5_5_Jina.jsonl` in the resource directory. Queries without any eligible historical candidate are skipped; the script reports their count.

For the CodeBERT variant:

```bash
python src/view3/embedding/codebert_build_index.py dataset/history/java.jsonl --output outputs/view3/java/codebert_diff_index.pkl
python src/view3/retriever/bm25_codebert.py --queries dataset/full/commits/java.jsonl --resource-dir outputs/view3/java
```

The CodeBERT retriever evaluates BM25/dense weights `(0, 10)`, `(3, 7)`, `(5, 5)`, `(7, 3)`, and `(10, 0)`. Once all three indexes exist, run both encoders over those weight settings with:

```bash
python src/view3/retriever/retrieve_query.py --queries dataset/full/commits/java.jsonl --resource-dir outputs/view3/java
```

Use `--output-dir` to write retrieval results separately from indexes. If no resource directory is given, indexes and results default to `outputs/view3/` resolved from the package location, independent of the current working directory. Explicit relative input/output paths are interpreted relative to the current working directory.

The retrieval output contains query and exemplar metadata, retrieved diffs and messages, and similarity scores. For full-set task generation, copy the locally generated result file to `dataset/full/contexts/view3/<language>.jsonl`; the task generator accepts its hyphenated `query-sha`, `retrieve-diff`, and `retrieve-message` keys. Ablation exports additionally require `common_id` joins to the fixed selection and belong in `dataset/ablation/contexts/view3/<language>.jsonl`. Generate these inputs before running a task configuration that uses View3; precomputed context files are not distributed with this package.

## 🛠️ Optional Utilities

Sample a fresh query subset (this does not reproduce the fixed ablation IDs):

```bash
python src/view3/preprocess/query_generate.py dataset/full/commits/java.jsonl --sample-size 1000 --seed 42 --output outputs/view3/java/query.jsonl
python src/view3/embedding/token_count.py dataset/history/java.jsonl
```

If a Jina index contains NaN vectors, prepare a JSONL file containing the affected corpus records and repair it:

```bash
python src/view3/embedding/jina_build_diff_index_fix_nan.py outputs/view3/java/nan_items.jsonl --input-index outputs/view3/java/jina_diff_index.pkl --output outputs/view3/java/jina_diff_index_fixed.pkl
python src/view3/retriever/bm25_jina.py --queries dataset/full/commits/java.jsonl --resource-dir outputs/view3/java --jina-index outputs/view3/java/jina_diff_index_fixed.pkl
```

The repair utility matches records by `commit_sha`; use a single repository corpus when repairing if the same SHA appears in more than one repository.

## 🧪 Regression Checks

The supplied tests use deterministic fake encoders, require only NumPy beyond the Python standard library, and do not download model weights:

```bash
python -m unittest discover -s src/view3/tests -v
```

They check all three retrieval entry points, exclusion of equal/future commits, owner/repository boundaries, timestamp handling, and separate query caches when two repositories share a SHA. Every command above also supports `--help` without loading model weights.
