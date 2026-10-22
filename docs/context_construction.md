# 🧩 Context Construction

This guide describes the context construction scripts shipped with CAMGen. Run commands from the replication package root. The package provides commit data and construction code; generate contexts locally before running a context-based configuration. Construction scripts write intermediate artifacts under `outputs/`, while commit inputs remain under `dataset/`. The [main README](../README.md) describes task generation and inference.

## 🛠️ Prerequisites

Install the generation and context dependencies described in the main README. Git must be on `PATH`; Java is also required for the ENRE-Java and ENRE-C++ JARs. Python ENRE source and the two JARs are provided in [ENRE-tools](../ENRE-tools/). Their original licenses are retained. The Python analyzer runs as `python -m enre` in its bundled directory; install any additional dependencies needed by that analyzer in the same environment.

Core entity identification uses an OpenAI-compatible endpoint. Configure `OPENAI_API_KEY` and `OPENAI_BASE_URL`, then select the desired model with `--model`. Its prompt is stored in [prompts/core_entity_identification.txt](../prompts/core_entity_identification.txt); the extractor loads this file directly.

## 🎯 Identify Core Entities

The following examples use the released Python ablation commits. Replace `python` with `java` or `cpp` and choose the corresponding language workflow for other languages.

```bash
python src/view1/utils_relatedcode/core_entities/extract_entities.py --input_file dataset/ablation/commits/python.jsonl --output_file outputs/view1/core_entities_python.jsonl --model YOUR_MODEL --workers 5
python src/view1/utils_relatedcode/extract_instance_ids.py dataset/ablation/commits/python.jsonl outputs/view1/commit_map_python.jsonl
```

The first command calls the LLM and writes records containing `instance_id` and `core_entities`. The second writes the `instance_id` to full commit SHA mapping. Both use the identifier convention `owner_repo_<first seven SHA characters>`. Entity records contain the entity type, path, name, and the optional enclosing class name. The mapping is used by all three structural workflows.

## 🔗 Construct the Structural Dependency View

### 📥 Prepare Repository Checkouts

```bash
python src/view1/clone_repos_v2.py --language python --dataset_file dataset/ablation/commits/python.jsonl --target_dir outputs/view1/repositories/python --clone_mode partial
python src/view1/utils_enre_reports/process_dataset.py --jsonl_file dataset/ablation/commits/python.jsonl --base_repo_path outputs/view1/repositories/python --log_file outputs/view1/fetch_python.log --max_items 0
```

Cloning accepts raw commit records and records containing a nested `subset_entry`. With no explicit dataset or destination, `--language` selects `dataset/full/commits/<language>.jsonl` and `outputs/view1/repositories/<language>/`. `--dry_run` prints the clone commands. Commit fetching defaults to five records; `--max_items 0` processes the complete input.

The analyzer and workflow scripts check out the requested commits in these dedicated repository copies. Use the clones under `outputs/` as their working inputs.

### 🐍 Analyze Python Repositories

```bash
python src/view1/utils_enre_reports/enre_report_generator_py.py --dataset_file dataset/ablation/commits/python.jsonl --repo_dir outputs/view1/repositories/python --output_dir outputs/view1/enre_python
python src/view1/workflow_processor_py.py --core_entities_file outputs/view1/core_entities_python.jsonl --commit_map_file outputs/view1/commit_map_python.jsonl --all_enre_report_dir outputs/view1/enre_python --all_repo_dir outputs/view1/repositories/python --output_file outputs/view1/related_code_python.jsonl --error_report_file outputs/view1/related_code_python_errors.jsonl
```

The Python wrapper finds `ENRE-tools/ENRE-py` automatically. `--enre_py_dir` selects another ENRE source checkout. Its optional `--cfg`, `--cg`, and `--compatible` flags are passed through to ENRE. The Python workflow also supports `--max_instances`, `--resume`, and `--log_file`.

### ☕ Analyze Java Repositories

Clone and fetch the Java commits, identify their core entities, and generate a commit mapping using the same commands above with the Java input. Then run:

```bash
python src/view1/utils_enre_reports/enre_report_generator_java.py --dataset_file dataset/ablation/commits/java.jsonl --repo_dir outputs/view1/repositories/java --enre_jar ENRE-tools/ENRE-java/enre_java.jar --output_dir outputs/view1/enre_java
python src/view1/workflow_processor_java.py --core_entities_file outputs/view1/core_entities_java.jsonl --commit_map_file outputs/view1/commit_map_java.jsonl --all_enre_report_dir outputs/view1/enre_java --all_repo_dir outputs/view1/repositories/java --output_file outputs/view1/related_code_java.jsonl --error_report_file outputs/view1/related_code_java_errors.jsonl
```

ENRE-Java can place reports in a `<repo>-enre-out` subdirectory. Keep its report directory structure available to the workflow. The Java code retriever uses the Tree-sitter Java grammar for source blocks when that dependency is installed.

### 🧱 Analyze C++ Repositories

Prepare the C++ commits, clones, core entities, and mapping in the same way, then run:

```bash
python src/view1/utils_enre_reports/enre_report_generator_cpp.py --dataset_file dataset/ablation/commits/cpp.jsonl --repo_dir outputs/view1/repositories/cpp --output_dir outputs/view1/enre_cpp
python src/view1/workflow_processor_cpp.py --core_entities_file outputs/view1/core_entities_cpp.jsonl --commit_map_file outputs/view1/commit_map_cpp.jsonl --all_enre_report_dir outputs/view1/enre_cpp --all_repo_dir outputs/view1/repositories/cpp --output_file outputs/view1/related_code_cpp.jsonl --error_report_file outputs/view1/related_code_cpp_errors.jsonl
```

The wrapper finds the bundled C++ JAR automatically. Use `--enre_jar` to override it. To skip particular commits, supply `--skip_instance_id` or your own error log through `--skip_from_error_log`. `--java_xmx`, `--program_environment`, and `--dir` configure the analyzer environment. `utils_relatedcode/clean_cpp_related_code.py --help` describes the optional C++ context cleanup step.

All three workflows produce JSONL context records keyed by `instance_id`, along with an error report for unresolved entries. `utils_enre_reports/retry_failed_fetches.py` retries Git failures from a fetch log; run it with `--help` for the log and repository arguments.

## 🧠 Construct the Functional Semantic View

### 📦 Available Components and Inputs

The package includes a repository snapshot downloader, an ENRE report parser, and the CodeBERT + UMAP + HDBSCAN clustering script. Prepare functional context locally, grouped by language and commit. Each generation input directory needs `core_entities.jsonl` and `core_clusters_simplified.jsonl`. The task generator reads these directories under `dataset/ablation/contexts/view2/` or `dataset/full/contexts/view2/`, depending on the selected dataset scope.

To run clustering, prepare one directory per repository snapshot containing:

| File | Content |
| --- | --- |
| `filtered_entities.json` | An object with an `entities` array; each entity includes its code and `qualifiedName` or `id`. |
| `core_entities.jsonl` | One enriched core entity per line, using the same `qualifiedName` or `id` values as the entities array. |

The clustering script also accepts the legacy spelling `core_eneities.jsonl`. The `core_entities.jsonl` input here is an enriched entity list; it is different from the per-commit list produced by the core identification command above. Matching those identified entities to ENRE entities, preparing source snapshots and reports, and converting clusters to the simplified generation schema require additional preparation; an integrated driver for those steps is not included.

### 🗂️ Download Snapshots and Parse Reports

```bash
python src/view2/download_repository_snapshots.py --input dataset/ablation/commits/python.jsonl --output-dir outputs/view2/repositories
python src/view2/parse_enre_report.py outputs/view1/enre_python/INSTANCE_enre_report.json --base-dir outputs/view1/repositories/python --output outputs/view2/repositories/INSTANCE/filtered_entities.json
```

Replace `INSTANCE` with the identifier of the actual report and prepared snapshot. The downloader saves `repo.zip` and `commit_message.txt` per commit; unpack a snapshot before using its source files for analysis. Optionally set `GITHUB_TOKEN` for authenticated GitHub requests and `HTTP_PROXY` or `HTTPS_PROXY` for a proxy. The parser resolves each report entity's `File` value against `--base-dir`, so select the source root matching those file paths. It filters `Class` and `Function` entities and extracts their code with indentation-based boundaries. This parser is suited to the Python report format; Java and C++ entity preparation must supply equivalent enriched records.

### 🧮 Cache CodeBERT and Run Clustering

The clustering script runs offline and loads `microsoft/codebert-base` with `local_files_only=True`. Download its tokenizer and model into the Hugging Face cache in a separate process before running the script:

```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/codebert-base'); AutoModel.from_pretrained('microsoft/codebert-base')"
python src/view2/clustering_codebert_umap_hdbscan.py --repos-dir outputs/view2/repositories
```

Run the cache preparation command in a process where `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are not enabled. Clustering uses CUDA when available and otherwise uses the CPU. The default UMAP parameters are 20 components, 30 neighbors, and a minimum distance of 0.0; HDBSCAN uses a minimum cluster size of 3 and one minimum sample. The script writes `clusters_hdbscan.json` per instance and skips existing outputs. Its UMAP random state is unset, so rerunning clustering can change the resulting clusters. Preserve the locally generated clusters when comparing generation configurations.

## 🕰️ Construct the Historical Exemplar View

The historical retrieval code and its environment are in [src/view3](../src/view3/). Refer to the [historical retrieval guide](historical_retrieval.md) for corpus preparation, indexing, and retrieval commands. Construct this view locally from a historical candidate corpus and export its retrieved exemplars before generating tasks that use View3.
