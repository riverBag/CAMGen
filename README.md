# CAMGen

**A Core Change-Aware Multi-View Framework for Commit Message Generation**

<div align="center">
    <a href="https://github.com/riverBag/CAMGen">
      <img src="https://img.shields.io/badge/Code-GitHub-2d333b?style=flat-square&logo=github" alt="github">
    </a>
    <a href="">
      <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <hr>
</div>

CAMGen generates commit messages by identifying the core entities of a code
change and combining three views of repository context: structural dependencies,
functional semantics, and historical exemplars. Together with the code diff,
these views help an LLM explain the main change and its functional impact.

This package provides the evaluation commits, the fixed 1,000-commit ablation
subset, prompt templates, context-construction code, task generation, API
inference, and automatic evaluation scripts.

## ⚙️ Usage as a Replication Package

Run the following commands from the package root. **Task construction requires
only Python 3.10 or later**, with no API credentials or model downloads:

```sh
python scripts/experiments/task_generator_ablation_full.py
```

This builds the diff-only configuration for the **1,000-commit ablation
set**, writing `cpp.jsonl`, `java.jsonl`, `python.jsonl`, `all.jsonl`, and
`manifest.json` to `outputs/tasks/ablation/ablation_direct/`.
Generate the required contexts locally before running configurations that
use repository context. `--dataset-scope` selects the commit set.

Install additional dependencies for the stages you want to run:

| Stage | Setup |
| --- | --- |
| API inference and core entity identification | `python -m pip install openai python-dotenv rich tenacity` |
| Structural and functional context construction | `python -m pip install openai python-dotenv rich tenacity tqdm PyYAML numpy torch transformers tree-sitter tree-sitter-python tree-sitter-java umap-learn hdbscan` |
| Historical exemplar retrieval | `conda env create -f environments/historical_retrieval.yml`, then `conda activate camgen-history` |
| Automatic evaluation | `python -m pip install evaluate fire sacrebleu numpy torch scikit-learn sentence-transformers rouge-score nltk absl-py` |

Context construction also needs repository snapshots, the appropriate ENRE
runtime, and pretrained models. See the [context construction guide](docs/context_construction.md)
and [historical retrieval guide](docs/historical_retrieval.md) for those stages.

## 📁 Package Structure

```text
replication_package/
├── README.md
├── .env.example                     # API configuration template
├── dataset/
│   ├── full/commits/                 # 7,522 evaluation commits
│   └── ablation/commits/             # 1,000 selected commits
├── prompts/                         # Entity identification + 8 CMG settings
├── src/
│   ├── view1/                       # Structural Dependency View
│   ├── view2/                       # Functional Semantic View
│   └── view3/                       # Historical Exemplar View
├── scripts/
│   ├── experiments/                 # Task construction
│   ├── generation/                  # Batch LLM inference
│   └── evaluation/                  # Metrics and local metric implementations
├── environments/                    # Historical retrieval Conda environment
├── ENRE-tools/                      # Bundled ENRE tools and upstream notices
└── docs/                            # Context construction and retrieval guides
```

Generated tasks, predictions, indices, and reports belong in `outputs/`, which
is created by the relevant commands and excluded from version control.

## 🧭 The Three Views

| View | Context supplied to the generator | Implementation |
| --- | --- | --- |
| **Structural Dependency View** (`view1`) | Code related to core entities through dependency relations | [src/view1](src/view1/) |
| **Functional Semantic View** (`view2`) | Semantic clusters containing core entities | [src/view2](src/view2/) |
| **Historical Exemplar View** (`view3`) | A retrieved code diff and commit message | [src/view3](src/view3/) |

Core entity identification is implemented in
[`extract_entities.py`](src/view1/utils_relatedcode/core_entities/extract_entities.py).
Its instruction template is stored separately in
[`prompts/core_entity_identification.txt`](prompts/core_entity_identification.txt).

## 📦 Dataset

The evaluation commits are derived from **ApacheCM** and cover three programming
languages. The ablation set is a fixed subset of the main evaluation set.

| Language | Main evaluation set | Ablation set |
| --- | ---: | ---: |
| C++ | 2,498 | 332 |
| Java | 2,469 | 328 |
| Python | 2,555 | 340 |
| **Total** | **7,522** | **1,000** |

Commit records are stored as JSONL in `dataset/full/commits/<language>.jsonl` and
`dataset/ablation/commits/<language>.jsonl`, using `cpp`, `java`, and `python` as
language names. Each record contains `git_url`, `owner`, `repo`, `commit_sha`,
`author`, `date`, `message`, `files`, `diff`, and `loc`. The `message` field is
the reference commit message; `diff` is the generation input.
Task identifiers and context join keys are derived from these commit fields
when tasks are built.

Both commit sets support diff-only task construction directly. For configurations
using additional views, first construct the corresponding context inputs by
following the guides in `docs/`.

## 🧩 Prompt Templates

All templates are stored in [`prompts/`](prompts/). View identifiers match the
implementation directories and the table above.

| Configuration | Template |
| --- | --- |
| Diff only | `ablation_direct_prompt.txt` |
| Structural dependencies | `ablation_view1_prompt.txt` |
| Functional semantics | `ablation_view2_prompt.txt` |
| Historical exemplar | `ablation_view3_prompt.txt` |
| Structural + functional | `ablation_view1_view2_prompt.txt` |
| Structural + historical | `ablation_view1_view3_prompt.txt` |
| Functional + historical | `ablation_view2_view3_prompt.txt` |
| **All three views** | `full_view1_view2_view3_prompt.txt` |

The [prompt directory](prompts/) contains the generation templates and the core
entity identification instruction.

## 🔁 Reproduction

### 🛠️ Generating Tasks

Generate diff-only tasks for the fixed ablation subset:

```sh
python scripts/experiments/task_generator_ablation_full.py --prompt-path prompts/ablation_direct_prompt.txt
```

Generate diff-only tasks for the complete evaluation set:

```sh
python scripts/experiments/task_generator_ablation_full.py --dataset-scope full --prompt-path prompts/ablation_direct_prompt.txt
```

After constructing the required contexts locally, select another configuration
with `--prompt-path`, such as `prompts/full_view1_view2_view3_prompt.txt` for all
three views. The [construction guide](docs/context_construction.md) describes
how to prepare these inputs.

Outputs are grouped under `outputs/tasks/<scope>/<configuration>/`. Use
`--languages cpp java python` to choose languages, `--output-dir` to choose an
output root, and `--overwrite` to replace existing task files. Each task contains
`task_id`, chat-style `messages`, a reference `label`, and metadata; only the
`messages` are sent to the model.

### 🚀 Running LLM Inference

Copy `.env.example` to `.env` and set your provider's `OPENAI_API_KEY` and
`OPENAI_BASE_URL`. The runner supports OpenAI-compatible chat completion APIs.
Use the model identifier exposed by your provider:

```sh
python scripts/generation/batch_commit_generator.py --input outputs/tasks/ablation/ablation_direct/all.jsonl --output outputs/predictions/deepseek-v3.2/ablation_direct.jsonl --model deepseek-v3.2-128k --workers 10
```

The runner uses temperature `0`, records `task_id`, `label`, `model`, and
`response`, and writes requests that fail after retries to a separate
`*_failures.jsonl` file. Each run replaces its output files. Use distinct output
paths for different models and configurations.

### 📏 Evaluating Predictions

```sh
python scripts/evaluation/eval_ablation.py --result_path outputs/predictions/deepseek-v3.2/ablation_direct.jsonl --output_dir outputs/evaluation/ablation_direct_cleaned.jsonl --report_file outputs/evaluation/ablation_direct.txt
```

The evaluator reports **BLEU, ROUGE-L, METEOR, CIDEr, and SBERT Cosine**. The
supplied BLEU-labelled score uses `evaluate`'s `google_bleu` (GLEU) metric.
The evaluator removes `<message>` tags from predictions and leading issue identifiers from
reference messages. Metric resources and `stsb-roberta-large` must be available
locally or downloaded on first use. Reports retain the supplied implementation's
scaling: BLEU/ROUGE-L/METEOR/SBERT ×100 and CIDEr ×10.

### 🏗️ Constructing Contexts from Repositories

Use the [context construction guide](docs/context_construction.md) for core
entity identification, ENRE reports, dependency retrieval, and semantic
clustering. Use the [historical retrieval guide](docs/historical_retrieval.md)
for corpus preparation, indexing, and exemplar retrieval.

For historical retrieval, the paper's protocol excludes all 7,522 evaluation
commits from the candidate corpus and restricts candidates to earlier commits
in the same repository. Apply these constraints when preparing the historical
corpus and retrieving exemplars.

## 🧪 Experiments in the Paper

| Research question | Experiment | Materials in this package |
| --- | --- | --- |
| **RQ1** | Overall CMG performance | 7,522 commits; CAMGen prompts, inference, and evaluation code |
| **RQ2.1** | Core entity identification | Identification code and prompt |
| **RQ2.2** | Contribution of the three views | Fixed 1,000-commit subset, context-construction code, and all 8 prompt configurations |
| **RQ3** | Human evaluation | Commits are drawn from the ablation set; annotation materials are separate |
| **RQ4** | Qualitative case analysis | Commit diffs are available for inspecting selected cases |

The paper evaluates DeepSeek-V3.2, DeepSeek-V4-Flash, and GPT-5.4, and uses
DeepSeek-V3.2 for the view ablation and human evaluation.

## 📝 Citation

```bibtex
@misc{xiong2026camgen,
  title         = {{CAMGen}: A Core Change-Aware Multi-View Framework for Commit Message Generation},
  author        = {Xiong, Bo and Chen, Weisheng and Ren, Zongen and Wang, Chong and Liang, Peng},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {},
  url           = {}
}
```

## 🙏 Acknowledgments

This package builds on ApacheCM, ENRE, Tree-sitter, CodeBERT, UMAP, HDBSCAN,
Jina code embeddings, and the metric implementations used by the evaluation
scripts. Bundled upstream license files remain with their components, including
[`src/view2/LICENSE`](src/view2/LICENSE) and
[`ENRE-tools/ENRE-py/LICENSE`](ENRE-tools/ENRE-py/LICENSE).

