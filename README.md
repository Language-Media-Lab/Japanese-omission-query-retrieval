# Japanese Omission-Query Retrieval

Dataset and evaluation code for the paper **“Systematic Evaluation of Retrieval and Query Expansion Methods for Japanese Questions with Omitted Information”** (Ihara and Rzepka, LAU Summer 2026).

## Overview

This repository provides:

- 276 manually evaluated Japanese omission queries in seven categories;
- character-bigram BM25 and dense-retrieval baselines;
- the paper's RM3-style and Query2doc-style query-expansion implementations;
- Reciprocal Rank Fusion and cross-fitted weighted Dense/BM25 fusion;
- category-routing and cluster-aware significance analyses;
- the prompts and fixed result files needed to audit the study.

The dataset is derived from [JDocQA](https://github.com/aiishii/JDocQA). In the experiments, one evidence page assigned by JDocQA is treated as one retrieval document. The underlying PDFs and the 510-document text corpus are not redistributed here; obtain JDocQA under its terms and use the included builders.

## Repository layout

```text
data/       Released 276-query evaluation dataset and schema notes
docs/       Annotation and Query2doc condition documentation
results/    Fixed result JSON files reported in the paper
scripts/    Dataset builders, retrieval methods, and analyses
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dense retrieval and Query2doc generation require an OpenAI API key supplied only through the environment:

```bash
export OPENAI_API_KEY="your-key"
```

Never commit an API key. Local `.env`, `API`, key files, embedding caches, and private annotation workbooks are excluded by `.gitignore`.

## Prepare JDocQA

Download the JDocQA annotation files according to the upstream instructions and place them under:

```text
JDocQA/dataset/annotation_files/
```

Then build the 510-document source data and corpus:

```bash
python scripts/build_jdocqa_510_dataset.py
```

The included `data/annotation_filtered_276.jsonl` contains the finalized omission-query evaluation set. See `data/README.md` for its schema and license.

## Run the main methods

Run commands from the repository root. Use `--help` for documented parameters.

```bash
python scripts/run_bm25_retrieval_baseline.py
python scripts/run_dense_retrieval_baseline.py
python scripts/run_rm3_retrieval.py
python scripts/run_rrf_retrieval.py
python scripts/run_weighted_hybrid_retrieval.py
python scripts/analyze_clustered_significance.py
python scripts/run_oracle_routing.py
```

Generate Query2doc pseudo-documents separately so API outputs can be cached and audited:

```bash
python scripts/generate_query2doc.py --mode both
python scripts/run_query2doc_retrieval.py \
  --pseudo-documents outputs/query2doc/generic.jsonl \
  --method-name BM25+Query2doc \
  --output outputs/query2doc_retrieval_results.json
```

The recorded retrieval condition concatenates each omission query once with `[SEP]` and its pseudo-document. `[SEP]` is ordinary text under the character-bigram tokenizer, not a special token. API generations are not guaranteed to reproduce the released fixed results exactly.

## Reproducibility notes

- BM25 uses NFKC normalization, lowercasing, whitespace removal, and overlapping character bigrams.
- Weighted fusion selects its Dense weight using development folds only.
- All questions sharing a gold document remain in the same fold.
- Significance tests use gold-document clusters.
- `generate_query2doc.py --seed` controls retry-delay jitter only; it does not seed model generation.
- Fixed outputs used in the paper are retained under `results/`.

## Licenses

- Source code: MIT License; see `LICENSE`.
- Released derived dataset: CC BY-SA 4.0; see `DATA_LICENSE.md`.
- JDocQA annotations remain subject to the upstream CC BY-SA 4.0 terms and citation request.
- Underlying PDFs are not redistributed.

## Citation

Citation metadata is provided in `CITATION.cff`. Please also cite JDocQA when using the released derived dataset.

## Contact

Language Media Laboratory, Hokkaido University
