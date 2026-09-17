# Japanese Omission-Query Retrieval

Dataset and evaluation code for Ihara and Rzepka's study of Japanese questions with omitted information (LAU 2026). This repository contains the revised evaluation and manuscript, updated 2026-09-17.

The 276 human-evaluated queries cover three grammatical cases and four semantic-role categories. Retrieval is evaluated against 510 evidence pages from [JDocQA](https://github.com/aiishii/JDocQA). The questions and annotations are unchanged by this revision.

## Current manuscript and results

- [Latest manuscript PDF](paper/paper_lualatex_v4.pdf) and [LaTeX source](paper/paper_lualatex_v4.tex)
- [Dataset and schema](data/README.md), [annotation procedure](docs/annotation_guidelines.md), [annotator IDs](docs/annotation_identifiers.md)
- [Revised results](results/README.md), [experimental protocol](docs/review_20260911/experiment_protocol.md), [quality notes](docs/quality_notes.md)
- [Generation conditions](docs/query2doc_generation_conditions.md) and [exact prompt resources](docs/prompts/)

| Method | Main MRR |
| --- | ---: |
| Dense | 0.7056 |
| BM25 | 0.8333 |
| RM3 | 0.8200 |
| Query2doc | 0.8382 |
| Query2doc with gold category | 0.8447 |
| RRF | 0.8073 |
| Weighted sparse–dense fusion | 0.8460 |

The Hybrid–BM25 difference is not statistically significant. Category-based selection (0.8387) did not outperform one-method selection (0.8422). The three research questions concern omission effects, retrieval-method performance, and category-based selection. Query weighting is a supplemental analysis within the method comparison.

## Install and verify saved results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/test_review_experiments.py
python scripts/verify_release.py
python scripts/analyze_review_experiments.py
```

These checks and reanalysis use released ranks and do not call an API. Reanalysis writes to `outputs/review/`, leaving reference results untouched. Randomization and bootstrap use 100,000 samples.

## Recompute retrieval

Obtain JDocQA under its upstream terms and put annotations in `JDocQA/dataset/annotation_files/`. Build the corpus with `python scripts/build_jdocqa_510_dataset.py`. Source PDFs and the 510-page corpus are not redistributed. The full grid additionally requires cached `text-embedding-3-small` embeddings for corpus pages, original questions and omission queries in `omission_query_dataset/embedding_cache/`; these caches are not distributed. New embeddings may require API access and may not reproduce historical outputs exactly.

```bash
python scripts/run_review_experiments.py
python scripts/analyze_review_experiments.py --input-dir outputs/review
```

This is the main experimental entry point: shared document-grouped outer folds, parameter selection on development data, and query count weighting. It uses released pseudo-documents in `data/query2doc/` and writes to `outputs/review/`. The script checks historical baselines against `results/submission/` before saving revised results.

Standalone `run_*_retrieval*.py` commands are fixed-setting utilities, not substitutes for the tuned main comparison. BM25 now counts repeated query terms. Query2doc uses separate bags with weight five, avoiding artificial boundary and `[SEP]` bigrams. For example:

```bash
python scripts/run_query2doc_retrieval.py --pseudo-documents data/query2doc/generic.jsonl --method-name Query2doc --output outputs/query2doc_fixed.json
```

Generation is separate from retrieval. Existing generated texts were not changed or regenerated for this release. See the quality notes for the historical completion checks. Provider request IDs are omitted from public cache records; query text and generated text are unchanged.

## Changes and citation

The revision fixes query weighting, adds held-out parameter selection and nested category routing, updates annotation definitions and audit notes, and replaces the primary result references. Historical results remain in `results/submission/` and in Git history. See [release notes](docs/release_notes.md).

Citation metadata is in [CITATION.cff](CITATION.cff). Cite the study and JDocQA when using the derived dataset. Code is MIT licensed; the derived dataset is CC BY-SA 4.0 (see [data license](DATA_LICENSE.md)). Private worker workbooks, credentials, source PDFs and embeddings are not included.
