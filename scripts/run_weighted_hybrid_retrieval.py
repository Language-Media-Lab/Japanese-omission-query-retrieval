#!/usr/bin/env python3
"""Evaluate a cross-fitted weighted Dense/BM25 score fusion.

Dense and BM25 score rows are min-max normalized before interpolation.  Whole
gold-document groups are assigned to one fold, and alpha is selected using
only the other folds' omission-query MRR.  This prevents questions sharing a
gold document from leaking between development and evaluation partitions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_bm25_retrieval_baseline import BM25, read_jsonl, tokenize
from run_rrf_retrieval import load_cached_embeddings


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_CORPUS = ROOT / "omission_query_dataset" / "jdocqa_510_corpus.jsonl"
DEFAULT_CACHE = ROOT / "outputs" / "embedding_cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "weighted_hybrid_retrieval_results.json"
KS = (1, 3, 5, 10)


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Independently scale each query's score row to the interval [0, 1]."""
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    ranges = maximum - minimum
    ranges[ranges == 0.0] = 1.0
    return (scores - minimum) / ranges


def make_group_folds(dataset: list[dict], fold_count: int, seed: int) -> list[list[int]]:
    """Assign whole gold-chunk groups to balanced deterministic folds."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        grouped[row["gold_chunk_id"]].append(index)
    if fold_count > len(grouped):
        raise ValueError("Fold count exceeds number of gold-chunk groups")
    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for _, indices in groups:
        target = min(range(fold_count), key=lambda fold: (len(folds[fold]), fold))
        folds[target].extend(indices)
    return [sorted(fold) for fold in folds]


def rank_indices(scores: np.ndarray, chunk_ids: list[str]) -> np.ndarray:
    return np.asarray(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), chunk_ids[index])))


def mean_reciprocal_rank(
    scores: np.ndarray,
    query_indices: list[int],
    gold_indices: np.ndarray,
    chunk_ids: list[str],
) -> float:
    total = 0.0
    for query_index in query_indices:
        ranking = rank_indices(scores[query_index], chunk_ids)
        inverse = np.empty(len(ranking), dtype=np.int32)
        inverse[ranking] = np.arange(1, len(ranking) + 1)
        total += 1.0 / int(inverse[gold_indices[query_index]])
    return total / len(query_indices)


def select_alpha(
    dense: np.ndarray,
    bm25: np.ndarray,
    train_indices: list[int],
    gold_indices: np.ndarray,
    chunk_ids: list[str],
    alphas: list[float],
) -> tuple[float, list[dict]]:
    """Choose alpha by development-set MRR without consulting the test fold."""
    candidates = []
    for alpha in alphas:
        scores = alpha * dense + (1.0 - alpha) * bm25
        mrr = mean_reciprocal_rank(scores, train_indices, gold_indices, chunk_ids)
        candidates.append({"alpha": alpha, "development_mrr": mrr})
    best_mrr = max(row["development_mrr"] for row in candidates)
    # If tied, prefer the alpha closest to equal weighting, then the smaller value.
    best = min(
        (row for row in candidates if abs(row["development_mrr"] - best_mrr) < 1e-12),
        key=lambda row: (abs(row["alpha"] - 0.5), row["alpha"]),
    )
    return best["alpha"], candidates


def evaluate_variant(
    dataset: list[dict],
    corpus: list[dict],
    dense: np.ndarray,
    bm25: np.ndarray,
    folds: list[list[int]],
    selected_alphas: list[float],
) -> tuple[dict, list[dict]]:
    """Apply each fold's preselected alpha and aggregate out-of-fold metrics."""
    chunk_ids = [row["chunk_id"] for row in corpus]
    chunk_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    gold_indices = np.asarray([chunk_index[row["gold_chunk_id"]] for row in dataset])
    recalls = {k: 0 for k in KS}
    reciprocal_rank_sum = 0.0
    details: list[dict | None] = [None] * len(dataset)
    for fold_id, test_indices in enumerate(folds):
        alpha = selected_alphas[fold_id]
        for query_index in test_indices:
            scores = alpha * dense[query_index] + (1.0 - alpha) * bm25[query_index]
            ranking = rank_indices(scores, chunk_ids)
            inverse = np.empty(len(ranking), dtype=np.int32)
            inverse[ranking] = np.arange(1, len(ranking) + 1)
            gold_rank = int(inverse[gold_indices[query_index]])
            reciprocal_rank_sum += 1.0 / gold_rank
            for k in KS:
                recalls[k] += gold_rank <= k
            top = ranking[: max(KS)]
            details[query_index] = {
                "eval_id": dataset[query_index]["eval_id"],
                "gold_chunk_id": dataset[query_index]["gold_chunk_id"],
                "gold_rank": gold_rank,
                "fold": fold_id + 1,
                "alpha": alpha,
                "top_chunk_ids": [chunk_ids[index] for index in top],
                "top_scores": [float(scores[index]) for index in top],
            }
    count = len(dataset)
    if any(row is None for row in details):
        raise RuntimeError("Some queries were not assigned an out-of-fold result")
    return ({
        "query_count": count,
        "recall": {f"R@{k}": recalls[k] / count for k in KS},
        "mrr": reciprocal_rank_sum / count,
    }, details)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Evaluation-query JSONL.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Retrieval-document JSONL.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Dense embedding cache directory.")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model used to construct the cache.")
    parser.add_argument("--folds", type=int, default=5, help="Number of grouped cross-validation folds.")
    parser.add_argument("--seed", type=int, default=20260802, help="Seed for deterministic gold-document group assignment.")
    parser.add_argument("--alpha-step", type=float, default=0.1, help="Spacing of the Dense-weight search grid from zero to one.")
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 term-frequency saturation parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 document-length normalization parameter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    args = parser.parse_args()
    if not 0.0 < args.alpha_step <= 1.0:
        raise ValueError("--alpha-step must be in (0, 1]")

    dataset = read_jsonl(args.dataset)
    corpus = read_jsonl(args.corpus)
    chunk_ids = [row["chunk_id"] for row in corpus]
    chunk_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    gold_indices = np.asarray([chunk_index[row["gold_chunk_id"]] for row in dataset])
    folds = make_group_folds(dataset, args.folds, args.seed)
    fold_by_index = {index: fold_id for fold_id, fold in enumerate(folds) for index in fold}
    for gold_id in {row["gold_chunk_id"] for row in dataset}:
        assigned = {fold_by_index[index] for index, row in enumerate(dataset) if row["gold_chunk_id"] == gold_id}
        if len(assigned) != 1:
            raise RuntimeError("Gold-chunk group leaked across folds")

    corpus_texts = [row["text"] for row in corpus]
    corpus_vectors = load_cached_embeddings(args.cache_dir, args.model, corpus_texts)
    bm25_model = BM25([tokenize(text) for text in corpus_texts], args.bm25_k1, args.bm25_b)
    variant_scores = {}
    for variant, field in (("original", "original_question"), ("omission", "omission_query")):
        query_vectors = load_cached_embeddings(args.cache_dir, args.model, [row[field] for row in dataset])
        dense = minmax_rows(query_vectors @ corpus_vectors.T)
        bm25 = minmax_rows(np.asarray([bm25_model.scores(tokenize(row[field])) for row in dataset]))
        variant_scores[variant] = (dense, bm25)

    steps = round(1.0 / args.alpha_step)
    alphas = sorted({round(index * args.alpha_step, 10) for index in range(steps + 1)} | {1.0})
    selected_alphas = []
    fold_selection = []
    omission_dense, omission_bm25 = variant_scores["omission"]
    all_indices = set(range(len(dataset)))
    for fold_id, test_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(test_indices))
        alpha, candidates = select_alpha(
            omission_dense, omission_bm25, train_indices, gold_indices, chunk_ids, alphas
        )
        selected_alphas.append(alpha)
        fold_selection.append({
            "fold": fold_id + 1,
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "test_gold_chunk_count": len({dataset[index]["gold_chunk_id"] for index in test_indices}),
            "selected_alpha": alpha,
            "candidates": candidates,
        })

    output = {"configuration": {
        "method": "Weighted Hybrid",
        "formula": "alpha * minmax(Dense) + (1-alpha) * minmax(BM25)",
        "selection_metric": "development MRR on omission queries",
        "grouping": "gold_chunk_id",
        "fold_count": args.folds,
        "seed": args.seed,
        "alpha_grid": alphas,
        "selected_alphas": selected_alphas,
        "selected_alpha_counts": dict(Counter(str(alpha) for alpha in selected_alphas)),
        "dense_model": args.model,
        "bm25_tokenizer": "NFKC-lowercase-character-bigram",
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "corpus_scope": "full",
        "corpus_size": len(corpus),
        "dataset_size": len(dataset),
        "ks": list(KS),
    }, "fold_selection": fold_selection}
    for variant in ("original", "omission"):
        summary, details = evaluate_variant(
            dataset, corpus, *variant_scores[variant], folds, selected_alphas
        )
        output[variant] = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_alphas": selected_alphas,
        "original": output["original"]["summary"],
        "omission": output["omission"]["summary"],
    }, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
