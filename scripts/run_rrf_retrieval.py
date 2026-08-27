#!/usr/bin/env python3
"""Fuse cached Dense and BM25 rankings with unweighted RRF.

For document ``d``, the score is ``1/(k+rank_dense(d)) +
1/(k+rank_bm25(d))``.  Dense embeddings must first be created by
``run_dense_retrieval_baseline.py``; this script makes no API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_bm25_retrieval_baseline import BM25, read_jsonl, tokenize
from run_dense_retrieval_baseline import cache_key, normalized


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_CORPUS = ROOT / "omission_query_dataset" / "jdocqa_510_corpus.jsonl"
DEFAULT_CACHE = ROOT / "outputs" / "embedding_cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "rrf_retrieval_results.json"
KS = (1, 3, 5, 10)


def load_cached_embeddings(cache_dir: Path, model: str, texts: list[str]) -> np.ndarray:
    """Load and normalize the exact cache for the requested model and texts."""
    path = cache_dir / f"{cache_key(model, texts)}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Dense embedding cache is missing: {path}. Run run_dense_retrieval_baseline.py first."
        )
    return normalized(np.load(path))


def ranks_from_scores(scores: np.ndarray, tie_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ranking indices and one-based inverse ranks for each query."""
    rankings = []
    inverse_ranks = []
    for row in scores:
        if tie_ids is None:
            ranking = np.argsort(-row, kind="stable")
        else:
            ranking = np.asarray(sorted(range(len(row)), key=lambda i: (-float(row[i]), tie_ids[i])))
        inverse = np.empty(len(ranking), dtype=np.int32)
        inverse[ranking] = np.arange(1, len(ranking) + 1)
        rankings.append(ranking)
        inverse_ranks.append(inverse)
    return np.asarray(rankings), np.asarray(inverse_ranks)


def evaluate(
    dataset: list[dict],
    corpus: list[dict],
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    rrf_k: int,
) -> tuple[dict, list[dict]]:
    """Compute RRF scores and standard retrieval metrics for one query variant."""
    dense_rankings, dense_ranks = ranks_from_scores(dense_scores)
    bm25_rankings, bm25_ranks = ranks_from_scores(
        bm25_scores, [row["chunk_id"] for row in corpus]
    )
    rrf_scores = 1.0 / (rrf_k + dense_ranks) + 1.0 / (rrf_k + bm25_ranks)
    rrf_rankings, rrf_ranks = ranks_from_scores(rrf_scores)
    corpus_index = {row["chunk_id"]: index for index, row in enumerate(corpus)}

    recalls = {k: 0 for k in KS}
    reciprocal_rank_sum = 0.0
    details = []
    for query_index, row in enumerate(dataset):
        gold_index = corpus_index[row["gold_chunk_id"]]
        gold_rank = int(rrf_ranks[query_index, gold_index])
        reciprocal_rank_sum += 1.0 / gold_rank
        for k in KS:
            recalls[k] += gold_rank <= k
        top = rrf_rankings[query_index, : max(KS)]
        details.append({
            "eval_id": row["eval_id"],
            "gold_chunk_id": row["gold_chunk_id"],
            "gold_rank": gold_rank,
            "dense_gold_rank": int(dense_ranks[query_index, gold_index]),
            "bm25_gold_rank": int(bm25_ranks[query_index, gold_index]),
            "top_chunk_ids": [corpus[index]["chunk_id"] for index in top],
            "top_scores": [float(rrf_scores[query_index, index]) for index in top],
        })
    count = len(dataset)
    return ({
        "query_count": count,
        "recall": {f"R@{k}": recalls[k] / count for k in KS},
        "mrr": reciprocal_rank_sum / count,
    }, details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Evaluation-query JSONL.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Retrieval-document JSONL.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Dense embedding cache directory.")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model used to construct the cache.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank-offset constant.")
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 term-frequency saturation parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 document-length normalization parameter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    args = parser.parse_args()
    if args.rrf_k < 0:
        raise ValueError("--rrf-k must be non-negative")

    dataset = read_jsonl(args.dataset)
    corpus = read_jsonl(args.corpus)
    if not {row["gold_chunk_id"] for row in dataset} <= {row["chunk_id"] for row in corpus}:
        raise ValueError("Some gold chunks are missing from the corpus")
    corpus_texts = [row["text"] for row in corpus]
    corpus_vectors = load_cached_embeddings(args.cache_dir, args.model, corpus_texts)
    bm25 = BM25([tokenize(text) for text in corpus_texts], k1=args.bm25_k1, b=args.bm25_b)
    output = {"configuration": {
        "method": "Reciprocal Rank Fusion",
        "components": ["Dense cosine", "BM25Okapi"],
        "rrf_k": args.rrf_k,
        "dense_model": args.model,
        "bm25_tokenizer": "NFKC-lowercase-character-bigram",
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "corpus_scope": "full",
        "corpus_size": len(corpus),
        "dataset_size": len(dataset),
        "ks": list(KS),
    }}
    for variant, field in (("original", "original_question"), ("omission", "omission_query")):
        query_vectors = load_cached_embeddings(args.cache_dir, args.model, [row[field] for row in dataset])
        dense_scores = query_vectors @ corpus_vectors.T
        bm25_scores = np.asarray([bm25.scores(tokenize(row[field])) for row in dataset])
        summary, details = evaluate(dataset, corpus, dense_scores, bm25_scores, args.rrf_k)
        output[variant] = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: output[name]["summary"] for name in ("original", "omission")}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
