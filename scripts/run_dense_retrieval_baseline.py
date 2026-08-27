#!/usr/bin/env python3
"""Dense-retrieval baseline for original and omission queries.

The implementation follows Sasaki and Yamamoto (NLP 2026): one evidence page
per chunk, text-embedding-3-small, cosine ranking, Recall@k, top-k score
standard deviation, mean pairwise similarity (MPS), and Clarity=MPS-sigma.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_CORPUS = ROOT / "omission_query_dataset" / "jdocqa_510_corpus.jsonl"
DEFAULT_CACHE = ROOT / "outputs" / "embedding_cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "retrieval_dense_results_276.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize row vectors so their inner product is cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding API returned a zero vector")
    return matrix / norms


def cache_key(model: str, texts: list[str]) -> str:
    """Hash the model name and ordered text sequence for an exact cache key."""
    digest = hashlib.sha256()
    digest.update(model.encode())
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode())
    return digest.hexdigest()


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
    cache_dir: Path,
    batch_size: int,
) -> np.ndarray:
    """Load exact cached embeddings or call the API in batches and cache them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(model, texts)}.npy"
    if path.exists():
        return np.load(path)
    vectors = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=model, input=texts[start : start + batch_size])
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    np.save(path, matrix)
    return matrix


def semantic_consistency(document_vectors: np.ndarray) -> tuple[float, float, float]:
    """Return mean pairwise similarity, its SD, and the derived clarity value."""
    similarities = document_vectors @ document_vectors.T
    upper = similarities[np.triu_indices(len(document_vectors), k=1)]
    if len(upper) == 0:
        return math.nan, math.nan, math.nan
    mps = float(np.mean(upper))
    sigma = float(np.std(upper, ddof=0))
    return mps, sigma, mps - sigma


def evaluate(
    queries: list[dict],
    query_vectors: np.ndarray,
    corpus: list[dict],
    corpus_vectors: np.ndarray,
    ks: Iterable[int],
) -> tuple[dict, list[dict]]:
    """Compute retrieval metrics and auxiliary top-k consistency statistics."""
    ks = sorted(set(ks))
    maximum_k = max(ks)
    recalls = {k: 0 for k in ks}
    score_std = {k: [] for k in ks}
    mps = {k: [] for k in ks}
    pair_sigma = {k: [] for k in ks}
    clarity = {k: [] for k in ks}
    details = []
    reciprocal_rank_sum = 0.0

    scores = query_vectors @ corpus_vectors.T
    for query, row in zip(queries, scores):
        full_ranking = np.argsort(-row)
        ranking = full_ranking[:maximum_k]
        ranked_ids = [corpus[index]["chunk_id"] for index in ranking]
        rank = next(
            position for position, index in enumerate(full_ranking, 1)
            if corpus[index]["chunk_id"] == query["gold_chunk_id"]
        )
        reciprocal_rank_sum += 1.0 / rank
        metrics = {}
        for k in ks:
            top = ranking[:k]
            recalls[k] += rank is not None and rank <= k
            score_std[k].append(float(np.std(row[top], ddof=0)))
            current_mps, current_sigma, current_clarity = semantic_consistency(corpus_vectors[top])
            if k > 1:
                mps[k].append(current_mps)
                pair_sigma[k].append(current_sigma)
                clarity[k].append(current_clarity)
            metrics[str(k)] = {"score_std": score_std[k][-1]}
            if k > 1:
                metrics[str(k)].update({
                    "mps": current_mps,
                    "pairwise_sigma": current_sigma,
                    "clarity": current_clarity,
                })
        details.append({
            "eval_id": query["eval_id"],
            "gold_chunk_id": query["gold_chunk_id"],
            "gold_rank": rank,
            "top_chunk_ids": ranked_ids,
            "top_scores": [float(row[index]) for index in ranking],
            "metrics": metrics,
        })

    count = len(queries)
    summary = {
        "query_count": count,
        "recall": {f"R@{k}": recalls[k] / count for k in ks},
        "mrr": reciprocal_rank_sum / count,
        "mean_score_std": {f"top_{k}": float(np.nanmean(score_std[k])) for k in ks},
        "mean_mps": {f"top_{k}": float(np.nanmean(mps[k])) for k in ks if k > 1},
        "mean_pairwise_sigma": {
            f"top_{k}": float(np.nanmean(pair_sigma[k])) for k in ks if k > 1
        },
        "mean_clarity": {f"top_{k}": float(np.nanmean(clarity[k])) for k in ks if k > 1},
    }
    return summary, details


def select_corpus(corpus: list[dict], dataset: list[dict], scope: str) -> list[dict]:
    """Select the full corpus or only documents that are gold in the query set."""
    if scope == "full":
        return corpus
    gold_ids = {row["gold_chunk_id"] for row in dataset}
    selected = [row for row in corpus if row["chunk_id"] in gold_ids]
    if {row["chunk_id"] for row in selected} != gold_ids:
        raise ValueError("Some gold chunks are missing from the corpus")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Evaluation-query JSONL.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Retrieval-document JSONL.")
    parser.add_argument("--corpus-scope", choices=("full", "gold"), default="full", help="Search all documents or only represented gold documents.")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--batch-size", type=int, default=128, help="Texts sent in each embedding API request.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Directory for exact .npy embedding caches.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading or generating embeddings.")
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    corpus = select_corpus(read_jsonl(args.corpus), dataset, args.corpus_scope)
    gold_ids = {row["gold_chunk_id"] for row in dataset}
    corpus_ids = {row["chunk_id"] for row in corpus}
    missing = gold_ids - corpus_ids
    if missing:
        raise ValueError(f"{len(missing)} gold chunks are missing")
    print(
        f"Ready: {len(dataset)} queries, {len(corpus)} chunks, "
        f"model={args.model}, scope={args.corpus_scope}"
    )
    if args.dry_run:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI()
    corpus_vectors = normalized(
        embed_texts(client, [row["text"] for row in corpus], args.model, args.cache_dir, args.batch_size)
    )
    output = {
        "configuration": {
            "model": args.model,
            "embedding_dimensions": int(corpus_vectors.shape[1]),
            "corpus_scope": args.corpus_scope,
            "corpus_size": len(corpus),
            "dataset_size": len(dataset),
            "similarity": "cosine",
            "ks": [1, 3, 5, 10],
        }
    }
    for name, field in (("original", "original_question"), ("omission", "omission_query")):
        vectors = normalized(
            embed_texts(
                client,
                [row[field] for row in dataset],
                args.model,
                args.cache_dir,
                args.batch_size,
            )
        )
        summary, details = evaluate(dataset, vectors, corpus, corpus_vectors, [1, 3, 5, 10])
        output[name] = {"summary": summary, "details": details}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: output[name]["summary"] for name in ("original", "omission")},
                     ensure_ascii=False, indent=2))
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
