#!/usr/bin/env python3
"""Evaluate cached Query2doc pseudo-documents with character-bigram BM25.

Query-term counts are multiplied by five (configurable), then added to
pseudo-document term counts. Components are tokenized separately.
Generation is deliberately separate so cached outputs can be audited and
reused without another API call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from run_review_experiments import expansion_weights
from run_rm3_retrieval import weighted_bm25_scores

from run_bm25_retrieval_baseline import BM25, KS, read_jsonl, tokenize


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_CORPUS = ROOT / "omission_query_dataset" / "jdocqa_510_corpus.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Evaluation-query JSONL.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Retrieval-document JSONL.")
    parser.add_argument("--pseudo-documents", type=Path, required=True, help="Cached generic or category-conditioned Query2doc JSONL.")
    parser.add_argument("--method-name", required=True, help="Method label stored in the result metadata.")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 term-frequency saturation parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 document-length normalization parameter.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--query-repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.query_repetitions < 1:
        parser.error("--query-repetitions must be positive")

    dataset = read_jsonl(args.dataset)
    corpus = read_jsonl(args.corpus)
    generated_rows = read_jsonl(args.pseudo_documents)
    generated = {
        row["eval_id"]: row for row in generated_rows
        if row.get("status") == "ok" and row.get("pseudo_document")
    }
    missing = [row["eval_id"] for row in dataset if row["eval_id"] not in generated]
    if missing:
        raise ValueError(f"Missing {len(missing)} pseudo-documents; first: {missing[:5]}")
    corpus_ids = [row["chunk_id"] for row in corpus]
    corpus_index = {chunk_id: index for index, chunk_id in enumerate(corpus_ids)}
    model = BM25([tokenize(row["text"]) for row in corpus], args.k1, args.b)

    recalls = {k: 0 for k in KS}
    reciprocal_rank_sum = 0.0
    details = []
    for row in dataset:
        pseudo = generated[row["eval_id"]]["pseudo_document"]
        weights = expansion_weights(row['omission_query'], pseudo, args.query_repetitions)
        scores = weighted_bm25_scores(model, weights)
        ranking = sorted(range(len(corpus)), key=lambda i: (-scores[i], corpus_ids[i]))
        gold_rank = ranking.index(corpus_index[row["gold_chunk_id"]]) + 1
        reciprocal_rank_sum += 1.0 / gold_rank
        for k in KS:
            recalls[k] += gold_rank <= k
        top = ranking[: max(KS)]
        details.append({
            "eval_id": row["eval_id"],
            "gold_chunk_id": row["gold_chunk_id"],
            "gold_rank": gold_rank,
            "top_chunk_ids": [corpus_ids[index] for index in top],
            "top_scores": [scores[index] for index in top],
        })

    count = len(dataset)
    source_models = sorted({row.get("model") for row in generated.values()})
    output = {
        "configuration": {
            "method": args.method_name,
            "query": "separate_token_bags",
            "query_repetitions": args.query_repetitions,
            "tokenizer": "NFKC-lowercase-character-bigram",
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "corpus_size": len(corpus),
            "dataset_size": count,
            "generation_models": source_models,
            "pseudo_documents": str(args.pseudo_documents),
            "ks": list(KS),
        },
        "omission": {
            "summary": {
                "query_count": count,
                "recall": {f"R@{k}": recalls[k] / count for k in KS},
                "mrr": reciprocal_rank_sum / count,
            },
            "details": details,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["omission"]["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
