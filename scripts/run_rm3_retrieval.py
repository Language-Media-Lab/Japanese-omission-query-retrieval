#!/usr/bin/env python3
"""Evaluate the paper's RM3-style feedback over character-bigram BM25.

This is not a reproduction of standard word-based RM3.  It estimates a
relevance model from the initial BM25 top documents, keeps its highest-weight
character bigrams, interpolates them with the original query model, and
reranks the corpus with weighted BM25.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from run_bm25_retrieval_baseline import BM25, KS, read_jsonl, tokenize


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_CORPUS = ROOT / "omission_query_dataset" / "jdocqa_510_corpus.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "rm3_retrieval_results.json"


def weighted_bm25_scores(model: BM25, weights: dict[str, float]) -> list[float]:
    """Score all documents while retaining interpolated query-term weights."""
    scores = [0.0] * len(model.lengths)
    for term, query_weight in weights.items():
        if term not in model.idf:
            continue
        for document_id, frequency in model.postings[term]:
            norm = 1.0 - model.b + model.b * model.lengths[document_id] / model.average_length
            contribution = model.idf[term] * frequency * (model.k1 + 1.0) / (
                frequency + model.k1 * norm
            )
            scores[document_id] += query_weight * contribution
    return scores


def rm3_query_model(
    query_tokens: list[str],
    initial_scores: list[float],
    model: BM25,
    fb_docs: int,
    fb_terms: int,
    original_weight: float,
    mu: float,
    collection_counts: Counter[str],
    collection_length: int,
) -> tuple[dict[str, float], list[int]]:
    """Build the interpolated query model and return feedback-document indices."""
    ranking = sorted(range(len(initial_scores)), key=lambda i: (-initial_scores[i], i))[:fb_docs]
    query_counts = Counter(query_tokens)
    log_weights = []
    for document_id in ranking:
        frequencies = model.term_frequencies[document_id]
        length = model.lengths[document_id]
        log_probability = 0.0
        for term, count in query_counts.items():
            collection_probability = collection_counts[term] / collection_length
            probability = (frequencies[term] + mu * collection_probability) / (length + mu)
            if probability > 0.0:
                log_probability += count * math.log(probability)
        log_weights.append(log_probability)
    maximum = max(log_weights)
    document_weights = [math.exp(value - maximum) for value in log_weights]
    total_weight = sum(document_weights)
    document_weights = [value / total_weight for value in document_weights]

    relevance_model: Counter[str] = Counter()
    for document_id, document_weight in zip(ranking, document_weights):
        frequencies = model.term_frequencies[document_id]
        length = model.lengths[document_id]
        for term, frequency in frequencies.items():
            relevance_model[term] += document_weight * frequency / length
    expansion = dict(relevance_model.most_common(fb_terms))
    expansion_total = sum(expansion.values()) or 1.0
    expansion = {term: value / expansion_total for term, value in expansion.items()}
    query_total = sum(query_counts.values()) or 1
    original = {term: count / query_total for term, count in query_counts.items()}
    combined = Counter({term: original_weight * value for term, value in original.items()})
    combined.update({term: (1.0 - original_weight) * value for term, value in expansion.items()})
    return dict(combined), ranking


def evaluate(dataset: list[dict], field: str, corpus: list[dict], model: BM25, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Evaluate one query field and retain feedback terms for error analysis."""
    corpus_ids = [row["chunk_id"] for row in corpus]
    corpus_index = {chunk_id: index for index, chunk_id in enumerate(corpus_ids)}
    recalls = {k: 0 for k in KS}
    reciprocal_rank_sum = 0.0
    details = []
    collection_counts: Counter[str] = Counter()
    for frequencies in model.term_frequencies:
        collection_counts.update(frequencies)
    collection_length = sum(collection_counts.values())
    for row in dataset:
        query_tokens = tokenize(row[field])
        initial_scores = model.scores(query_tokens)
        weights, feedback_documents = rm3_query_model(
            query_tokens, initial_scores, model, args.fb_docs, args.fb_terms,
            args.original_query_weight, args.mu, collection_counts, collection_length,
        )
        scores = weighted_bm25_scores(model, weights)
        ranking = sorted(range(len(corpus)), key=lambda i: (-scores[i], corpus_ids[i]))
        gold_rank = ranking.index(corpus_index[row["gold_chunk_id"]]) + 1
        reciprocal_rank_sum += 1.0 / gold_rank
        for k in KS:
            recalls[k] += gold_rank <= k
        top = ranking[:max(KS)]
        details.append({
            "eval_id": row["eval_id"],
            "gold_chunk_id": row["gold_chunk_id"],
            "gold_rank": gold_rank,
            "top_chunk_ids": [corpus_ids[index] for index in top],
            "top_scores": [scores[index] for index in top],
            "feedback_chunk_ids": [corpus_ids[index] for index in feedback_documents],
            "expansion_terms": sorted(weights.items(), key=lambda item: -item[1])[:args.fb_terms],
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
    parser.add_argument("--fb-docs", type=int, default=10, help="Number of initial top documents used as pseudo-relevant feedback.")
    parser.add_argument("--fb-terms", type=int, default=10, help="Number of highest-weight feedback bigrams retained.")
    parser.add_argument("--original-query-weight", type=float, default=0.5, help="Interpolation weight assigned to the original query model.")
    parser.add_argument("--mu", type=float, default=1000.0, help="Dirichlet smoothing parameter for feedback-document likelihoods.")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 term-frequency saturation parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 document-length normalization parameter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    args = parser.parse_args()
    dataset = read_jsonl(args.dataset)
    corpus = read_jsonl(args.corpus)
    model = BM25([tokenize(row["text"]) for row in corpus], args.k1, args.b)
    output = {"configuration": {
        "method": "BM25+RM3",
        "tokenizer": "NFKC-lowercase-character-bigram",
        "fb_docs": args.fb_docs,
        "fb_terms": args.fb_terms,
        "original_query_weight": args.original_query_weight,
        "dirichlet_mu": args.mu,
        "bm25_k1": args.k1,
        "bm25_b": args.b,
        "corpus_scope": "full",
        "corpus_size": len(corpus),
        "dataset_size": len(dataset),
        "ks": list(KS),
    }}
    for variant, field in (("original", "original_question"), ("omission", "omission_query")):
        summary, details = evaluate(dataset, field, corpus, model, args)
        output[variant] = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: output[name]["summary"] for name in ("original", "omission")}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
