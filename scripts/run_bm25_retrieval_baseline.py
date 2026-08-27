#!/usr/bin/env python3
"""Evaluate original and omission queries with character-bigram BM25.

The corpus contains one retrieval document per JDocQA evidence page.  Ranking
ties are resolved by document ID so repeated runs are deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_CORPUS = ROOT / "omission_query_dataset" / "jdocqa_510_corpus.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "bm25_retrieval_results.json"
KS = (1, 3, 5, 10)


def read_jsonl(path: Path) -> list[dict]:
    """Load a UTF-8 JSON Lines file into memory."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    """Return overlapping character bigrams after conservative normalization.

    Character n-grams avoid a dictionary/version dependency and work for mixed
    Japanese, Latin text, numerals, and OCR noise. Whitespace is removed, while
    punctuation is retained because it provides useful boundaries.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[index : index + 2] for index in range(len(compact) - 1)]


class BM25:
    """Small BM25Okapi implementation exposing scores needed by later methods."""
    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.lengths = [len(tokens) for tokens in documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.term_frequencies = [Counter(tokens) for tokens in documents]
        document_frequencies: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_id, frequencies in enumerate(self.term_frequencies):
            document_frequencies.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self.postings[term].append((document_id, frequency))
        count = len(documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        """Return one BM25 score per corpus document."""
        scores = [0.0] * len(self.lengths)
        for term in set(query_tokens):
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for document_id, frequency in self.postings[term]:
                norm = 1.0 - self.b + self.b * self.lengths[document_id] / self.average_length
                scores[document_id] += idf * frequency * (self.k1 + 1.0) / (
                    frequency + self.k1 * norm
                )
        return scores


def select_corpus(corpus: list[dict], dataset: list[dict], scope: str) -> list[dict]:
    """Select the full corpus or only documents that are gold for this dataset."""
    if scope == "full":
        return corpus
    gold_ids = {row["gold_chunk_id"] for row in dataset}
    selected = [row for row in corpus if row["chunk_id"] in gold_ids]
    if {row["chunk_id"] for row in selected} != gold_ids:
        raise ValueError("Some gold chunks are missing from the corpus")
    return selected


def evaluate(dataset: list[dict], field: str, corpus: list[dict], model: BM25) -> tuple[dict, list[dict]]:
    """Compute Recall@k, MRR, and per-query rankings for one query field."""
    recalls = {k: 0 for k in KS}
    reciprocal_rank_sum = 0.0
    details = []
    for row in dataset:
        scores = model.scores(tokenize(row[field]))
        ranking = sorted(range(len(corpus)), key=lambda index: (-scores[index], corpus[index]["chunk_id"]))
        gold_rank = next(
            rank for rank, index in enumerate(ranking, 1)
            if corpus[index]["chunk_id"] == row["gold_chunk_id"]
        )
        reciprocal_rank_sum += 1.0 / gold_rank
        for k in KS:
            recalls[k] += gold_rank <= k
        top = ranking[: max(KS)]
        details.append({
            "eval_id": row["eval_id"],
            "gold_chunk_id": row["gold_chunk_id"],
            "gold_rank": gold_rank,
            "top_chunk_ids": [corpus[index]["chunk_id"] for index in top],
            "top_scores": [scores[index] for index in top],
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
    parser.add_argument("--corpus-scope", choices=("full", "gold"), default="full", help="Search all documents or only gold documents represented in the evaluation set.")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 term-frequency saturation parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 document-length normalization parameter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without scoring queries.")
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    corpus = select_corpus(read_jsonl(args.corpus), dataset, args.corpus_scope)
    if not {row["gold_chunk_id"] for row in dataset} <= {row["chunk_id"] for row in corpus}:
        raise ValueError("Some gold chunks are missing from the corpus")
    print(f"Ready: {len(dataset)} queries, {len(corpus)} chunks, scope={args.corpus_scope}")
    if args.dry_run:
        return

    model = BM25([tokenize(row["text"]) for row in corpus], k1=args.k1, b=args.b)
    output = {"configuration": {
        "method": "BM25Okapi",
        "tokenizer": "NFKC-lowercase-character-bigram",
        "k1": args.k1,
        "b": args.b,
        "corpus_scope": args.corpus_scope,
        "corpus_size": len(corpus),
        "dataset_size": len(dataset),
        "ks": list(KS),
    }}
    for name, field in (("original", "original_question"), ("omission", "omission_query")):
        summary, details = evaluate(dataset, field, corpus, model)
        output[name] = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: output[name]["summary"] for name in ("original", "omission")}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
