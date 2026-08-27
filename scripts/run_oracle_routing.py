#!/usr/bin/env python3
"""Evaluate category routing and a separate per-query Oracle upper bound.

For deployable comparisons, each fold selects the globally or categorically
best method using development-fold MRR only.  The per-query Oracle reported as
an upper bound is computed separately and must not be interpreted as a usable
retrieval method.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)
METHOD_ORDER = ("BM25", "Weighted Hybrid", "RRF", "Dense")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_details(results: dict) -> dict[str, dict]:
    return {row["eval_id"]: row for row in results["omission"]["details"]}


def mrr(eval_ids: list[str], details: dict[str, dict]) -> float:
    return statistics.fmean(1.0 / details[eval_id]["gold_rank"] for eval_id in eval_ids)


def choose_method(
    eval_ids: list[str],
    methods: dict[str, dict[str, dict]],
    tie_preference: list[str],
) -> tuple[str, dict[str, float]]:
    """Select the highest-development-MRR method with deterministic tie-breaking."""
    values = {name: mrr(eval_ids, details) for name, details in methods.items()}
    maximum = max(values.values())
    tied = {name for name, value in values.items() if abs(value - maximum) < 1e-12}
    selected = next(name for name in tie_preference if name in tied)
    return selected, values


def summarize(details: list[dict]) -> dict:
    count = len(details)
    return {
        "query_count": count,
        "recall": {
            f"R@{k}": sum(row["gold_rank"] <= k for row in details) / count for k in KS
        },
        "mrr": statistics.fmean(1.0 / row["gold_rank"] for row in details),
    }


def exact_mcnemar(left: list[dict], right: list[dict], k: int) -> dict:
    """Compare paired Recall@k successes using an exact two-sided McNemar test."""
    right_by_id = {row["eval_id"]: row for row in right}
    counts = [0, 0, 0, 0]
    for row in left:
        pair = (row["gold_rank"] <= k, right_by_id[row["eval_id"]]["gold_rank"] <= k)
        counts[{(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}[pair]] += 1
    discordant = counts[1] + counts[2]
    if discordant == 0:
        p_value = 1.0
    else:
        low = min(counts[1], counts[2])
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, i) for i in range(low + 1)) / 2**discordant)
    return {
        "both_success": counts[0],
        "left_only": counts[1],
        "right_only": counts[2],
        "both_failure": counts[3],
        "p_value": p_value,
    }


def category_summary(dataset: list[dict], details: list[dict]) -> list[dict]:
    metadata = {row["eval_id"]: row for row in dataset}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        groups[metadata[row["eval_id"]]["omission_category_name"]].append(row)
    return [
        {"category": category, **summarize(rows)} for category, rows in sorted(groups.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "annotation_filtered_276.jsonl")
    parser.add_argument("--dense", type=Path, default=ROOT / "outputs" / "retrieval_dense_results_276.json")
    parser.add_argument("--bm25", type=Path, default=ROOT / "outputs" / "bm25_retrieval_results.json")
    parser.add_argument("--rrf", type=Path, default=ROOT / "outputs" / "rrf_retrieval_results.json")
    parser.add_argument("--hybrid", type=Path, default=ROOT / "outputs" / "weighted_hybrid_retrieval_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "oracle_routing_results.json")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "oracle_routing_report.md")
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    metadata = {row["eval_id"]: row for row in dataset}
    loaded = {
        "Dense": json.loads(args.dense.read_text(encoding="utf-8")),
        "BM25": json.loads(args.bm25.read_text(encoding="utf-8")),
        "RRF": json.loads(args.rrf.read_text(encoding="utf-8")),
        "Weighted Hybrid": json.loads(args.hybrid.read_text(encoding="utf-8")),
    }
    methods = {name: index_details(results) for name, results in loaded.items()}
    expected_ids = set(metadata)
    for name, details in methods.items():
        if set(details) != expected_ids:
            raise ValueError(f"{name} evaluation IDs differ from dataset")

    hybrid_details = methods["Weighted Hybrid"]
    folds: dict[int, list[str]] = defaultdict(list)
    for eval_id, row in hybrid_details.items():
        folds[int(row["fold"])].append(eval_id)
    if sorted(folds) != list(range(1, len(folds) + 1)):
        raise ValueError("Hybrid fold IDs are not contiguous")
    for gold_id in {row["gold_chunk_id"] for row in dataset}:
        assigned = {hybrid_details[row["eval_id"]]["fold"] for row in dataset if row["gold_chunk_id"] == gold_id}
        if len(assigned) != 1:
            raise RuntimeError("Gold-chunk group leaked across folds")

    all_ids = set(expected_ids)
    oracle_details = []
    single_details = []
    fold_selections = []
    for fold_id in sorted(folds):
        test_ids = sorted(folds[fold_id])
        train_ids = sorted(all_ids - set(test_ids))
        global_method, global_values = choose_method(train_ids, methods, list(METHOD_ORDER))
        tie_preference = [global_method] + [name for name in METHOD_ORDER if name != global_method]
        train_by_category: dict[str, list[str]] = defaultdict(list)
        for eval_id in train_ids:
            train_by_category[metadata[eval_id]["omission_category_name"]].append(eval_id)
        category_methods = {}
        category_values = {}
        for category, eval_ids in sorted(train_by_category.items()):
            selected, values = choose_method(eval_ids, methods, tie_preference)
            category_methods[category] = selected
            category_values[category] = values
        fold_selections.append({
            "fold": fold_id,
            "train_size": len(train_ids),
            "test_size": len(test_ids),
            "best_single_method": global_method,
            "best_single_development_mrr": global_values,
            "category_methods": category_methods,
            "category_development_mrr": category_values,
        })
        for eval_id in test_ids:
            category = metadata[eval_id]["omission_category_name"]
            oracle_method = category_methods[category]
            source = methods[oracle_method][eval_id]
            oracle_details.append({
                **source,
                "selected_method": oracle_method,
                "category": category,
                "fold": fold_id,
            })
            single_source = methods[global_method][eval_id]
            single_details.append({
                **single_source,
                "selected_method": global_method,
                "fold": fold_id,
            })

    oracle_details.sort(key=lambda row: row["eval_id"])
    single_details.sort(key=lambda row: row["eval_id"])
    hybrid_rows = sorted(hybrid_details.values(), key=lambda row: row["eval_id"])
    output = {
        "configuration": {
            "selection_metric": "development MRR on omission queries",
            "grouping": "gold_chunk_id",
            "fold_count": len(folds),
            "candidate_methods": list(methods),
            "hybrid_development_scores": "cross-fitted out-of-fold predictions",
            "dataset_size": len(dataset),
            "corpus_size": loaded["BM25"]["configuration"]["corpus_size"],
        },
        "fold_selections": fold_selections,
        "best_single": {
            "summary": summarize(single_details),
            "details": single_details,
            "selection_counts": dict(Counter(row["selected_method"] for row in single_details)),
        },
        "oracle": {
            "summary": summarize(oracle_details),
            "details": oracle_details,
            "selection_counts": dict(Counter(row["selected_method"] for row in oracle_details)),
            "categories": category_summary(dataset, oracle_details),
        },
        "comparisons": {
            "oracle_vs_best_single": {f"R@{k}": exact_mcnemar(oracle_details, single_details, k) for k in KS},
            "oracle_vs_weighted_hybrid": {f"R@{k}": exact_mcnemar(oracle_details, hybrid_rows, k) for k in KS},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        "# Oracle routing評価", "",
        "- 条件: 276件・510文書",
        "- 分割: 根拠文書単位5-fold",
        "- 選択指標: 開発側省略質問MRR",
        "- 候補: Dense、BM25、RRF、Weighted Hybrid", "",
        "## 全体", "",
        "|方式|R@1|R@3|R@5|R@10|MRR|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, block in (("Best single", output["best_single"]), ("Oracle routing", output["oracle"])):
        summary = block["summary"]
        lines.append(f"|{label}|{pct(summary['recall']['R@1'])}|{pct(summary['recall']['R@3'])}|{pct(summary['recall']['R@5'])}|{pct(summary['recall']['R@10'])}|{summary['mrr']:.4f}|")
    lines += ["", "## Fold別選択", "",
              "|fold|最良単一|カテゴリ別選択|",
              "|---:|---|---|"]
    for row in fold_selections:
        selections = "、".join(f"{category}: {method}" for category, method in row["category_methods"].items())
        lines.append(f"|{row['fold']}|{row['best_single_method']}|{selections}|")
    lines += ["", "## Oracle対Best single", "",
              "|k|Oracleのみ成功|Best singleのみ成功|正確McNemar p|",
              "|---:|---:|---:|---:|"]
    for k in KS:
        row = output["comparisons"]["oracle_vs_best_single"][f"R@{k}"]
        lines.append(f"|{k}|{row['left_only']}|{row['right_only']}|{row['p_value']:.6g}|")
    lines += ["", "## Oracleのカテゴリ別結果", "",
              "|カテゴリ|n|R@1|R@3|R@5|R@10|MRR|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in output["oracle"]["categories"]:
        lines.append(f"|{row['category']}|{row['query_count']}|{pct(row['recall']['R@1'])}|{pct(row['recall']['R@3'])}|{pct(row['recall']['R@5'])}|{pct(row['recall']['R@10'])}|{row['mrr']:.4f}|")
    report = "\n".join(lines) + "\n"
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
