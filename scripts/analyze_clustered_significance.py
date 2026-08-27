#!/usr/bin/env python3
"""Cluster-aware paired inference for weighted hybrid versus BM25.

Questions sharing a gold document form one cluster.  The paired permutation
test flips each cluster's aggregate difference as a unit, while the confidence
interval resamples whole clusters with replacement.  This preserves the known
within-document dependence among the 276 evaluation questions.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BM25_PATH = ROOT / "outputs/bm25_retrieval_results.json"
HYBRID_PATH = ROOT / "outputs/weighted_hybrid_retrieval_results.json"
OUT_PATH = ROOT / "outputs/clustered_significance_hybrid_vs_bm25.json"
KS = (1, 3, 5, 10)
SEED = 20260820
N_PERM = 100_000
N_BOOT = 100_000


def load_details(path: Path):
    """Load per-query omission results from a retrieval-result JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)["omission"]["details"]


def score(rank: int, metric: str) -> float:
    """Convert one gold rank to reciprocal rank or a Recall@k indicator."""
    if metric == "MRR":
        return 1.0 / rank
    return float(rank <= int(metric.split("@")[1]))


def main() -> None:
    bm25 = {row["eval_id"]: row for row in load_details(BM25_PATH)}
    hybrid = {row["eval_id"]: row for row in load_details(HYBRID_PATH)}
    if bm25.keys() != hybrid.keys():
        raise ValueError("BM25 and hybrid eval_id sets differ")

    rng = np.random.default_rng(SEED)
    metrics = ["MRR", *(f"R@{k}" for k in KS)]
    results = {}
    for metric in metrics:
        # Keep every question sharing a gold document in the same inference unit.
        clusters = defaultdict(list)
        for eval_id in sorted(bm25):
            b = bm25[eval_id]
            h = hybrid[eval_id]
            if b["gold_chunk_id"] != h["gold_chunk_id"]:
                raise ValueError(f"gold chunk mismatch: {eval_id}")
            clusters[b["gold_chunk_id"]].append(
                score(h["gold_rank"], metric) - score(b["gold_rank"], metric)
            )

        arrays = [np.asarray(v, dtype=float) for v in clusters.values()]
        cluster_sums = np.asarray([v.sum() for v in arrays])
        observed = float(cluster_sums.sum() / len(bm25))

        extreme = 0
        chunk = 10_000
        # Draw sign-flip permutations in chunks to bound peak memory use.
        for start in range(0, N_PERM, chunk):
            n = min(chunk, N_PERM - start)
            signs = rng.choice((-1.0, 1.0), size=(n, len(cluster_sums)))
            permuted = signs @ cluster_sums / len(bm25)
            extreme += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        p_value = (extreme + 1) / (N_PERM + 1)

        boot = np.empty(N_BOOT, dtype=float)
        for i in range(N_BOOT):
            sampled = rng.integers(0, len(arrays), size=len(arrays))
            numerator = sum(arrays[j].sum() for j in sampled)
            denominator = sum(len(arrays[j]) for j in sampled)
            boot[i] = numerator / denominator
        ci_low, ci_high = np.quantile(boot, (0.025, 0.975))

        results[metric] = {
            "difference_hybrid_minus_bm25": observed,
            "cluster_sign_flip_two_sided_p": float(p_value),
            "cluster_bootstrap_95ci": [float(ci_low), float(ci_high)],
        }

    output = {
        "comparison": "weighted_hybrid_vs_bm25",
        "unit_count": len(bm25),
        "cluster_count": len({row["gold_chunk_id"] for row in bm25.values()}),
        "cluster_key": "gold_chunk_id",
        "primary_metric": "MRR",
        "secondary_metrics": [f"R@{k}" for k in KS],
        "seed": SEED,
        "permutations": N_PERM,
        "bootstrap_resamples": N_BOOT,
        "metrics": results,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
