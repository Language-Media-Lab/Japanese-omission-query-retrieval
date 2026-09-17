# Revised results

`review/experiments.json`: main results are `tuned.count`; `fixed.count_*` are supplemental fixed-parameter comparisons. `binary` is a legacy diagnostic, not the main result.

`review/candidate_ranks.json`: ranks for every parameter candidate, used for held-out selection and nested routing.

`review/analysis.json`: cluster-aware significance, method interactions, rank transitions, and nested routing (`routing.count`).

`review/verification.json`: historical independent score/annotation verification. Local source hashes and environment describe that original verification, not the adapted public scripts.

`submission/`: historical results retained exclusively for comparison and reproduction assertions. They are superseded by `review/` and must not be reported as the current paper's results.

Run `python scripts/verify_release.py` to independently check current saved metrics, dataset alignment, fold separation, routing metrics, and distributed file hashes without API access or source PDFs.

`review/supplemental_checks.json` contains the retained quality-audit aggregates and re-tuned subset sensitivity results (supplemental, not replacements for the 276-query main results).
