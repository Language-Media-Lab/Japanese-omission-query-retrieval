# Quality notes for the revised evaluation

The evaluation set and generated texts were retained, not regenerated or filtered to improve results. Retrieval measures rediscovery of one original evidence page; other potentially relevant pages were not judged.

A subsequent Codex-assisted audit examined 85 cases where recorded deletion text did not mechanically reproduce the query. It classified 52 as grammatical adjustments only; the other 33 had other findings, not necessarily invalidity. Recorded omission descriptions can differ from actual edits because of both generation and post-processing.

Among 552 current generated texts, 545 had no clear interruption, four had strong interruption indications, and three could also be interpreted as lists. Provider completion status was not stored; the cache's `status=ok` only reflects the historical local acceptance check. These are not independent human judgments. Excluding the union of seven affected queries in a supplemental re-tuned analysis retained Hybrid as the highest mean MRR method.

The generation script retains the historical procedure for transparency; this release does not claim to have fixed or rerun generation. Generated text, exact prompt templates, reconstructed inputs, and retrieval weighting are documented separately. Reconstructed inputs are not captured API request logs. Category-conditioned expansion and category routing use gold labels.
