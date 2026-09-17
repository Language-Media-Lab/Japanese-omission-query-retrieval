# Revision 2026-09-17

- Publish current v4 manuscript and source, including three consolidated research questions and concise dataset terminology.
- Replace primary results with corrected query-frequency evaluation and development-selected comparison; retain historical results in `results/submission/`.
- Publish candidate ranks, nested-routing analysis, frozen pseudo-documents, exact prompts and reconstructed inputs.
- Fix standalone BM25 query frequency and Query2doc separate-bag weighting.
- Correct the annotation guide to the actual answerability assumption and seven-category design. Document batch-local annotator identifiers outside the manuscript.
- Preserve the 276 evaluation records and all generated text. Public pseudo-document records omit provider response IDs only.

Full score recomputation requires upstream corpus data and historical embeddings. Released ranks support API-free result checking and statistical reanalysis. The machine-assisted quality audit is distinguished from the original human annotations. No new API generation was performed.

Chapter 4 now explains the purpose of development-only parameter selection in plain language. Detailed generation settings and statistical procedures are in the appendices; duplicate appendix prefixes are corrected.

- Reorganize the discussion around the three research questions, group limitations into three categories, and normalize the final section typography.
