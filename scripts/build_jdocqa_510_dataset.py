#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""先行研究準拠106件にtrain由来404件を加えた拡張セットを構築する。

質問の品質条件は106件版から変更しない。

* answer_type == "4"（自由記述）
* no_reason == "1"（回答可能）
* type_of_image == ""（画像・表・図に依存しない）

変更するのは対象splitだけで、train、validation、testをすべて使用する。
106件のコアセットは別ファイルとしてそのまま保持する。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from build_jdocqa_106_dataset import build_corpus, build_records, write_jsonl


DATASET_DIR = Path("JDocQA/dataset/annotation_files")
OUTPUT_DIR = Path("omission_query_dataset")
INPUT_FILES = [
    "jdocqa_train_all.json",
    "jdocqa_validation_all.json",
    "jdocqa_test_all.json",
]
EXPECTED_SPLIT_COUNTS = {"train": 404, "validation": 49, "test": 57}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    records = build_records(args.dataset_dir, INPUT_FILES)
    corpus = build_corpus(records)
    split_counts = Counter(record["split"] for record in records)

    if len(records) != 510:
        raise SystemExit(f"expected 510 records, got {len(records)}")
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(
            f"unexpected split counts: expected {EXPECTED_SPLIT_COUNTS}, got {dict(split_counts)}"
        )
    if len(corpus) != len(records):
        raise SystemExit(
            f"expected one unique evidence page per question, got {len(corpus)} pages "
            f"for {len(records)} questions"
        )

    source_path = args.output_dir / "jdocqa_510_source.jsonl"
    corpus_path = args.output_dir / "jdocqa_510_corpus.jsonl"
    write_jsonl(source_path, records)
    write_jsonl(corpus_path, corpus)

    print(f"source records: {len(records)}")
    print(f"corpus chunks: {len(corpus)}")
    print(f"split counts: {dict(split_counts)}")
    print(f"source: {source_path}")
    print(f"corpus: {corpus_path}")


if __name__ == "__main__":
    main()
