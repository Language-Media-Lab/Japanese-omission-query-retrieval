# -*- coding: utf-8 -*-

"""
Build the 106-question JDocQA subset used as source data for omission-query
generation.

Selection rule, aligned with the prior-work setup:
- annotation split: validation + test
- answer_type == "4"  (open-ended/generative answer)
- no_reason == "1"   (answerable)
- type_of_image == "" (no image/table/figure type annotation)

Outputs:
- omission_query_dataset/jdocqa_106_source.jsonl
- omission_query_dataset/jdocqa_106_corpus.jsonl

Each source record is one query. Each corpus record is the corresponding
evidence page chunk, so the retrieval setting is 106 queries and 106 chunks.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASET_DIR = Path("JDocQA/dataset/annotation_files")
OUTPUT_DIR = Path("omission_query_dataset")

INPUT_FILES = [
    "jdocqa_validation_all.json",
    "jdocqa_test_all.json",
]


def clean_question(question: Any) -> str:
    """質問末尾の回答形式指示を削り、検索クエリとして使う本文だけにする。"""
    text = str(question).strip()
    text = re.sub(r"\s*解答は[^。]*してください。?\s*$", "", text)
    return text.strip()


def make_qa_id(row: Dict[str, Any], split: str) -> str:
    """同じ入力行から常に同じIDを作るため、主要フィールドをハッシュ化する。"""
    raw = "|".join(
        [
            split,
            str(row.get("pdf_name", "")),
            str(row.get("question_page_number", "")),
            str(row.get("question", "")),
            str(row.get("answer", "")),
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """JDocQAのアノテーションファイルを1行ずつJSONとして読み込む。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def choose_chunk_text(row: Dict[str, Any]) -> str:
    """証拠文として使える本文フィールドを、優先順に探して返す。"""
    for key in ["context", "text_from_ocr_pdf", "text_from_pdf", "original_context"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def is_target_row(row: Dict[str, Any]) -> bool:
    """先行研究の条件に合う、生成回答・回答可能・画像なしの質問だけを残す。"""
    return (
        row.get("answer_type") == "4"
        and row.get("no_reason") == "1"
        and str(row.get("type_of_image", "")).strip() == ""
    )


def build_records(
    dataset_dir: Path, input_files: Iterable[str] = INPUT_FILES
) -> List[Dict[str, Any]]:
    """指定したJDocQA splitから、後続生成に必要な質問レコードを作る。"""
    records: List[Dict[str, Any]] = []

    for file_name in input_files:
        path = dataset_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)

        for row in read_jsonl(path):
            if not is_target_row(row):
                continue

            # PDF名とページ番号をchunk_idにして、質問と証拠ページを1対1で結び付ける。
            qa_id = make_qa_id(row, file_name)
            pdf_name = str(row.get("pdf_name", ""))
            page = str(row.get("question_page_number", ""))
            chunk_id = f"{pdf_name}#page={page}"

            record = {
                "qa_id": qa_id,
                "annotation_file": file_name,
                "split": file_name.replace("jdocqa_", "").replace("_all.json", ""),
                "pdf_category": str(row.get("pdf_category", "")),
                "pdf_name": pdf_name,
                "page": page,
                "question_number": str(row.get("question_number", "")),
                "original_question": str(row.get("question", "")).strip(),
                "question": clean_question(row.get("question", "")),
                "answer": str(row.get("answer", "")).strip(),
                "original_answer": str(row.get("original_answer", "")).strip(),
                "normalized_answer": str(row.get("normalized_answer", "")).strip(),
                "answer_type": str(row.get("answer_type", "")),
                "no_reason": str(row.get("no_reason", "")),
                "type_of_image": str(row.get("type_of_image", "")),
                "chunk_id": chunk_id,
                "evidence": choose_chunk_text(row),
            }
            records.append(record)

    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """1レコード1行のJSONLとして、後続スクリプトが読みやすい形式で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_corpus(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """検索評価用に、質問レコードから証拠ページだけを切り出したcorpusを作る。"""
    corpus = []
    seen = set()

    for record in records:
        chunk_id = record["chunk_id"]
        # 106問それぞれに一意な証拠チャンクがある前提なので、重複はデータ不整合として止める。
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)

        corpus.append(
            {
                "chunk_id": chunk_id,
                "qa_id": record["qa_id"],
                "pdf_name": record["pdf_name"],
                "pdf_category": record["pdf_category"],
                "page": record["page"],
                "text": record["evidence"],
            }
        )

    return corpus


def main() -> None:
    """コマンドライン引数を受け取り、source/corpusの2種類のJSONLを生成する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    records = build_records(args.dataset_dir)
    corpus = build_corpus(records)

    source_path = args.output_dir / "jdocqa_106_source.jsonl"
    corpus_path = args.output_dir / "jdocqa_106_corpus.jsonl"

    write_jsonl(source_path, records)
    write_jsonl(corpus_path, corpus)

    print(f"source records: {len(records)}")
    print(f"corpus chunks: {len(corpus)}")
    print(f"source: {source_path}")
    print(f"corpus: {corpus_path}")

    # 固定データセットとして106件になることを最後に検証する。
    if len(records) != 106:
        raise SystemExit(f"expected 106 records, got {len(records)}")


if __name__ == "__main__":
    main()
