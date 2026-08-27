#!/usr/bin/env python3
"""Generate and cache Query2doc pseudo-documents for omitted queries.

Only the omission query is sent in generic mode.  Category mode additionally
sends the gold omission-category name and definition, but never the original
question, omitted string, answer, or gold document.  API generation has no
seed; ``--seed`` controls only deterministic retry-delay jitter.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from run_bm25_retrieval_baseline import read_jsonl


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "annotation_filtered_276.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "query2doc"
PROMPT_VERSION = "query2doc-ja-v1"

CATEGORY_GUIDANCE = {
    "person_org": "人物・組織情報（行為者、実施主体、運営者、所有者、情報源など）",
    "location": "場所・管轄情報（地域、所在地、会場、適用地域など）",
    "time": "時間情報（年度、日付、期間、時刻、期限など）",
    "what": "対象・内容・属性情報（制度、商品、条件、金額、数量など）",
    "ga_case": "ガ格の項（動作・状態の主体など）",
    "wo_case": "ヲ格の項（動作の対象など）",
    "ni_case": "ニ格の項（相手、到達点、対象など）",
}

FEW_SHOTS = """次の例のように、質問への回答を含みそうな短い文書を作ってください。

質問: 利用できる時間帯はいつですか？
擬似文書: サービスを利用できる時間帯や受付時間、曜日、休業日について案内する。利用可能な開始時刻と終了時刻、例外となる日程も確認できる。

質問: 申請に必要なものは何ですか？
擬似文書: 申請手続きに必要な書類、本人確認資料、提出方法、申請条件を説明する。窓口やオンラインで提出する際の注意事項も示す。

質問: どこで開催されますか？
擬似文書: 催しの開催場所、会場名、所在地、交通アクセスを案内する。会場までの行き方や周辺施設についても記載する。

質問: 誰が運営していますか？
擬似文書: 事業を運営・実施する組織、担当部署、関係団体について説明する。各組織の役割と問い合わせ先も案内する。"""


def build_prompt(row: dict, mode: str) -> str:
    """Insert one query, and optionally its gold category, into the 4-shot prompt."""
    category = ""
    if mode == "category":
        label = CATEGORY_GUIDANCE[row["omission_category_id"]]
        category = (
            f"\nこの質問では「{label}」が省略されている可能性があります。"
            "その種類の情報を補う検索語を自然な文脈として含めてください。"
        )
    return f"""{FEW_SHOTS}

以下の質問について、検索対象の文書にありそうな語句を含む日本語の擬似文書を100〜200字で1つ作ってください。
事実を知っているふりをせず、固有名詞・数値・日付を推測で断定しないでください。
質問を言い換えるだけでなく、関連しそうな説明語や検索語を加えてください。
出力は擬似文書本文だけにしてください。{category}

質問: {row['omission_query']}
擬似文書:"""


def load_completed(path: Path) -> set[str]:
    """Return IDs already cached with a non-empty successful response."""
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("pseudo_document"):
                completed.add(row["eval_id"])
    return completed


def generate(client: OpenAI, row: dict, mode: str, model: str, max_tokens: int) -> tuple[str, str | None]:
    """Call the Responses API once and reject visibly incomplete short output."""
    response = client.responses.create(
        model=model,
        instructions=(
            "あなたは情報検索用のQuery2doc生成器です。与えられた質問から、"
            "検索対象にありそうな短い擬似文書だけを日本語で出力してください。"
        ),
        input=build_prompt(row, mode),
        max_output_tokens=max_tokens,
    )
    text = response.output_text.strip()
    if len(text) < 50:
        raise ValueError(f"Incomplete response ({len(text)} characters)")
    return text, getattr(response, "id", None)


def generate_record(
    client: OpenAI, row: dict, mode: str, model: str, max_tokens: int,
    max_retries: int, seed: int,
) -> dict:
    """Generate one cached record with bounded exponential-backoff retries."""
    rng = random.Random(f"{seed}:{mode}:{row['eval_id']}")
    error = None
    for attempt in range(max_retries):
        try:
            text, response_id = generate(client, row, mode, model, max_tokens)
            record = {
                "eval_id": row["eval_id"], "mode": mode, "model": model,
                "prompt_version": PROMPT_VERSION,
                "omission_query": row["omission_query"],
                "pseudo_document": text, "response_id": response_id, "status": "ok",
            }
            if mode == "category":
                record["omission_category_id"] = row["omission_category_id"]
                record["omission_category_name"] = row["omission_category_name"]
            return record
        except Exception as exc:  # API errors vary by SDK/version
            error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max_retries:
                time.sleep(min(30.0, (2 ** attempt) + rng.random()))
    raise RuntimeError(f"{mode} {row['eval_id']} failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Evaluation-query JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for resumable generation caches.")
    parser.add_argument("--mode", choices=("generic", "category", "both"), default="both", help="Generate without category information, with gold-category guidance, or both.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="OpenAI model alias or snapshot.")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Responses API output-token ceiling; distinct from the prompt's character target.")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum API attempts per query and mode.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent API requests.")
    parser.add_argument("--limit", type=int, help="Process only the first N rows for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for retry-delay jitter only; it does not seed model generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt for the first query without calling the API.")
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    modes = ("generic", "category") if args.mode == "both" else (args.mode,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        sample = dataset[0]
        for mode in modes:
            print(f"--- {mode} / {sample['eval_id']} ---\n{build_prompt(sample, mode)}\n")
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for mode in modes:
        output = args.output_dir / ("generic.jsonl" if mode == "generic" else "category_conditioned.jsonl")
        completed = load_completed(output)
        pending = [row for row in dataset if row["eval_id"] not in completed]
        if args.limit is not None:
            pending = pending[: args.limit]
        print(f"{mode}: {len(completed)} cached, {len(pending)} pending")
        with output.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        generate_record, client, row, mode, args.model,
                        args.max_output_tokens, args.max_retries, args.seed,
                    ): row
                    for row in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"[{index}/{len(pending)}] {record['eval_id']} ok", flush=True)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
