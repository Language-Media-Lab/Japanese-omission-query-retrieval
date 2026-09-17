# Query2docのLLM・プロンプト・生成条件

## 1. 確定した構成

Query2doc生成は`generate_query2doc.py`からOpenAI Responses APIを呼び出して行った。通常版とカテゴリ条件付き版で同じモデル・基本プロンプトを使い、カテゴリ条件の有無だけを変えた。

| 項目 | 設定 |
|---|---|
| API | OpenAI Responses API (`client.responses.create`) |
| Python SDK | `openai` 2.44.0（現在の`.venv`） |
| モデル | `gpt-5-mini` |
| 保存されたモデル表記 | 全552件で`gpt-5-mini` |
| prompt version | `query2doc-ja-v1` |
| 入力 | 省略後質問のみ。カテゴリ条件付き版はgold省略カテゴリの名称・定義も追加 |
| 出力形式 | 日本語の擬似文書本文1件 |
| API出力上限 | `max_output_tokens=256`（現行コードの既定値） |
| temperature | **指定していない** |
| reasoning effort | **指定していない** |
| tools / web search | 指定していない |
| seed（API生成） | 指定していない |
| 並列数 | 8 workers（現行コードの既定値） |
| API再試行 | 最大5回（現行コードの既定値） |
| 完成判定 | 前後空白除去後50文字以上 |

重要な区別として、モデル名とprompt versionは各キャッシュ行に保存されているため実行記録から確認できる。一方、`max_output_tokens`、再試行数、workersはキャッシュ行へ保存されていない。上表の値は現行コードの既定値であり、当時コマンドラインで上書きしなかったという既存作業記録に基づく。将来の再現性を高めるには、全リクエスト引数とSDKバージョンを出力メタデータへ保存すべきである。

## 2. モデルとAPI

実装は次の形でResponses APIを呼ぶ。

```python
response = client.responses.create(
    model=model,
    instructions=SYSTEM_INSTRUCTION,
    input=build_prompt(row, mode),
    max_output_tokens=max_tokens,
)
```

モデルの既定値は環境変数`OPENAI_MODEL`がなければ`gpt-5-mini`である。最終キャッシュ552件すべての`model`フィールドが`gpt-5-mini`なので、実際の生成モデル名は確認できる。

ただし、保存されているのはエイリアス`gpt-5-mini`で、固定snapshot名ではない。OpenAI公式モデルページではResponses API対応とsnapshotが案内されているが、キャッシュから当時エイリアスが指していた厳密なsnapshotまでは復元できない。再実験では利用可能な固定snapshotを明示し、当時結果とは別runとして扱うのが安全である。

公式情報: [GPT-5 mini Model | OpenAI API](https://developers.openai.com/api/docs/models/gpt-5-mini)

## 3. instructions

通常版・カテゴリ条件付き版に共通する`instructions`は次のとおりである。

```text
あなたは情報検索用のQuery2doc生成器です。与えられた質問から、
検索対象にありそうな短い擬似文書だけを日本語で出力してください。
```

ここでは役割、目的、言語、出力形式だけを指定する。

## 4. 共通few-shot

両方式に同じ4例を与える。

```text
次の例のように、質問への回答を含みそうな短い文書を作ってください。

質問: 利用できる時間帯はいつですか？
擬似文書: サービスを利用できる時間帯や受付時間、曜日、休業日について案内する。利用可能な開始時刻と終了時刻、例外となる日程も確認できる。

質問: 申請に必要なものは何ですか？
擬似文書: 申請手続きに必要な書類、本人確認資料、提出方法、申請条件を説明する。窓口やオンラインで提出する際の注意事項も示す。

質問: どこで開催されますか？
擬似文書: 催しの開催場所、会場名、所在地、交通アクセスを案内する。会場までの行き方や周辺施設についても記載する。

質問: 誰が運営していますか？
擬似文書: 事業を運営・実施する組織、担当部署、関係団体について説明する。各組織の役割と問い合わせ先も案内する。
```

例は時間、必要物、場所、主体を含むが、7カテゴリそれぞれに一例ずつ対応する設計ではない。

## 5. 通常Query2docの入力プロンプト

few-shotの後に次を追加する。

```text
以下の質問について、検索対象の文書にありそうな語句を含む日本語の擬似文書を100〜200字で1つ作ってください。
事実を知っているふりをせず、固有名詞・数値・日付を推測で断定しないでください。
質問を言い換えるだけでなく、関連しそうな説明語や検索語を加えてください。
出力は擬似文書本文だけにしてください。

質問: {omission_query}
擬似文書:
```

モデルへ渡す質問は`omission_query`だけである。次は渡していない。

- 元質問`original_question`
- 実際に省略した文字列`omitted_information`
- 正解回答
- 正解チャンク本文またはID
- PDF名・ページ
- 省略カテゴリ

## 6. Category-conditioned Query2doc

カテゴリ条件付き版は共通プロンプトへ次の2文だけを追加する。

```text
この質問では「{カテゴリ名称と定義}」が省略されている可能性があります。
その種類の情報を補う検索語を自然な文脈として含めてください。
```

挿入するカテゴリ定義は次のとおりである。

| ID | プロンプト上の説明 |
|---|---|
| `person_org` | 人物・組織情報（行為者、実施主体、運営者、所有者、情報源など） |
| `location` | 場所・管轄情報（地域、所在地、会場、適用地域など） |
| `time` | 時間情報（年度、日付、期間、時刻、期限など） |
| `what` | 対象・内容・属性情報（制度、商品、条件、金額、数量など） |
| `ga_case` | ガ格の項（動作・状態の主体など） |
| `wo_case` | ヲ格の項（動作の対象など） |
| `ni_case` | ニ格の項（相手、到達点、対象など） |

これは正解ラベルである省略カテゴリを使うOracle的な入力条件だが、実際に省略された固有名詞や値は渡さない。したがって「何の種類が欠けているか」は分かっても、「具体的に何が欠けているか」は分からない。

## 7. temperatureと再現性

`temperature`はAPI呼び出しに含まれていない。したがって、論文や説明資料で`temperature=0`などと記載してはいけない。正確な表現は「temperatureは明示指定せず、API・モデル既定値を使用」である。

同様にAPI側の乱数seedは指定していない。コードの`--seed 42`は生成内容を固定するseedではなく、API失敗後の待機時間へ加えるjitterを決定するためだけに使う。よって同じプロンプトを再送しても同一文面になる保証はない。

## 8. 出力長

プロンプトは100〜200**字**を要求するが、APIの`max_output_tokens=256`は最大出力**トークン**数であり、256文字ではない。両者は別の制約である。

最終キャッシュの文字数は次のとおりだった。文字数はPython/JQのUnicode文字列長に相当する集計で、トークン数ではない。

| mode | 件数 | 最小 | 中央値 | 平均 | 最大 | 100字未満 | 200字超 |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic | 276 | 66 | 124 | 124.7 | 181 | 14 | 0 |
| category | 276 | 50 | 133 | 134.5 | 213 | 5 | 1 |

100〜200字はプロンプトによるsoft constraintであり、コードはこの範囲を強制していない。機械的に拒否するのは50字未満だけなので、50〜99字や201字以上も成功として保存され得る。

## 9. 再試行と不完全応答の修復

`generate_record`は1クエリ・1 modeについて最大5回試す。例外が起きると、指数バックオフと決定的jitterを組み合わせて待機する。

```text
待機秒 = min(30, 2^attempt + jitter)
```

対象となる例外にはAPIエラーだけでなく、出力が50字未満の`ValueError`も含まれる。5回すべて失敗するとrun全体へ`RuntimeError`を返す。

処理済み判定は、既存JSONL行が`status == "ok"`かつ`pseudo_document`非空であることだけを見る。通常の再開では50字以上かを再検査しない。このため初回runで`ok`として保存された短い4件は、後からバックアップを作って再生成した。

| eval_id | 修復前文字数 | 修復前末尾 |
|---|---:|---|
| EVAL-0060 | 27 | `貸し付け`で途切れた |
| EVAL-0105 | 19 | `直売所整備、`で途切れた |
| EVAL-0155 | 22 | `老舗、屋`で途切れた |
| EVAL-0174 | 4 | `冷凍空調`だけだった |

修復前ファイルは`generic.pre_incomplete_repair.jsonl`として保存されている。最終genericは4件を再生成済みで、最小66字である。

## 10. キャッシュの監査結果

| 検査 | generic | category |
|---|---:|---:|
| 行数 | 276 | 276 |
| unique `eval_id` | 276 | 276 |
| `status=ok` | 276 | 276 |
| model=`gpt-5-mini` | 276 | 276 |
| prompt version=`query2doc-ja-v1` | 276 | 276 |
| response IDあり | 276 | 276 |
| 50字未満 | 0 | 0 |
| category情報あり | 対象外 | 276 |

API応答本文、response ID、質問、mode、モデル名、prompt versionは保存される。使用量、生成トークン数、finish reason、temperature、全リクエスト引数、SDKバージョン、実行日時は保存されていない。

## 11. 検索時の結合条件

生成後は`run_query2doc_retrieval.py`で、次のように元の省略質問と疑似文書を1回ずつ空白で連結し、文字bigram BM25へ入力した。

```text
expanded_query = omission_query + " [SEP] " + pseudo_document
```

元クエリを5回反復するQuery2doc原論文のSparse条件は採用せず、同論文のDense条件に近い`query [SEP] pseudo-document`形式を文字bigram BM25へ適用した。この違いは性能不振の原因候補であり、生成条件と検索時結合条件を分けて説明する必要がある。

## 12. 論文へ記載する推奨文

> Query2docの擬似文書生成にはOpenAI Responses APIの`gpt-5-mini`を用いた。4件のfew-shot例を含む日本語プロンプトに省略後質問を入力し、100〜200字の擬似文書本文を要求した。`max_output_tokens`は256とし、temperatureおよびreasoning effortは明示指定しなかった。カテゴリ条件付き版では、正解省略カテゴリの名称と定義のみを追加し、元質問、正解回答、正解文書、実際の省略文字列は与えなかった。50字未満の不完全出力は再生成した。

## 13. 関連ファイル

- 生成実装: `generate_query2doc.py`
- 通常版キャッシュ: `omission_query_dataset/query2doc/generic.jsonl`
- カテゴリ条件付きキャッシュ: `omission_query_dataset/query2doc/category_conditioned.jsonl`
- 修復前バックアップ: `omission_query_dataset/query2doc/generic.pre_incomplete_repair.jsonl`
- 検索実装: `run_query2doc_retrieval.py`
- 先行研究との実装差: `docs/query_expansion_prior_work_audit.md`

## Revised retrieval condition (2026-09-17)

The current paper uses separate bigram bags: `5 * count(query) + count(pseudo_document)`. `[SEP]` and repetition-boundary bigrams are not included. Weight 1 is a supplemental comparison. BM25 parameters are selected using document-grouped development folds in the main comparison. See [protocol](review_20260911/experiment_protocol.md), [exact prompts](prompts/), and [quality notes](quality_notes.md).
