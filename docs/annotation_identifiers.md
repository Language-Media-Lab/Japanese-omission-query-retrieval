# Annotation identifiers

Annotator identifiers used in the released dataset and annotation records.

R1, R2, and R3 denote annotator slots within each annotation batch, not globally unique annotator IDs. The same label in different batches refers to different people. Identify a main-round annotator using the pair (batch ID, annotator label); do not join ratings across batches using R1, R2, or R3 alone.

Inter-annotator agreement reported in the paper uses the initial three ratings for each candidate. The two additional annotators who evaluated disagreement cases are separate from the twelve main-round annotators. Do not infer their identities from main-round slot labels.

## 日本語

R1・R2・R3は各バッチ内の評価者を区別するラベルであり，全バッチ共通の評価者IDではない。同じR1でもバッチが異なれば別人である。初回評価者は「バッチIDと評価者ラベル」の組で識別し，ラベルだけでバッチ間の評価を連結しない。

論文の評価者間一致度は，各候補に対する初回3名の評価を対象とする。不一致項目を追加評価した2名は初回評価の12名とは別であり，初回評価のラベルから追加評価者の同一性を推定しない。
