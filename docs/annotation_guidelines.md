# Annotation protocol

This is a summary of the procedure actually used for the released dataset, replacing the earlier draft guidelines.

306 candidates (120 case omissions and 186 semantic-role omissions) were divided into four batches. Twelve main annotators each evaluated 76–77 candidates, providing three independent ratings per candidate. Two different annotators additionally rated 76 candidates with disagreement between scores 1 and 3. Annotators saw the original question, omission query, recorded omitted information, and gold answer; semantic-role items also displayed the category.

| Axis | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Naturalness | Natural | Somewhat unnatural | Unnatural |
| Answerability | Answerable | Conditionally answerable | Difficult to answer |
| Category validity (semantic-role items only) | Appropriate | Ambiguous classification | Inappropriate |

Answerability assumes the appropriate evidence document has already been identified and is available. Score 2 means that the gold answer is a candidate but a heading, surrounding context, or target clarification is needed. This is not a judgment of retrieval success or unique intent from the query alone.

The median of three ratings (five for additional-review items) was used. Thirty items with median naturalness or answerability 3 were excluded, leaving 276. Category validity was not an exclusion criterion. Two retained items have median category validity 3 (EVAL-0053 and EVAL-0256).

Case categories are ga, wo, and ni. Semantic-role categories are person/organization, location/jurisdiction, time, and other target/content/attribute, determined by the role in the original question. No fifth semantic category was used.

See [annotator identifiers](annotation_identifiers.md) for batch-local R1/R2/R3 labels. Agreement in the paper is computed from initial three-person ratings using ordinal Krippendorff alpha. Supplemental machine-assisted quality auditing does not constitute additional human annotation.
