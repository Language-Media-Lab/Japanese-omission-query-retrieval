# Released dataset

`annotation_filtered_276.jsonl` contains one finalized omission query per line. It contains no worker names, email addresses, API keys, or private workbook metadata.

Important fields:

- `eval_id`: stable public evaluation identifier;
- `original_question`: question before omission generation;
- `omission_query`: question after deleting information;
- `omitted_information`: text removed from the original question;
- `omission_category_id` and `omission_category_name`: one of seven categories;
- `gold_chunk_id`: identifier of the corresponding evidence document;
- `ratings`: anonymized naturalness, answerability, and applicable category-validity ratings.

The retrieval corpus is intentionally not redistributed. Follow the root README to construct it from JDocQA.

License: CC BY-SA 4.0. See `../DATA_LICENSE.md`.
