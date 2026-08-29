# Frozen benchmark preprocessing and targets

Every benchmark model receives the normalized abstract only: no title, category, or other metadata. The preprocessing version is `inline_math_24_formula_token_v2` and the maximum input is 512 tokenizer tokens.

Long inline and display-math expressions are replaced with the special token `<FORMULA>`.

Short inline expressions are still flattened into readable text. For example, `$H_2$O` becomes `H2O`. This preserves useful common notation while avoiding long expressions whose rare symbols and WordPiece fragments consume context without helping broad-subject classification.

`<FORMULA>` is registered as an additional special token for every tokenizer before token-count reporting and training. Full fine-tuning resizes the backbone embeddings, so that embedding is trainable. Saved artifacts must contain that tokenizer.

Labels are independent. A mapped primary category has target `1.0`; a mapped secondary-only category has target `0.5`. These values are fixed experiment configuration, not tuning parameters.

Run token accounting for all benchmark tokenizers before training:

```bash
uv run python -m scripts.tokenizer_eda data/arxiv-benchmark-snapshot.json > logs/benchmark-tokenizers.json
```

The report records each tokenizer's token-count distribution and 512-token truncation rate.
