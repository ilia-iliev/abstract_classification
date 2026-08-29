# Frozen preprocessing and targets

Every model receives the normalized abstract only: no title, category, or other metadata. Conventional lowercasing, lemmatization, and stopword removal regressed performance in an initial 10,000-record experiment, so they are not applied. The preprocessing version is `inline_math_24_formula_token_v2` and the maximum input is 512 tokenizer tokens.

Long inline and display-math expressions are replaced with the special token `<FORMULA>`.

Short inline expressions are still flattened into readable text. For example, `$H_2$O` becomes `H2O`. This preserves useful common notation while avoiding long expressions whose rare symbols and WordPiece fragments consume context without helping broad-subject classification.

`<FORMULA>` is registered as an additional special token for every tokenizer before token-count reporting and training. Full fine-tuning resizes the backbone embeddings, so that embedding is trainable. Saved artifacts contain the matching tokenizer.

On a seeded 10,000-record BERT sample, normalization reduced the median length from 217 to 207 tokens, the 95th percentile from 385 to 357, and the share over 512 tokens from 0.49% to 0.03%. The normalized sample had no `[UNK]` tokens.

Labels are independent. A mapped primary category has target `1.0`; a mapped secondary-only category has target `0.5`. These values are fixed experiment configuration, not tuning parameters.

Run token accounting for all tokenizers before training:

```bash
uv run python -m scripts.tokenizer_eda data/arxiv-snapshot.json > logs/tokenizers.json
```

The report records each tokenizer's token-count distribution and 512-token truncation rate.
