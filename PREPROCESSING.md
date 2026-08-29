# Preprocessing and label weighting

Every model receives the normalized abstract only: no title, category, or other metadata. Conventional lowercasing, lemmatization, and stopword removal regressed performance in an initial 10,000-record experiment, so they are not applied. The preprocessing version is `inline_math_24_formula_token_v2` and the maximum input is 512 tokenizer tokens.

Long inline and display-math expressions are replaced with the special token `<FORMULA>`.

Short inline expressions are still flattened into readable text. For example, `$H_2$O` becomes `H2O`. This preserves useful common notation while avoiding long expressions whose rare symbols and WordPiece fragments consume context without helping broad-subject classification.

`<FORMULA>` is registered as an additional special token for every tokenizer before token-count reporting and training. Full fine-tuning resizes the backbone embeddings, so that embedding is trainable. Saved artifacts contain the matching tokenizer.

On a seeded 10,000-record BERT sample, normalization reduced the median length from 217 to 207 tokens, the 95th percentile from 385 to 357, and the share over 512 tokens from 0.49% to 0.03%. The normalized sample had no `[UNK]` tokens.

Labels are independent binary positives. Primary categories receive full loss weight (`1.0`); secondary-only categories remain positive labels but receive half loss weight (`0.5`). The half-weighted loss still pushes a secondary prediction toward `1.0`; it does not treat `0.5` as the desired probability. These weights are fixed experiment configuration, not tuning parameters.

Before training, token counts and 512-token truncation rates were measured separately for each tokenizer.
