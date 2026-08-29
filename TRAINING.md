# Training summary

This is a record of the completed experiment, reconstructed from its logs. The code has since been revised, and the original intermediate artifacts are not included in the repository. The commands in the README describe the current training path; they are not a promise of bit-for-bit reproduction of these historical runs.

## Models

Four pretrained backbones were compared:

- `google-bert/bert-base-uncased` — 110M parameters (2018)
- `answerdotai/ModernBERT-base` — 149M parameters (2024)
- `google/embeddinggemma-300m` — 308M parameters (2025)
- `Qwen/Qwen3-Embedding-0.6B` — 600M parameters (2025)

Each model used a five-output linear head with independent sigmoid outputs. The backbone was first frozen to train a head-only probe, then unfrozen for end-to-end fine-tuning.

## Experiment procedure

- Build deterministic, disjoint training, validation, and holdout splits.
- Deduplicate normalized abstracts across the selected records.
- Keep validation and holdout at their natural label distribution.
- Select 100,000 training records while retaining at least 8,000 positives for each rare label where available.
- Run 16 one-epoch Optuna trials per backbone on a fixed 5,000-record tuning subset.
- Search learning rate, weight decay, and warmup ratio using validation macro F1.
- Tune per-label decision thresholds on validation data.
- Train each selected configuration once on 100,000 records.
- Evaluate each final model once on the 20,000-record holdout.

Primary categories received full loss weight. Secondary-only categories remained positive labels but received half loss weight.

## BERT run

The selected BERT configuration used 512-token inputs, one epoch, learning rate `1.964e-5`, AdamW weight decay `0.001`, warmup ratio `0.0474`, and gradient clipping at `1.0`.

| Split | Exact match | Micro F1 | Macro F1 |
|---|---:|---:|---:|
| Validation | 0.9097 | 0.9467 | 0.7871 |

The search ran on two RTX 3090 GPUs. The resulting BERT model was the smallest model in the final comparison and remained close to the larger backbones on the holdout.
