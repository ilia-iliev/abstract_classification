# Evaluation summary

These figures are retained from the experiment logs. The raw logs and predictions are not part of this repository, and the current code should not be treated as a bit-for-bit reproduction of the original runs.

All model choices, including hyperparameters and per-label thresholds, were made using the training and validation splits. The four resulting models were then evaluated once on a disjoint 20,000-record final holdout. No model was changed after seeing the holdout results.

## Overall results

| Model | Macro F1 | Micro F1 | Exact match | Top-1 |
|---|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B | **0.8087** | **0.9526** | **0.9174** | **0.9723** |
| embeddinggemma-300m | 0.7998 | 0.9506 | 0.9152 | 0.9684 |
| ModernBERT-base | 0.7895 | 0.9467 | 0.9091 | 0.9648 |
| bert-base-uncased | 0.7878 | 0.9457 | 0.9078 | 0.9624 |

Confidence intervals were calculated by resampling holdout examples. Each model was trained once, so those intervals measured holdout-sample uncertainty rather than variation between training runs.

## LLM comparison

A separate single-label prompt experiment ran Qwen3.8-27B in FP8 on the same 20,000-record holdout. A prediction counted as correct when it matched any expected broad label. It achieved 92.38% top-1 accuracy in about 1 hour 30 minutes, compared with 96.24% for fine-tuned BERT.

The LLM experiment did not produce multi-label probabilities, so macro and micro F1 were not compared.
