# Final-holdout evaluation

All model choices, including hyperparameters and per-label thresholds, were selected using the training and validation splits. The four resulting models were then frozen and evaluated once on a disjoint 20,000-record final holdout. No model was retrained, retuned, or otherwise changed after this evaluation.

## Overall results

| Model | Macro F1 | Micro F1 | Exact match | Top-1 |
|---|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B | **0.8087** | **0.9526** | **0.9174** | **0.9723** |
| embeddinggemma-300m | 0.7998 | 0.9506 | 0.9152 | 0.9684 |
| ModernBERT-base | 0.7895 | 0.9467 | 0.9091 | 0.9648 |
| bert-base-uncased | 0.7878 | 0.9457 | 0.9078 | 0.9624 |

Confidence intervals resample holdout examples only; the models were each trained once, so they do not represent variation between training runs. Full metrics, predictions, and confidence intervals are in `artifacts/final-holdout-evaluation/report.json`.

## LLM comparison

A separate single-label prompt experiment ran Qwen3.8-27B in FP8 on the same 20,000-record holdout. A prediction counted as correct when it matched any expected broad label. It achieved 92.38% top-1 accuracy in about 1 hour 30 minutes, compared with 96.24% for fine-tuned BERT. The LLM experiment did not produce multi-label probabilities, so macro and micro F1 are not reported for it. Results are in `logs/qwen-llm-final-holdout-summary.json`.
