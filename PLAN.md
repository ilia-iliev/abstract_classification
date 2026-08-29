# Experiment design

This records the design used for the completed experiment. The implementation has since evolved and is not intended as an exact replay of the original runs.

Four pretrained backbones were trained and compared for five-label arXiv abstract classification.

| Split | Records | Use |
|---|---:|---|
| Training | 100,000 | Model fitting |
| Validation | 20,000 | Hyperparameter, checkpoint, and threshold selection |
| Final holdout | 20,000 | One-time evaluation of the fully fine-tuned models |

The splits were deterministic, disjoint, and deduplicated after text normalization. Training was tag-aware; validation and holdout retained the natural label distribution.

Each model received 16 Optuna trials on a fixed 5,000-record subset of the training split. Hyperparameters and per-label thresholds were selected by validation macro F1, then the selected configuration was trained once on 100,000 records.

Before evaluation, the checkpoints, preprocessing, thresholds, manifests, metric code, and throughput protocol were frozen. Every fully fine-tuned model was evaluated once on the final holdout without subsequent retraining, retuning, or threshold changes.
