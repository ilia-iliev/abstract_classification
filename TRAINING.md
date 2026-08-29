# Training summary

`artifacts/model` is the PyTorch `google-bert/bert-base-uncased` benchmark baseline (`classifier.pt`).

## Configuration

- 512-token inputs with shared train/serve TeX-safe preprocessing; long/display math uses the learned `<FORMULA>` token
- 100,000 immutable-manifest training records; 20,000 validation records
- Deterministic 5%-overfetched candidate selection with normalized deduplication
- Training selection preserves at least 8,000 positives per label where available
- One epoch, Optuna-selected learning rate `1.964e-5`, AdamW weight decay `0.001`, warmup ratio `0.0474`, clip norm `1.0`
- Primary arXiv tags use target 1.0; secondary-only tags use the frozen soft target 0.5
- Per-label validation thresholds: `0.8, 0.7, 0.3, 0.5, 0.7`

## Result

| Split | Accuracy | Micro F1 | Macro F1 |
|---|---:|---:|---:|
| Validation | 0.9097 | 0.9467 | 0.7871 |

The 16-trial search used a fixed 5,000-record tuning subset, full validation scoring, and two RTX 3090 GPUs. The historical comparison on the same validation manifest scored macro F1 0.7834.

## Hyperparameter tuning

Full fine-tuning uses the immutable benchmark manifests, not the historical development split. Tune each backbone with 16 one-epoch trials on the fixed 5,000-record tuning subset:

```bash
uv run --extra training python -m scripts.tune_hyperparameters \
  data/arxiv-benchmark-snapshot.json data/benchmark-manifests \
  --output artifacts/hyperparameter-tuning
```

The common search space is learning rate (`8e-6`–`1e-4`, log scale), weight decay (`0`, `0.001`, `0.01`), and warmup ratio (`0`–`0.1`). Two CUDA devices run trials in parallel; selection is validation macro F1 after per-label threshold tuning. The command refuses to overwrite output and records all 16 trials, including failures.

## Frozen-representation probes

Run head-only probes on the immutable benchmark manifests before fine-tuning:

```bash
uv run --extra training python -m scripts.run_frozen_probes \
  data/arxiv-metadata-oai-snapshot.json data/benchmark-manifests \
  --output artifacts/frozen-probes
```

The command refuses to overwrite its output. Each model directory contains the frozen-backbone artifact, tokenizer, configuration, validation logits/probabilities and labels, metrics, and runtime/peak-VRAM data. `report.md` and `report.json` are deliberately separate from full fine-tuning results.
