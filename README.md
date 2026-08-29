# arXiv abstract classifier

A multi-label transformer classifier for five broad arXiv areas: biology, chemistry, computer science, physics, and social sciences. The repository includes training and evaluation tools plus a Django REST API.

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra training
uv run manage.py runserver
```

Classify an abstract:

```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H 'Content-Type: application/json' \
  -d '{"abstract":"We introduce a transformer algorithm for image classification."}'
```

Use `{"abstracts": ["...", "..."]}` to classify a batch of up to 32. `GET /api/health/` reports the active backend.

By default, the API loads `artifacts/model`. Set `MODEL_DIR` to use another artifact. If no model is available, it uses a keyword fallback so a fresh checkout remains runnable; the health endpoint identifies this clearly.

## Models and results

Four backbones were evaluated on the same untouched 20,000-record holdout split:

| Model | Macro F1 | Micro F1 | Artifact size |
|---|---:|---:|---:|
| Qwen3-Embedding-0.6B | **0.8087** | **0.9526** | 2.40 GB |
| embeddinggemma-300m | 0.7998 | 0.9506 | 1.25 GB |
| ModernBERT-base | 0.7895 | 0.9467 | 0.60 GB |
| bert-base-uncased | 0.7878 | 0.9457 | **0.44 GB** |

Qwen has the best overall quality. EmbeddingGemma offers a smaller, faster alternative with a modest quality difference, while BERT produces the smallest artifact. See [EVALUATION.md](EVALUATION.md) and [THROUGHPUT.md](THROUGHPUT.md) for the full metrics, confidence intervals, and hardware measurements.

Frozen models are available on Hugging Face:

- [Qwen3-Embedding-0.6B](https://huggingface.co/Ilia-Iliev/arxiv-abstract-classifier-qwen3-embedding-0.6b)
- [EmbeddingGemma 300M](https://huggingface.co/Ilia-Iliev/arxiv-abstract-classifier-embeddinggemma-300m)
- [ModernBERT base](https://huggingface.co/Ilia-Iliev/arxiv-abstract-classifier-modernbert-base)
- [BERT base uncased](https://huggingface.co/Ilia-Iliev/arxiv-abstract-classifier-bert-base-uncased)

## Data and training

Download the [Cornell arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) into `data/`:

```bash
kaggle datasets download Cornell-University/arxiv -p data --unzip
```

Train the default BERT model:

```bash
uv sync --extra training
uv run python -m scripts.train data/arxiv-metadata-oai-snapshot.json
```

Pass `--model-name` to choose another backbone and run `uv run python -m scripts.train --help` for all options. Training uses normalized, deduplicated abstracts, deterministic splits, and per-label thresholds selected on validation data.

Further documentation:

- [EDA.md](EDA.md) — dataset summary and label mapping
- [PREPROCESSING.md](PREPROCESSING.md) — text and formula normalization
- [TRAINING.md](TRAINING.md) — training setup and model details
- [EVALUATION.md](EVALUATION.md) — benchmark protocol and results
- [THROUGHPUT.md](THROUGHPUT.md) — performance measurement protocol

## Tests

```bash
uv run --extra dev pytest
```
