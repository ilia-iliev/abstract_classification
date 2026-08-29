# arXiv abstract classifier

A multi-label transformer classifier for five broad arXiv areas: biology, chemistry, computer science, physics, and social sciences. The repository includes training and evaluation tools plus a Django REST API.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From a fresh clone:

```bash
./scripts/setup.sh
uv run manage.py runserver
```

Setup creates the Python environment and downloads the best-performing model from Hugging Face into `artifacts/model`.

Check that the model loaded:

```bash
curl http://127.0.0.1:8000/api/health/
# {"status":"ok","classifier_backend":"pytorch"}
```

Classify an abstract:

```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H 'Content-Type: application/json' \
  -d '{"abstract":"We introduce a transformer algorithm for image classification."}'
```

Use `{"abstracts": ["...", "..."]}` to classify a batch of up to 32. The first request loads the model and can take longer. CPU is the default; set `MODEL_DEVICE=cuda` for a CUDA GPU.

## Models and results

Four fully fine-tuned models were evaluated once on the untouched 20,000-record final holdout:

| Model | Macro F1 | Micro F1 | Exact match | Top-1 | Artifact size |
|---|---:|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B | **0.8087** | **0.9526** | **0.9174** | **0.9723** | 2.40 GB |
| embeddinggemma-300m | 0.7998 | 0.9506 | 0.9152 | 0.9684 | 1.25 GB |
| ModernBERT-base | 0.7895 | 0.9467 | 0.9091 | 0.9648 | 0.60 GB |
| bert-base-uncased | 0.7878 | 0.9457 | 0.9078 | 0.9624 | **0.44 GB** |

The four fine-tuned models are available on Hugging Face:

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

## Tests

```bash
uv run --extra dev pytest
```
