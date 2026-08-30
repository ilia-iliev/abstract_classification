# arXiv abstract classifier

Text classifier for abstracts. The repository includes training and evaluation code and a Django REST API. The [experiment write-up](https://ilia.foo/blog/old_school_text_classification) explains the analysis and model comparison.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From a fresh clone:

```bash
uv run python -m scripts.download_model
uv run manage.py runserver
```

Open <http://127.0.0.1:8000/> for the browser interface. Paste an abstract and the page shows its predicted categories and confidence scores.

`uv run` creates the Python environment and the setup command downloads a model from Hugging Face into `artifacts/model`.

```bash
uv run python -m scripts.download_model --model bert-base
```

The available choices are `bert-base`, `modernbert`, `embedding-gemma`, and `qwen` (the default).

Check that the model loaded:

```bash
curl http://127.0.0.1:8000/api/health/
# {"status":"ok","classifier_backend":"pytorch"}
```

The API fails closed when the model is missing or incompatible: health and classification return HTTP 503 rather than serving substitute predictions.

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

These are historical measurements retained from the experiment logs. The code evolved after the runs, and raw logs, predictions, and intermediate artifacts are not included. The published model weights are downloaded from Hugging Face by the setup command above.

The checked-in experiment notes cover [EDA](EDA.md), [preprocessing](PREPROCESSING.md), [training](TRAINING.md), [evaluation](EVALUATION.md), and [throughput methodology](THROUGHPUT.md).

## Data and training

```bash
kaggle datasets download Cornell-University/arxiv -p data --unzip
```

Train the default BERT model:

```bash
uv sync --extra training
uv run python -m scripts.train data/arxiv-metadata-oai-snapshot.json
```

Pass `--model-name` to choose backbone and run `uv run python -m scripts.train --help` for all options.

