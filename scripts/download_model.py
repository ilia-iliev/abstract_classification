"""Download a classifier artifact from Hugging Face."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def cache_is_writable(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return False
    return os.access(path, os.W_OK)


def configure_huggingface_home():
    """Use a writable cache even when HF_HOME points to a shared read-only path."""
    configured = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    if cache_is_writable(configured):
        return configured

    configured = Path.home() / ".cache" / "huggingface"
    configured.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(configured)
    print(f"HF_HOME is not writable; using {configured} for the download cache.")
    return configured


MODELS = {
    "bert-base": {
        "base_model": "google-bert/bert-base-uncased",
        "repo": "Ilia-Iliev/arxiv-abstract-classifier-bert-base-uncased",
        "revision": "c63651ad87980b04cd6cf99807e96f2626b3dfaa",
        "size": "440 MB",
    },
    "modernbert": {
        "base_model": "answerdotai/ModernBERT-base",
        "repo": "Ilia-Iliev/arxiv-abstract-classifier-modernbert-base",
        "revision": "9428d2bff5fa0744ce370d1b1b7e4ef0dd65ebaa",
        "size": "600 MB",
    },
    "embedding-gemma": {
        "base_model": "google/embeddinggemma-300m",
        "repo": "Ilia-Iliev/arxiv-abstract-classifier-embeddinggemma-300m",
        "revision": "1d970d3b6a4e2a171e9d9e18a1cf5c63daf05305",
        "size": "1.25 GB",
    },
    "qwen": {
        "base_model": "Qwen/Qwen3-Embedding-0.6B",
        "repo": "Ilia-Iliev/arxiv-abstract-classifier-qwen3-embedding-0.6b",
        "revision": "7470b26157314257070f4c4f056376befa96d280",
        "size": "2.4 GB",
    },
}
REQUIRED_FILES = {
    "classifier.pt",
    "config.json",
    "metadata.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
DOWNLOAD_PATTERNS = [
    "added_tokens.json",
    "chat_template.jinja",
    "classifier.pt",
    "config.json",
    "merges.txt",
    "metadata.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
]


def model_dir():
    default = Path(__file__).resolve().parents[1] / "artifacts" / "model"
    return Path(os.getenv("MODEL_DIR", default)).expanduser().resolve()


def is_complete(destination, base_model):
    if not all((destination / name).is_file() for name in REQUIRED_FILES):
        return False
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    return metadata.get("base_model") == base_model


def parse_args():
    parser = argparse.ArgumentParser(description="Download a classifier model from Hugging Face.")
    parser.add_argument("--model", choices=MODELS, default="qwen")
    return parser.parse_args()


def main():
    args = parse_args()
    model = MODELS[args.model]
    destination = model_dir()
    if is_complete(destination, model["base_model"]):
        print(f"{args.model} is already installed at {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    cache = configure_huggingface_home()
    print(f"Downloading {model['repo']} to {destination} (about {model['size']})...")
    snapshot_download(
        repo_id=model["repo"],
        revision=model["revision"],
        local_dir=destination,
        cache_dir=cache,
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    if not is_complete(destination, model["base_model"]):
        raise RuntimeError(f"Downloaded model is incomplete: {destination}")
    print(f"{args.model} installed at {destination}")


if __name__ == "__main__":
    main()
