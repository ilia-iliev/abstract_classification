"""Download the production classifier artifact from Hugging Face."""

import json
import os
from pathlib import Path

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


configure_huggingface_home()

from huggingface_hub import snapshot_download

MODEL_REPO = "Ilia-Iliev/arxiv-abstract-classifier-qwen3-embedding-0.6b"
MODEL_REVISION = "7470b26157314257070f4c4f056376befa96d280"
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
]


def model_dir():
    default = Path(__file__).resolve().parents[1] / "artifacts" / "model"
    return Path(os.getenv("MODEL_DIR", default)).expanduser().resolve()


def is_complete(destination):
    if not all((destination / name).is_file() for name in REQUIRED_FILES):
        return False
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    return metadata.get("base_model") == "Qwen/Qwen3-Embedding-0.6B"


def main():
    destination = model_dir()
    if is_complete(destination):
        print(f"Best model is already installed at {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPO} to {destination} (about 2.4 GB)...")
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    if not is_complete(destination):
        raise RuntimeError(f"Downloaded model is incomplete: {destination}")
    print(f"Model installed at {destination}")


if __name__ == "__main__":
    main()
