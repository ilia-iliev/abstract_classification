import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

from classifier.labels import LABELS
from classifier.preprocessing import MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION, prepare_abstract

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class AbstractClassifier:
    def __init__(self):
        default_artifact_dir = Path(__file__).resolve().parents[1] / "artifacts" / "model"
        self.artifact_dir = Path(os.getenv("MODEL_DIR", default_artifact_dir)).expanduser().resolve()
        self._model = None
        self._tokenizer = None
        self._labels = LABELS
        self._threshold = 0.5
        self._device = "cpu"
        self._load_attempted = False
        self._load_error = None
        self._lock = threading.Lock()

    @property
    def backend(self):
        self._load_model()
        return "pytorch" if self._model is not None else None

    @property
    def load_error(self):
        self._load_model()
        return self._load_error

    def _reject_artifact(self, reason):
        self._load_error = reason
        logger.error(reason)

    def _load_model(self):
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            if not (self.artifact_dir / "config.json").exists():
                self._reject_artifact(f"Model artifact is missing from {self.artifact_dir}")
                return
            metadata_path = self.artifact_dir / "metadata.json"
            if not metadata_path.exists():
                self._reject_artifact(
                    f"Model artifact has no preprocessing metadata: {self.artifact_dir}"
                )
                return
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                self._reject_artifact(f"Cannot read model metadata at {metadata_path}: {error}")
                return
            if metadata.get("preprocessing") != PREPROCESSING_VERSION:
                self._reject_artifact(
                    f"Model artifact uses incompatible preprocessing; expected {PREPROCESSING_VERSION}"
                )
                return
            if metadata.get("backend") != "pytorch" or not (
                self.artifact_dir / "classifier.pt"
            ).exists():
                self._reject_artifact(
                    f"Model artifact is not a PyTorch classifier: {self.artifact_dir}"
                )
                return

            from classifier.modeling import MultilabelClassifier, load_tokenizer

            labels, threshold = metadata.get("labels"), metadata.get("threshold")
            if labels != LABELS or not isinstance(threshold, list) or len(threshold) != len(labels):
                self._reject_artifact(
                    f"Model artifact has incompatible labels or thresholds: {self.artifact_dir}"
                )
                return
            requested_device = os.getenv("MODEL_DEVICE", "cpu")
            import torch

            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA was requested but is unavailable; using CPU")
                requested_device = "cpu"
            try:
                tokenizer = load_tokenizer(self.artifact_dir)
                model = MultilabelClassifier.load(self.artifact_dir, len(labels)).to(
                    requested_device
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._load_error = f"Cannot load model artifact at {self.artifact_dir}: {error}"
                logger.exception(self._load_error)
                return
            model.eval()
            self._labels = labels
            self._tokenizer = tokenizer
            self._model = model
            self._device = requested_device
            self._threshold = np.asarray(threshold, dtype=float)

    def predict(self, abstracts):
        self._load_model()
        if self._model is None:
            raise ModelUnavailableError(self._load_error or "Model is unavailable")
        texts = [prepare_abstract(text) for text in abstracts]
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
            return_tensors="pt",
        )
        import torch

        with torch.inference_mode():
            logits = self._model(
                **{key: value.to(self._device) for key, value in encoded.items()}
            ).cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))
        return [self._format_scores(row, "pytorch") for row in scores]

    def _format_scores(self, scores, backend):
        thresholds = np.broadcast_to(self._threshold, len(self._labels))
        selected = [
            label
            for label, score, threshold in zip(self._labels, scores, thresholds)
            if score >= threshold
        ]
        if not selected:
            selected = [self._labels[int(np.argmax(scores))]]
        return {
            "categories": selected,
            "scores": {
                label: round(float(score), 6) for label, score in zip(self._labels, scores)
            },
            "backend": backend,
        }


classifier = AbstractClassifier()
