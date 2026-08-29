import json
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np

from classifier.preprocessing import MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION, prepare_abstract

logger = logging.getLogger(__name__)

LABELS = ["biology", "chemistry", "computer_science", "physics", "social_sciences"]
_FALLBACK_WORD_PATTERN = re.compile(r"[a-z]+")
_KEYWORDS = {
    "biology": {"cell", "gene", "protein", "organism", "genome", "neural", "biological", "species", "disease"},
    "chemistry": {"chemical", "molecule", "molecular", "reaction", "catalyst", "polymer", "synthesis", "compound"},
    "computer_science": {"algorithm", "computer", "software", "network", "database", "learning", "model", "program"},
    "physics": {"quantum", "particle", "energy", "field", "gravity", "matter", "optical", "relativity", "physics"},
    "social_sciences": {"economic", "financial", "market", "social", "policy", "society", "human", "behavior"},
}


class AbstractClassifier:
    def __init__(self):
        self.artifact_dir = Path(os.getenv("MODEL_DIR", "artifacts/model"))
        self._model = None
        self._tokenizer = None
        self._labels = LABELS
        self._threshold = 0.5
        self._device = "cpu"
        self._load_attempted = False
        self._lock = threading.Lock()

    @property
    def backend(self):
        self._load_model()
        return "pytorch" if self._model is not None else "keyword_fallback"

    def _load_model(self):
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            if not (self.artifact_dir / "config.json").exists():
                return
            metadata_path = self.artifact_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning("Model artifact has no preprocessing metadata: %s", self.artifact_dir)
                return
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("preprocessing") != PREPROCESSING_VERSION:
                logger.warning(
                    "Model artifact uses incompatible preprocessing; retrain it with %s",
                    PREPROCESSING_VERSION,
                )
                return

            if metadata.get("backend") != "pytorch" or not (self.artifact_dir / "classifier.pt").exists():
                logger.warning("Model artifact is not a PyTorch classifier: %s", self.artifact_dir)
                return

            from transformers import AutoTokenizer
            from classifier.modeling import MultilabelClassifier

            labels, threshold = metadata.get("labels"), metadata.get("threshold")
            if labels != LABELS or not isinstance(threshold, list) or len(threshold) != len(labels):
                logger.warning("Model artifact has incompatible labels or thresholds: %s", self.artifact_dir)
                return
            requested_device = os.getenv("MODEL_DEVICE", "cpu")
            import torch

            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA was requested but is unavailable; using CPU")
                requested_device = "cpu"
            self._labels = labels
            self._tokenizer = AutoTokenizer.from_pretrained(self.artifact_dir)
            self._model = MultilabelClassifier.load(self.artifact_dir, len(self._labels)).to(requested_device)
            self._model.eval()
            self._device = requested_device
            self._threshold = np.asarray(threshold, dtype=float)

    def predict(self, abstracts):
        self._load_model()
        if self._model is None:
            return [self._fallback(text) for text in abstracts]
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
            logits = self._model(**{key: value.to(self._device) for key, value in encoded.items()}).cpu().numpy()
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
            "scores": {label: round(float(score), 6) for label, score in zip(self._labels, scores)},
            "backend": backend,
        }

    def _fallback(self, text):
        words = set(_FALLBACK_WORD_PATTERN.findall(text.lower()))
        raw_scores = np.array([
            sum(any(word.startswith(keyword[:5]) for word in words) for keyword in _KEYWORDS[label])
            for label in self._labels
        ], dtype=float)
        scores = (raw_scores + 0.1) / (raw_scores.sum() + 0.1 * len(self._labels))
        return self._format_scores(scores, "keyword_fallback")


classifier = AbstractClassifier()
