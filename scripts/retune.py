import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from classifier.modeling import MultilabelClassifier
from classifier.preprocessing import PREPROCESSING_VERSION
from scripts.data import load_category_examples, load_uniform_examples
from scripts.train import arrays, device_for, encoded_dataset, metrics, probabilities, tune_thresholds


def retune(args):
    model_dir = Path(args.model)
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("backend") != "pytorch" or not (model_dir / "classifier.pt").exists():
        raise ValueError("Model artifact is not a PyTorch classifier")
    if metadata.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError(f"Model artifact must use {PREPROCESSING_VERSION} preprocessing")
    split = load_uniform_examples(args.dataset, metadata["per_label_target"], metadata["validation_records"], metadata["test_records"]) if metadata.get("sampling") == "uniform" else load_category_examples(args.dataset, metadata["sample_size"], metadata["validation_records"], metadata["test_records"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = device_for(args)
    model = MultilabelClassifier.load(model_dir, len(metadata["labels"])).to(device)
    validation_texts, y_validation = arrays(split["validation"])
    test_texts, y_test = arrays(split["test"])
    validation_probabilities = probabilities(model, encoded_dataset(tokenizer, validation_texts, y_validation, args.batch_size), device)
    candidates = np.arange(.05, 1, .05).round(2).tolist()
    thresholds = tune_thresholds(y_validation, validation_probabilities, candidates)
    metadata.update({"training_threshold": metadata["threshold"], "pre_retune_validation_metrics": metadata["validation_metrics"], "pre_retune_test_metrics": metadata["test_metrics"], "threshold": thresholds, "threshold_candidates": candidates, "validation_metrics": metrics(y_validation, validation_probabilities, np.asarray(thresholds)), "test_metrics": metrics(y_test, probabilities(model, encoded_dataset(tokenizer, test_texts, y_test, args.batch_size), device), np.asarray(thresholds))})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "validation_macro_f1": metadata["validation_metrics"]["f1_macro"], "test_macro_f1": metadata["test_metrics"]["f1_macro"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--model", default="artifacts/model"); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--device"); retune(parser.parse_args())


if __name__ == "__main__": main()
