import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from classifier.modeling import MultilabelClassifier
from classifier.preprocessing import PREPROCESSING_VERSION
from scripts.data import load_category_examples, load_uniform_examples
from scripts.train import arrays, device_for, encoded_dataset, metrics, probabilities


def evaluate(args):
    model_dir = Path(args.model)
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("backend") != "pytorch" or not (model_dir / "classifier.pt").exists():
        raise ValueError("Model artifact is not a PyTorch classifier")
    if metadata.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError(f"Model artifact must use {PREPROCESSING_VERSION} preprocessing")
    split = load_uniform_examples(args.dataset, metadata["per_label_target"], metadata["validation_records"], metadata["test_records"]) if metadata.get("sampling") == "uniform" else load_category_examples(args.dataset, metadata["sample_size"], metadata["validation_records"], metadata["test_records"])
    validation_texts, y_validation = arrays(split["validation"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = device_for(args)
    model = MultilabelClassifier.load(model_dir, len(metadata["labels"])).to(device)
    result = metrics(y_validation, probabilities(model, encoded_dataset(tokenizer, validation_texts, y_validation, args.batch_size), device), np.asarray(metadata["threshold"]))
    if not np.isclose(result["f1_micro"], metadata["validation_metrics"]["f1_micro"]):
        raise ValueError("Reproduced validation score does not match the training run")
    metadata["validation_metrics"] = result
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("category\tsupport\tprecision\trecall\tf1\tFP\tFN")
    for category, values in result["per_category"].items():
        print(f"{category}\t{values['support']}\t{values['precision']:.4f}\t{values['recall']:.4f}\t{values['f1']:.4f}\t{values['false_positive']}\t{values['false_negative']}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--model", default="artifacts/model"); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--device"); evaluate(parser.parse_args())


if __name__ == "__main__": main()
