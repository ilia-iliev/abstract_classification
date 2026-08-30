import argparse
import json
from pathlib import Path

import numpy as np
import torch
from classifier.modeling import MultilabelClassifier, load_tokenizer
from classifier.preprocessing import MAX_CONTEXT_LENGTH, prepare_abstract
from scripts.data import LABELS, load_partition_records


def prediction_errors(records, probabilities, label_index, error_type, limit):
    expected = 0 if error_type == "false positive" else 1
    predicted = 1 - expected
    errors = []
    for record, scores in zip(records, probabilities):
        truth = record["labels"][label_index]
        decision = int(scores[label_index] >= record["thresholds"][label_index])
        if truth == expected and decision == predicted:
            errors.append((float(scores[label_index]), record, scores))
    errors.sort(key=lambda item: item[0], reverse=error_type == "false positive")
    return errors[:limit]


def render_example(score, record, scores):
    prepared = prepare_abstract(record["abstract"])
    labels = [label for label, value in zip(LABELS, record["labels"]) if value]
    score_text = ", ".join(f"{label}={value:.4f}" for label, value in zip(LABELS, scores))
    return [
        f"#### `{record['id']}` — error score {score:.4f}", "",
        f"- arXiv subjects: `{record['categories']}`",
        f"- broad labels: `{', '.join(labels)}`",
        f"- scores: {score_text}", "", "**Raw abstract**", "",
        record["abstract"].replace("\n", " "), "", "**Preprocessed abstract**", "", prepared, "",
    ]


def predict(model, tokenizer, texts, batch_size, device):
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH, return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in encoded.items()})
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def generate(args):
    model_dir = Path(args.model)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    records = load_partition_records(args.dataset, args.partition, metadata[f"{args.partition}_records"])
    thresholds = np.asarray(metadata["threshold"])
    for record in records:
        record["thresholds"] = thresholds
    texts = [prepare_abstract(record["abstract"]) for record in records]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_dir)
    model = MultilabelClassifier.load(model_dir, len(LABELS), map_location=device).to(device).eval()
    probabilities = predict(model, tokenizer, texts, args.batch_size, device)

    lines = [
        "# Classification error examples", "",
        f"Partition: **{args.partition}** ({len(records):,} records)", "",
        "Examples are ordered by confidence: highest-score false positives and lowest-score false negatives.", "",
    ]
    for label_index, label in enumerate(LABELS):
        lines.extend([f"## {label.replace('_', ' ').title()}", ""])
        for error_type in ("false positive", "false negative"):
            lines.extend([f"### {error_type.title()}", ""])
            for score, record, scores in prediction_errors(records, probabilities, label_index, error_type, args.examples):
                lines.extend(render_example(score, record, scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", default="artifacts/model")
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("logs/error-report.md"))
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
