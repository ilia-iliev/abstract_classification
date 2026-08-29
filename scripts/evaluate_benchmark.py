"""Evaluate the four frozen classifiers once on the immutable benchmark split."""

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

from classifier.modeling import BACKBONES, MultilabelClassifier
from classifier.preprocessing import FORMULA_TOKEN, MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION
from scripts.data import LABELS, load_manifest_examples, records
from scripts.train import encoded_dataset

EVALUATION_VERSION = 2
DEFAULT_BOOTSTRAP_SAMPLES = 1000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ece(labels, probabilities, bins=15):
    """Expected calibration error over independent label decisions."""
    truth = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(probabilities, dtype=np.float64).ravel()
    total = len(truth)
    value = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (scores >= lower) & ((scores < upper) if upper < 1 else (scores <= upper))
        if mask.any():
            value += mask.mean() * abs(truth[mask].mean() - scores[mask].mean())
    return float(value)


def top1_accuracy(labels, probabilities):
    top = np.asarray(probabilities).argmax(axis=1)
    return float(np.asarray(labels)[np.arange(len(labels)), top].mean())


def quality_metrics(labels, probabilities, thresholds):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= thresholds).astype(int)
    result = {
        "exact_match_accuracy": float(accuracy_score(labels, predictions)),
        "top1_accuracy": top1_accuracy(labels, probabilities),
        "bce_log_loss": float(log_loss(labels.ravel(), probabilities.ravel(), labels=[0, 1])),
        "calibration_error_ece_15": ece(labels, probabilities),
    }
    for average in ("micro", "macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        result.update({
            f"precision_{average}": float(precision),
            f"recall_{average}": float(recall),
            f"f1_{average}": float(f1),
        })
    per_label = {}
    for index, label in enumerate(LABELS):
        truth, predicted = labels[:, index], predictions[:, index]
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        per_label[label] = {
            "support": int(truth.sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "pr_auc": float(average_precision_score(truth, probabilities[:, index])),
        }
    result["per_label"] = per_label
    return result


def bootstrap(metrics_function, labels, model_probabilities, thresholds, samples, seed, split):
    """Paired example bootstrap for each model and every model difference."""
    rng = np.random.default_rng(seed)
    count = len(labels)
    names = tuple(model_probabilities)
    scalar_names = tuple(key for key, value in metrics_function(labels, model_probabilities[names[0]], thresholds[names[0]]).items() if isinstance(value, float))
    values = {name: {metric: [] for metric in scalar_names} for name in names}
    differences = {
        f"{left}__minus__{right}": {metric: [] for metric in scalar_names}
        for position, left in enumerate(names) for right in names[position + 1:]
    }
    for _ in range(samples):
        indices = rng.integers(0, count, size=count)
        measured = {
            name: metrics_function(labels[indices], probabilities[indices], thresholds[name])
            for name, probabilities in model_probabilities.items()
        }
        for name in names:
            for metric in scalar_names:
                values[name][metric].append(measured[name][metric])
        for comparison, metric_values in differences.items():
            left, right = comparison.split("__minus__")
            for metric in scalar_names:
                metric_values[metric].append(measured[left][metric] - measured[right][metric])

    def interval(point, observations):
        lower, upper = np.percentile(observations, (2.5, 97.5))
        return {"point": float(point), "lower": float(lower), "upper": float(upper), "level": 0.95}

    points = {name: metrics_function(labels, probabilities, thresholds[name]) for name, probabilities in model_probabilities.items()}
    return {
        "method": f"paired nonparametric bootstrap over {split} examples; uncertainty excludes training-seed variation",
        "samples": samples, "seed": seed,
        "models": {name: {metric: interval(points[name][metric], values[name][metric]) for metric in scalar_names} for name in names},
        "differences": {
            comparison: {
                metric: interval(points[left][metric] - points[right][metric], observations)
                for metric, observations in metric_values.items()
            }
            for comparison, metric_values in differences.items()
            for left, right in [comparison.split("__minus__")]
        },
    }


def split_rows(snapshot, frozen, split):
    manifest = frozen / "manifests" / f"{split}.jsonl"
    dataset = json.loads((frozen / "manifests" / "dataset.json").read_text(encoding="utf-8"))
    expected = dataset.get("splits", {}).get(split)
    if expected is None:
        raise ValueError(f"Frozen dataset has no {split} split")
    if sha256(manifest) != expected["manifest_sha256"]:
        raise ValueError(f"Frozen {split} manifest hash mismatch")
    rows = load_manifest_examples(snapshot, frozen / "manifests", split)
    if len(rows) != expected["records"]:
        raise ValueError(f"Frozen {split} manifest count mismatch")
    primary_by_id = {str(record["id"]): record.get("categories", "").split(maxsplit=1)[0] for record in records(snapshot)}
    for row in rows:
        if row["id"] not in primary_by_id:
            raise ValueError(f"Snapshot is missing {split} record {row['id']}")
        row["primary_category"] = primary_by_id[row["id"]]
    return dataset, rows


def validate_freeze(frozen, snapshot, metric_code_key="benchmark_metric_code", additional_metric_code_keys=()):
    frozen = Path(frozen)
    freeze = json.loads((frozen / "freeze.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_benchmark_evaluation":
        raise ValueError("Experiment was not frozen before benchmark evaluation")
    if sha256(snapshot) != freeze.get("dataset", {}).get("snapshot_sha256"):
        raise ValueError("Snapshot does not match frozen experiment")
    root = Path(__file__).resolve().parents[1]
    for item in freeze.get("models", []):
        directory = frozen / item["artifact"]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for file in item["files"]:
            path = directory / file["path"]
            if not path.is_file() or sha256(path) != file["sha256"]:
                raise ValueError(f"Frozen model file hash mismatch: {path}")
    if set(item["model"] for item in freeze.get("models", [])) != set(BACKBONES):
        raise ValueError("Frozen experiment does not contain exactly the four benchmark models")
    metric_code = [freeze.get(metric_code_key)]
    metric_code.extend(freeze.get(key) for key in additional_metric_code_keys)
    if any(item is None for item in metric_code):
        missing = [key for key, item in zip((metric_code_key, *additional_metric_code_keys), metric_code) if item is None]
        raise ValueError(f"Frozen experiment does not attest {', '.join(missing)}")
    frozen_code = (freeze["preprocessing"]["code"], freeze["classification_head_and_pooling"], freeze["data_loading_code"], freeze["metric_code"], *metric_code)
    for details in frozen_code:
        if sha256(frozen / "code" / details["path"]) != details["sha256"]:
            raise ValueError(f"Frozen code hash mismatch: {details['path']}")
        if sha256(root / details["path"]) != details["sha256"]:
            raise ValueError(f"Evaluation code has changed since freeze: {details['path']}")
    return freeze


def predict(model_directory, rows, batch_size, device):
    configuration = json.loads((model_directory / "configuration.json").read_text(encoding="utf-8"))
    if configuration.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError(f"Unexpected preprocessing in {model_directory}")
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = MultilabelClassifier.load(model_directory, len(LABELS)).to(device).eval()
    texts = np.asarray([row["text"] for row in rows], dtype=object)
    labels = np.asarray([row["labels"] for row in rows], dtype=np.float32)
    logits = []
    with torch.inference_mode():
        for tokens, _ in encoded_dataset(tokenizer, texts, labels, batch_size):
            logits.append(model(**{key: value.to(device) for key, value in tokens.items()}).cpu())
    token_lengths = np.asarray([len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts])
    values = torch.cat(logits).numpy()
    return configuration, values, 1 / (1 + np.exp(-values)), token_lengths


def subgroup_masks(rows, token_lengths):
    labels = np.asarray([row["labels"] for row in rows], dtype=int)
    targets = np.asarray([row["targets"] for row in rows], dtype=float)
    formula_counts = np.asarray([row["text"].count(FORMULA_TOKEN) for row in rows])
    word_counts = np.asarray([len(row["text"].split()) for row in rows])
    return {
        "primary_label_decisions": targets == 1.0,
        "secondary_only_label_decisions": targets == 0.5,
        "chemistry_physics_overlap": (labels[:, 1] & labels[:, 3]).astype(bool),
        "multilabel_records": labels.sum(axis=1) > 1,
        "single_label_records": labels.sum(axis=1) == 1,
        "formula_heavy": formula_counts >= 2,
        "truncated": token_lengths > MAX_CONTEXT_LENGTH,
        "non_truncated": token_lengths <= MAX_CONTEXT_LENGTH,
        "length_words_0_99": word_counts < 100,
        "length_words_100_199": (word_counts >= 100) & (word_counts < 200),
        "length_words_200_399": (word_counts >= 200) & (word_counts < 400),
        "length_words_400_plus": word_counts >= 400,
    }


def subgroup_metrics(rows, probabilities, thresholds, token_lengths):
    labels = np.asarray([row["labels"] for row in rows], dtype=int)
    masks = subgroup_masks(rows, token_lengths)
    result = {}
    for name, mask in masks.items():
        if mask.ndim == 2:
            truth, scores, limit = labels[mask], probabilities[mask], np.broadcast_to(thresholds, labels.shape)[mask]
            predictions = (scores >= limit).astype(int)
            result[name] = {"decisions": int(mask.sum()), "precision": float(precision_recall_fscore_support(truth, predictions, average="binary", zero_division=0)[0]), "recall": float(precision_recall_fscore_support(truth, predictions, average="binary", zero_division=0)[1]), "f1": float(f1_score(truth, predictions, zero_division=0))}
        elif mask.any():
            result[name] = {"records": int(mask.sum()), **quality_metrics(labels[mask], probabilities[mask], thresholds)}
        else:
            result[name] = {"records": 0}
    return result


def evaluate_split(frozen, snapshot, output, split, metric_code_key, batch_size=16, device=None, bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES, seed=42, additional_metric_code_keys=()):
    frozen, snapshot, output = Path(frozen), Path(snapshot), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {split} report: {output}")
    freeze = validate_freeze(frozen, snapshot, metric_code_key, additional_metric_code_keys)
    dataset, rows = split_rows(snapshot, frozen, split)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output.parent.mkdir(parents=True, exist_ok=True)
    probabilities, thresholds, predictions, prediction_files, tokenization = {}, {}, {}, {}, {}
    subgroup = {}
    for item in freeze["models"]:
        directory = frozen / item["artifact"]
        configuration, logits, scores, token_lengths = predict(directory, rows, batch_size, device)
        if configuration["base_model"] != item["model"] or configuration["threshold"] != item["thresholds"]:
            raise ValueError(f"Frozen configuration mismatch: {directory}")
        name = item["model"]
        probabilities[name], thresholds[name] = scores, np.asarray(configuration["threshold"], dtype=float)
        predictions[name] = (scores >= thresholds[name]).astype(int)
        subgroup[name] = subgroup_metrics(rows, scores, thresholds[name], token_lengths)
        tokenization[name] = {
            "thresholds": thresholds[name].tolist(),
            "token_length": {"min": int(token_lengths.min()), "median": float(np.median(token_lengths)), "max": int(token_lengths.max())},
            "truncated_records": int((token_lengths > MAX_CONTEXT_LENGTH).sum()),
            "truncation_rate": float((token_lengths > MAX_CONTEXT_LENGTH).mean()),
        }
        prediction_files[name] = {
            "ids": np.asarray([row["id"] for row in rows]), "labels": np.asarray([row["labels"] for row in rows]),
            "logits": logits, "probabilities": scores, "predictions": predictions[name],
        }
    labels = np.asarray([row["labels"] for row in rows], dtype=int)
    all_predictions = np.stack(list(predictions.values()))
    disagreements = {"any_prediction_difference": int(np.any(all_predictions != all_predictions[0], axis=(0, 2)).sum())}
    for left_index, left in enumerate(predictions):
        for right in list(predictions)[left_index + 1:]:
            disagreements[f"{left}__vs__{right}"] = int(np.any(predictions[left] != predictions[right], axis=1).sum())
    model_metrics = {name: quality_metrics(labels, scores, thresholds[name]) for name, scores in probabilities.items()}
    report = {"evaluation_version": EVALUATION_VERSION, "split": split, "dataset_id": dataset["dataset_id"], "freeze_sha256": sha256(frozen / "freeze.json"), "records": len(rows), "models": model_metrics, "tokenization": tokenization, "subgroups": subgroup, "model_disagreement_counts": disagreements, "bootstrap": bootstrap(quality_metrics, labels, probabilities, thresholds, bootstrap_samples, seed, split), "uncertainty_note": f"Confidence intervals resample {split} examples only; models were trained once and these are not training-seed intervals."}
    if split == "benchmark":
        report["provisional_ranking"] = [
            name for name, _ in sorted(model_metrics.items(), key=lambda item: (-item[1]["f1_macro"], item[0]))
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, values in prediction_files.items():
            np.savez_compressed(staging / f"{name.replace('/', '--')}.npz", **values)
        write_json(staging / "report.json", report)
        staging.replace(output)
    except Exception:
        for path in staging.rglob("*"):
            if path.is_file(): path.unlink()
        staging.rmdir()
        raise
    return report


def evaluate(frozen, snapshot, output, batch_size=16, device=None, bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES, seed=42):
    return evaluate_split(
        frozen, snapshot, output, "benchmark", "benchmark_metric_code", batch_size, device,
        bootstrap_samples, seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate all frozen models once on the benchmark split.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("frozen", type=Path, help="artifacts/frozen-experiment")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("--batch-size and --bootstrap-samples must be positive")
    report = evaluate(**vars(args))
    print(json.dumps({"output": str(args.output), "records": report["records"], "models": list(report["models"])}, sort_keys=True))


if __name__ == "__main__":
    main()
