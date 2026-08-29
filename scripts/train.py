import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from classifier.modeling import BACKBONES, DEFAULT_BACKBONE, MultilabelClassifier, backbone_spec, weighted_multilabel_loss
from classifier.preprocessing import FORMULA_TOKEN, MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION, SECONDARY_LABEL_LOSS_WEIGHT, register_formula_token
from scripts.data import LABELS, build_weighted_labels, load_category_examples, load_tag_aware_examples, load_uniform_examples

MODEL_NAME = DEFAULT_BACKBONE


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    result = {"accuracy": float(accuracy_score(y_true, predictions))}
    for average in ("micro", "macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average=average, zero_division=0)
        result.update({f"precision_{average}": float(precision), f"recall_{average}": float(recall), f"f1_{average}": float(f1)})
    result["per_category"] = {}
    for index, label in enumerate(LABELS):
        truth, predicted = y_true[:, index].astype(int), predictions[:, index]
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
        result["per_category"][label] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(truth.sum()), "predicted_positive": int(predicted.sum()), "true_positive": int(((truth == 1) & (predicted == 1)).sum()), "false_positive": int(((truth == 0) & (predicted == 1)).sum()), "false_negative": int(((truth == 1) & (predicted == 0)).sum())}
    return result


def tune_thresholds(y_true, probabilities, candidates):
    return [max(candidates, key=lambda value: f1_score(y_true[:, index], probabilities[:, index] >= value, zero_division=0)) for index in range(y_true.shape[1])]


def balanced_training_indices(labels, size, minimum_per_label, seed=42):
    if size > len(labels):
        raise ValueError("Training size cannot exceed the candidate pool")
    rng, selected = np.random.default_rng(seed), set()
    for label_index in np.argsort(labels.sum(axis=0)):
        needed = max(0, int(minimum_per_label - sum(labels[index, label_index] for index in selected)))
        candidates = np.array([index for index in np.flatnonzero(labels[:, label_index]) if index not in selected])
        if len(candidates):
            selected.update(rng.choice(candidates, size=min(needed, len(candidates)), replace=False).tolist())
    if len(selected) > size:
        raise ValueError("Training size is too small for the requested label minimum")
    remaining = np.array([index for index in range(len(labels)) if index not in selected])
    selected.update(rng.choice(remaining, size=size - len(selected), replace=False).tolist())
    indices = np.array(sorted(selected)); rng.shuffle(indices)
    return indices


def class_weights(labels, method):
    inverse = (len(labels) - labels.sum(axis=0)) / np.maximum(labels.sum(axis=0), 1.0)
    if method == "none": return np.ones(labels.shape[1], dtype=np.float32)
    if method == "sqrt_inverse": return np.sqrt(inverse).astype(np.float32)
    return np.clip(inverse, 1.0, 50.0).astype(np.float32)



def add_formula_token(tokenizer, model):
    register_formula_token(tokenizer)
    model.backbone.resize_token_embeddings(len(tokenizer))


def arrays(examples):
    return np.array([item[0] for item in examples], dtype=object), np.array([item[1] for item in examples], dtype=np.float32)


class TextDataset(Dataset):
    def __init__(self, texts, labels): self.texts, self.labels = texts.tolist(), labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, index): return self.texts[index], self.labels[index]


def encoded_dataset(tokenizer, texts, labels, batch_size, shuffle=False, generator=None):
    def collate(batch):
        batch_texts, batch_labels = zip(*batch)
        tokens = tokenizer(list(batch_texts), padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH, return_tensors="pt")
        return tokens, torch.tensor(np.asarray(batch_labels), dtype=torch.float32)
    return DataLoader(TextDataset(np.asarray(texts, dtype=object), np.asarray(labels, dtype=np.float32)), batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


def prepare_data(args):
    if args.sampling == "uniform":
        split = load_uniform_examples(args.dataset, args.per_label, args.validation_limit, args.test_limit)
        train_texts, y_train = arrays(split["training"]); validation_texts, y_validation = arrays(split["validation"]); test_texts, y_test = arrays(split["test"])
        return train_texts, y_train, y_train, validation_texts, y_validation, test_texts, y_test, {"sampling": "uniform", "preprocessing": PREPROCESSING_VERSION, "per_label_target": args.per_label, "training_candidates": split["training_candidates"], "eligible_records": split["eligible"], "validation_records": len(validation_texts), "test_records": len(test_texts)}
    split = load_tag_aware_examples(args.dataset, args.train_limit, args.minimum_label_examples, args.validation_limit, args.test_limit) if args.sampling == "tag_aware" else load_category_examples(args.dataset, args.limit, args.validation_limit, args.test_limit)
    candidate_texts, candidate_labels = arrays(split["training"]); validation_texts, y_validation = arrays(split["validation"]); test_texts, y_test = arrays(split["test"])
    indices = balanced_training_indices(candidate_labels, args.train_limit, args.minimum_label_examples)
    train_texts, y_train = candidate_texts[indices], candidate_labels[indices]
    weighted_labels = y_train
    primary_counts = y_train.sum(axis=0).astype(int)
    secondary_counts = np.zeros(len(LABELS), dtype=int)
    if args.sampling == "tag_aware":
        primary = np.asarray(split["training_primary_values"], dtype=np.float32)[indices]
        weighted_labels = build_weighted_labels(y_train, primary)
        secondary = (y_train == 1) & (primary == 0)
        primary_counts, secondary_counts = ((y_train == 1) & ~secondary).sum(axis=0), secondary.sum(axis=0)
    details = {"sampling": args.sampling, "preprocessing": PREPROCESSING_VERSION, "sample_size": len(candidate_texts), "candidate_training_size": len(candidate_texts), "eligible_records": split["eligible"], "training_candidates": split["training_candidates"], "sample_label_counts": split["training_label_counts"], "minimum_label_examples": args.minimum_label_examples, "secondary_label_loss_weight": SECONDARY_LABEL_LOSS_WEIGHT, "training_primary_label_counts": dict(zip(LABELS, primary_counts.astype(int).tolist())), "training_secondary_label_counts": dict(zip(LABELS, secondary_counts.astype(int).tolist())), "deduplication": "normalized_abstract", "validation_records": len(validation_texts), "test_records": len(test_texts)}
    return train_texts, y_train, weighted_labels, validation_texts, y_validation, test_texts, y_test, details


def device_for(args): return torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def probabilities(model, data, device):
    model.eval(); values = []
    with torch.inference_mode():
        for tokens, _ in data:
            values.append(torch.sigmoid(model(**{key: value.to(device) for key, value in tokens.items()})).cpu())
    return torch.cat(values).numpy()


def train_epoch(model, data, optimizer, scheduler, weights, device):
    model.train()
    for tokens, weighted_labels in data:
        optimizer.zero_grad(set_to_none=True)
        logits = model(**{key: value.to(device) for key, value in tokens.items()})
        loss = weighted_multilabel_loss(logits, weighted_labels.to(device), weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()


def train(args):
    torch.manual_seed(42); np.random.seed(42)
    train_texts, y_train, weighted_labels, validation_texts, y_validation, test_texts, y_test, details = prepare_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultilabelClassifier.from_pretrained(args.model_name, len(LABELS), backbone_spec(args.model_name).pooling)
    add_formula_token(tokenizer, model)
    device = device_for(args); model.to(device)
    training = encoded_dataset(tokenizer, train_texts, weighted_labels, args.batch_size, shuffle=True); validation = encoded_dataset(tokenizer, validation_texts, y_validation, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, round(len(training) * args.epochs * args.warmup_ratio)), len(training) * args.epochs)
    for _ in range(args.epochs): train_epoch(model, training, optimizer, scheduler, class_weights(y_train, args.class_weighting), device)
    validation_probabilities = probabilities(model, validation, device)
    selected = tune_thresholds(y_validation, validation_probabilities, args.thresholds)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); model.save(output); tokenizer.save_pretrained(output)
    test_metrics = metrics(y_test, probabilities(model, encoded_dataset(tokenizer, test_texts, y_test, args.batch_size), device), np.asarray(selected)) if test_texts is not None else None
    metadata = {"backend": "pytorch", "base_model": args.model_name, "pooling": model.pooling, "labels": LABELS, "threshold": selected, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "clipnorm": 1.0, "warmup_ratio": args.warmup_ratio, "validation_metrics": metrics(y_validation, validation_probabilities, np.asarray(selected)), "threshold_metrics": [{"threshold": value, **metrics(y_validation, validation_probabilities, value)} for value in args.thresholds], "test_metrics": test_metrics, **details, "training_size": len(train_texts), "training_label_counts": dict(zip(LABELS, y_train.sum(axis=0).astype(int).tolist())), "class_weighting": args.class_weighting, "positive_class_weights": dict(zip(LABELS, class_weights(y_train, args.class_weighting).tolist())), "epochs": args.epochs, "max_length": MAX_CONTEXT_LENGTH, "formula_token": FORMULA_TOKEN}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": selected, "metrics": metadata["validation_metrics"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--output", default="artifacts/model"); parser.add_argument("--model-name", choices=tuple(BACKBONES), default=MODEL_NAME); parser.add_argument("--sampling", choices=("tag_aware", "category_aware", "uniform"), default="tag_aware"); parser.add_argument("--per-label", type=int, default=5000); parser.add_argument("--validation-limit", type=int, default=20000); parser.add_argument("--test-limit", type=int, default=20000); parser.add_argument("--limit", type=int, default=250000); parser.add_argument("--train-limit", type=int, default=100000); parser.add_argument("--minimum-label-examples", type=int, default=8000); parser.add_argument("--class-weighting", choices=("none", "sqrt_inverse", "inverse"), default="none"); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--learning-rate", type=float, default=3e-5); parser.add_argument("--weight-decay", type=float, default=0.0); parser.add_argument("--warmup-ratio", type=float, default=0.0); parser.add_argument("--device"); parser.add_argument("--thresholds", type=float, nargs="+", default=[.3, .4, .5, .6, .7, .8, .9, .95]); train(parser.parse_args())


if __name__ == "__main__": main()
