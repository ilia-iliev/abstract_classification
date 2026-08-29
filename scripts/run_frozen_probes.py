"""Train frozen-backbone probes on the immutable benchmark manifests."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from classifier.modeling import BACKBONES, MultilabelClassifier, backbone_spec, weighted_multilabel_loss
from classifier.preprocessing import FORMULA_TOKEN, MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION, register_formula_token
from scripts.data import LABELS, load_manifest_examples
from scripts.train import encoded_dataset, metrics, tune_thresholds


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(snapshot, manifests):
    metadata = json.loads((Path(manifests) / "dataset.json").read_text(encoding="utf-8"))
    if metadata.get("labels") != LABELS:
        raise ValueError("Manifest labels do not match classifier labels")
    if sha256(snapshot) != metadata.get("snapshot_sha256"):
        raise ValueError("Snapshot hash does not match the immutable dataset manifest")
    result = {}
    for name in ("training", "validation"):
        path = Path(manifests) / f"{name}.jsonl"
        expected = metadata["splits"][name]
        if sha256(path) != expected["manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch: {path}")
        rows = load_manifest_examples(snapshot, manifests, name)
        if len(rows) != expected["records"]:
            raise ValueError(f"Manifest count mismatch: {path}")
        result[name] = rows
    return metadata, result


def arrays(rows, target_key):
    return (
        np.asarray([row["id"] for row in rows]),
        np.asarray([row["text"] for row in rows], dtype=object),
        np.asarray([row[target_key] for row in rows], dtype=np.float32),
    )


def freeze_backbone(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Backbone freeze failed")


def logits_and_probabilities(model, data, device):
    model.eval()
    logits = []
    with torch.inference_mode():
        for tokens, _ in data:
            batch = {key: value.to(device) for key, value in tokens.items()}
            logits.append(model(**batch).cpu())
    values = torch.cat(logits).numpy()
    return values, 1.0 / (1.0 + np.exp(-values))


def train_head(model, training, optimizer, scheduler, device):
    model.train()
    for tokens, targets in training:
        optimizer.zero_grad(set_to_none=True)
        logits = model(**{key: value.to(device) for key, value in tokens.items()})
        loss = weighted_multilabel_loss(logits, targets.to(device), np.ones(len(LABELS), dtype=np.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


def peak_vram(device):
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def run_model(args, dataset, model_name, output):
    train_ids, train_texts, train_targets = arrays(dataset["training"], "targets")
    validation_ids, validation_texts, validation_labels = arrays(dataset["validation"], "labels")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    register_formula_token(tokenizer)
    model = MultilabelClassifier.from_pretrained(model_name, len(LABELS), backbone_spec(model_name).pooling)
    model.backbone.resize_token_embeddings(len(tokenizer))
    freeze_backbone(model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    training = encoded_dataset(tokenizer, train_texts, train_targets, args.batch_size, shuffle=True, generator=generator)
    validation = encoded_dataset(tokenizer, validation_texts, validation_labels, args.batch_size)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(training) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_steps * args.warmup_ratio), total_steps)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.epochs):
        train_head(model, training, optimizer, scheduler, device)
    duration = time.perf_counter() - started
    training_peak_vram = peak_vram(device)
    logits, probabilities = logits_and_probabilities(model, validation, device)
    thresholds = tune_thresholds(validation_labels, probabilities, args.thresholds)
    result = metrics(validation_labels, probabilities, np.asarray(thresholds))
    output.mkdir(parents=True)
    model.save(output)
    tokenizer.save_pretrained(output)
    np.savez_compressed(output / "validation_predictions.npz", ids=validation_ids, labels=validation_labels, logits=logits, probabilities=probabilities)
    configuration = {
        "stage": "frozen_representation_probe",
        "backend": "pytorch",
        "base_model": model_name,
        "pooling": model.pooling,
        "backbone_frozen": True,
        "trainable_parameters": [name for name, value in model.named_parameters() if value.requires_grad],
        "labels": LABELS,
        "preprocessing": PREPROCESSING_VERSION,
        "formula_token": FORMULA_TOKEN,
        "max_length": MAX_CONTEXT_LENGTH,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "threshold_candidates": args.thresholds,
        "threshold": thresholds,
    }
    runtime = {"training_wall_seconds": duration, "peak_vram_bytes": training_peak_vram, "device": str(device)}
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    return {"model": model_name, "artifact": str(output), "macro_f1": result["f1_macro"], "micro_f1": result["f1_micro"], "wall_seconds": duration, "peak_vram_bytes": runtime["peak_vram_bytes"]}


def write_report(output, dataset_metadata, results):
    report = {"stage": "frozen_representation_probe", "dataset_id": dataset_metadata["dataset_id"], "snapshot_sha256": dataset_metadata["snapshot_sha256"], "split_seed": dataset_metadata["split_seed"], "splits": {name: dataset_metadata["splits"][name] for name in ("training", "validation")}, "results": results}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen-representation probes", "", "| Model | Macro F1 | Micro F1 | Train seconds | Peak VRAM (bytes) |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['model']} | {row['macro_f1']:.4f} | {row['micro_f1']:.4f} | {row['wall_seconds']:.2f} | {row['peak_vram_bytes'] if row['peak_vram_bytes'] is not None else 'n/a'} |" for row in results)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run head-only probes for every benchmark backbone.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/frozen-probes"))
    parser.add_argument("--models", nargs="+", choices=tuple(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.3, .4, .5, .6, .7, .8, .9, .95])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite probe output: {args.output}")
    metadata, dataset = load_splits(args.snapshot, args.manifests)
    args.output.mkdir(parents=True)
    results = []
    for index, model_name in enumerate(args.models):
        directory = args.output / f"{index + 1:02d}-{model_name.rsplit('/', 1)[-1]}"
        results.append(run_model(args, dataset, model_name, directory))
    write_report(args.output, metadata, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
