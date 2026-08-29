"""Train each Optuna-selected benchmark configuration exactly once."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from classifier.modeling import BACKBONES, MultilabelClassifier, backbone_spec, weighted_multilabel_loss
from classifier.preprocessing import FORMULA_TOKEN, MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION, SECONDARY_LABEL_TARGET, register_formula_token
from scripts.data import LABELS
from scripts.run_frozen_probes import arrays, load_splits, logits_and_probabilities
from scripts.train import encoded_dataset, metrics, tune_thresholds

STAGE = "selected_configuration_training"
CHECKPOINT_SELECTION_RULE = "final checkpoint after the required single epoch"
LOSS_DESCRIPTION = "weighted_multilabel_loss with unit positive weights"
PRECISION_POLICY = "float32"
GRADIENT_CLIP_NORM = 1.0
TUNING_TRIAL_BUDGET = 16


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dependency_versions():
    packages = ("numpy", "optuna", "scikit-learn", "torch", "transformers")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def git_commit():
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def peak_vram(device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def train_one_epoch(model, data, optimizer, scheduler, device, gradient_accumulation):
    """Train with the same mean-loss accumulation semantics used in tuning."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batches = len(data)
    for batch_index, (tokens, targets) in enumerate(data, 1):
        logits = model(**{key: value.to(device) for key, value in tokens.items()})
        loss = weighted_multilabel_loss(
            logits, targets.to(device), np.ones(len(LABELS), dtype=np.float32)
        )
        total_loss += loss.item()
        remainder = batches % gradient_accumulation
        group_size = remainder if remainder and batch_index > batches - remainder else gradient_accumulation
        (loss / group_size).backward()
        if batch_index % gradient_accumulation == 0 or batch_index == batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return {"epoch": 1, "mean_training_loss": total_loss / batches, "optimizer_steps": math.ceil(batches / gradient_accumulation)}


def selected_parameters(tuning, model_name):
    path = Path(tuning) / model_name.replace("/", "--") / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuning summary for {model_name}: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    params = summary.get("best_params")
    if summary.get("model") != model_name or not isinstance(params, dict):
        raise ValueError(f"Invalid tuning summary: {path}")
    expected = {"learning_rate", "weight_decay", "warmup_ratio"}
    if set(params) != expected:
        raise ValueError(f"Tuning summary has unexpected selected parameters: {path}")
    trials = summary.get("trials", [])
    selected = [trial for trial in trials if trial.get("number") == summary.get("best_trial")]
    if len(trials) != TUNING_TRIAL_BUDGET or len(selected) != 1 or selected[0].get("state") != "COMPLETE":
        raise ValueError(f"Tuning trial budget or selected trial is invalid: {path}")
    if selected[0].get("params") != params:
        raise ValueError(f"Selected parameters do not match the selected trial: {path}")
    return summary, params, path


def validate_tuning_configuration(tuning, dataset_metadata):
    path = Path(tuning) / "configuration.json"
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "stage": "hyperparameter_tuning",
        "dataset_id": dataset_metadata["dataset_id"],
        "snapshot_sha256": dataset_metadata["snapshot_sha256"],
        "split_seed": dataset_metadata["split_seed"],
        "epochs_per_trial": 1,
        "loss": LOSS_DESCRIPTION,
        "secondary_label_target": SECONDARY_LABEL_TARGET,
        "max_length": MAX_CONTEXT_LENGTH,
        "formula_token": FORMULA_TOKEN,
        "preprocessing": PREPROCESSING_VERSION,
        "precision_policy": PRECISION_POLICY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
    }
    mismatched = [key for key, value in required.items() if configuration.get(key) != value]
    if mismatched:
        raise ValueError(f"Tuning configuration is incompatible: {', '.join(mismatched)}")
    return configuration


def train_model(args, model_name, dataset, dataset_metadata, tuning_configuration, output):
    summary, params, summary_path = selected_parameters(args.tuning, model_name)
    train_ids, train_texts, train_targets = arrays(dataset["training"], "targets")
    validation_ids, validation_texts, validation_labels = arrays(dataset["validation"], "labels")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    register_formula_token(tokenizer)
    model = MultilabelClassifier.from_pretrained(model_name, len(LABELS), backbone_spec(model_name).pooling)
    model.backbone.resize_token_embeddings(len(tokenizer))
    model.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    training = encoded_dataset(tokenizer, train_texts, train_targets, tuning_configuration["batch_size"], shuffle=True, generator=generator)
    validation = encoded_dataset(tokenizer, validation_texts, validation_labels, tuning_configuration["batch_size"])
    gradient_accumulation = tuning_configuration["gradient_accumulation"]
    total_steps = math.ceil(len(training) / gradient_accumulation)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_steps * params["warmup_ratio"]), total_steps)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history = [train_one_epoch(model, training, optimizer, scheduler, device, gradient_accumulation)]
    duration = time.perf_counter() - started
    training_peak_vram = peak_vram(device)
    logits, probabilities = logits_and_probabilities(model, validation, device)
    thresholds = tune_thresholds(validation_labels, probabilities, tuning_configuration["threshold_candidates"])
    validation_metrics = metrics(validation_labels, probabilities, np.asarray(thresholds))
    output.mkdir()
    model.save(output)
    tokenizer.save_pretrained(output)
    np.savez_compressed(output / "validation_predictions.npz", ids=validation_ids, labels=validation_labels, logits=logits, probabilities=probabilities)
    runtime = {
        "training_wall_seconds": duration,
        "gpu_hours": duration / 3600 if device.type == "cuda" else 0.0,
        "peak_vram_bytes": training_peak_vram,
        "device": str(device),
        "precision_policy": PRECISION_POLICY,
    }
    data_configuration = {
        "dataset_id": dataset_metadata["dataset_id"],
        "snapshot_sha256": dataset_metadata["snapshot_sha256"],
        "split_seed": dataset_metadata["split_seed"],
        "manifests": {name: dataset_metadata["splits"][name]["manifest_sha256"] for name in ("training", "validation")},
        "training_records": len(train_ids),
        "validation_records": len(validation_ids),
        "training_order_sha256": sha256_json(train_ids.tolist()),
        "validation_order_sha256": sha256_json(validation_ids.tolist()),
    }
    configuration = {
        "stage": STAGE,
        "backend": "pytorch",
        "base_model": model_name,
        "pooling": model.pooling,
        "labels": LABELS,
        "selected_from": {"summary": str(summary_path), "trial": summary["best_trial"]},
        "hyperparameters": params,
        "epochs": 1,
        "batch_size": tuning_configuration["batch_size"],
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": tuning_configuration["effective_batch_size"],
        "loss": LOSS_DESCRIPTION,
        "secondary_label_target": SECONDARY_LABEL_TARGET,
        "max_length": MAX_CONTEXT_LENGTH,
        "formula_token": FORMULA_TOKEN,
        "preprocessing": PREPROCESSING_VERSION,
        "precision_policy": PRECISION_POLICY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "threshold_candidates": tuning_configuration["threshold_candidates"],
        "threshold": thresholds,
        "data": data_configuration,
        "dependencies": dependency_versions(),
        "git_commit": git_commit(),
    }
    metadata = {**configuration, "validation_metrics": validation_metrics}
    write_json(output / "configuration.json", configuration)
    write_json(output / "metadata.json", metadata)
    write_json(output / "metrics.json", validation_metrics)
    write_json(output / "history.json", history)
    write_json(output / "runtime.json", runtime)
    return {"model": model_name, "artifact": str(output), "validation_macro_f1": validation_metrics["f1_macro"], **runtime}


def main():
    parser = argparse.ArgumentParser(description="Train the four selected one-epoch benchmark configurations.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("tuning", type=Path, help="Completed artifacts/hyperparameter-tuning directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/selected-configurations"))
    parser.add_argument("--models", nargs="+", choices=tuple(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--seed", type=int, help="Must match the fixed tuning training seed")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite final-training output: {args.output}")
    dataset_metadata, dataset = load_splits(args.snapshot, args.manifests)
    tuning_configuration = validate_tuning_configuration(args.tuning, dataset_metadata)
    if tuning_configuration["batch_size"] < 1 or tuning_configuration["gradient_accumulation"] < 1:
        raise ValueError("Tuning configuration has an invalid batch size")
    if any(model not in tuning_configuration.get("models", []) for model in args.models):
        raise ValueError("Requested model was not included in the tuning configuration")
    configured_seed = tuning_configuration.get("seed")
    if configured_seed is None:
        raise ValueError("Tuning configuration does not record its training seed")
    if args.seed is not None and args.seed != configured_seed:
        raise ValueError("--seed must match the tuning configuration")
    args.seed = configured_seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    try:
        results = []
        for index, model_name in enumerate(args.models):
            directory = staging / f"{index + 1:02d}-{model_name.replace('/', '--')}"
            results.append(train_model(args, model_name, dataset, dataset_metadata, tuning_configuration, directory))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for result in results:
            result["artifact"] = str(args.output / Path(result["artifact"]).name)
        write_json(staging / "configuration.json", {
            "stage": STAGE, "models": args.models, "seed": args.seed,
            "tuning_configuration": str(args.tuning / "configuration.json"),
            "dataset_id": dataset_metadata["dataset_id"],
        })
        write_json(staging / "summary.json", {"stage": STAGE, "results": results})
        staging.replace(args.output)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
