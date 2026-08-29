"""Run a parallel, fixed-subset Optuna search for candidate backbones."""

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from classifier.modeling import BACKBONES, MultilabelClassifier, backbone_spec
from classifier.preprocessing import FORMULA_TOKEN, MAX_CONTEXT_LENGTH, PREPROCESSING_VERSION, SECONDARY_LABEL_TARGET, register_formula_token
from scripts.artifacts import write_json
from scripts.data import LABELS
from scripts.run_frozen_probes import arrays, load_splits, logits_and_probabilities
from scripts.training import train_one_epoch
from scripts.train import encoded_dataset, metrics, tune_thresholds

TRIALS_PER_MODEL = 16
TUNING_RECORDS = 5_000
TUNING_MINIMUM_POSITIVES = 500
SEARCH_SPACE = {
    "learning_rate": {"low": 8e-6, "high": 1e-4, "log": True},
    "weight_decay": [0.0, 0.001, 0.01],
    "warmup_ratio": {"low": 0.0, "high": 0.1},
}



def deterministic_rank(seed, record_id, purpose):
    return hashlib.sha256(f"{seed}\0{purpose}\0{record_id}".encode("utf-8")).hexdigest()


def select_tuning_records(training, seed):
    """Select one small, deterministic, rare-label-aware tuning subset."""
    selected = set()
    for index, label in enumerate(LABELS):
        matches = sorted(
            (row for row in training if row["labels"][index]),
            key=lambda row: deterministic_rank(seed, row["id"], f"tuning-{label}"),
        )
        selected.update(row["id"] for row in matches[:TUNING_MINIMUM_POSITIVES])
    if len(selected) > TUNING_RECORDS:
        raise ValueError("Rare-label tuning selection exceeds the fixed tuning subset")
    ordered = sorted(training, key=lambda row: deterministic_rank(seed, row["id"], "tuning-natural"))
    selected.update(row["id"] for row in ordered if len(selected) < TUNING_RECORDS)
    subset = [row for row in ordered if row["id"] in selected]
    if len(subset) != TUNING_RECORDS:
        raise ValueError("Training manifest cannot provide the requested tuning subset")
    return subset


def optimizer_steps(data, gradient_accumulation):
    return math.ceil(len(data) / gradient_accumulation)



def peak_vram(device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def trial_runner(args, model_name, train, validation, trial_root):
    _, train_texts, train_targets = arrays(train, "targets")
    validation_ids, validation_texts, validation_labels = arrays(validation, "labels")
    device = torch.device(args.device)

    def run(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", **SEARCH_SPACE["learning_rate"]),
            "weight_decay": trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", **SEARCH_SPACE["warmup_ratio"]),
        }
        directory = trial_root / f"trial-{trial.number:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        model = None
        started = time.perf_counter()
        try:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            register_formula_token(tokenizer)
            model = MultilabelClassifier.from_pretrained(model_name, len(LABELS), backbone_spec(model_name).pooling)
            model.backbone.resize_token_embeddings(len(tokenizer))
            model.to(device)
            generator = torch.Generator().manual_seed(args.seed)
            training = encoded_dataset(tokenizer, train_texts, train_targets, args.batch_size, shuffle=True, generator=generator)
            validation_data = encoded_dataset(tokenizer, validation_texts, validation_labels, args.batch_size)
            steps = optimizer_steps(training, args.gradient_accumulation)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
            scheduler = get_linear_schedule_with_warmup(optimizer, round(steps * params["warmup_ratio"]), steps)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            train_one_epoch(model, training, optimizer, scheduler, device, args.gradient_accumulation, len(LABELS))
            logits, scores = logits_and_probabilities(model, validation_data, device)
            thresholds = tune_thresholds(validation_labels, scores, args.thresholds)
            result = metrics(validation_labels, scores, np.asarray(thresholds))
            duration = time.perf_counter() - started
            np.savez_compressed(directory / "validation_predictions.npz", ids=validation_ids, labels=validation_labels, logits=logits, probabilities=scores)
            report = {
                "status": "complete", "trial": trial.number, "model": model_name, "device": str(device), "params": params,
                "selection_metric": "validation_macro_f1", "validation_macro_f1": result["f1_macro"], "metrics": result,
                "threshold": thresholds, "duration_seconds": duration, "peak_vram_bytes": peak_vram(device), "epochs": 1,
                "training_records": len(train), "effective_batch_size": args.batch_size * args.gradient_accumulation,
            }
            write_json(directory / "report.json", report)
            trial.set_user_attr("report", str(directory / "report.json"))
            trial.set_user_attr("peak_vram_bytes", report["peak_vram_bytes"])
            return result["f1_macro"]
        except Exception as error:
            report = {"status": "failed", "trial": trial.number, "model": model_name, "device": str(device), "params": params,
                      "duration_seconds": time.perf_counter() - started, "peak_vram_bytes": peak_vram(device),
                      "failure": {"type": type(error).__name__, "message": str(error)}}
            write_json(directory / "report.json", report)
            trial.set_user_attr("report", str(directory / "report.json"))
            raise
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    return run


def sampler(seed):
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=4, constant_liar=True)


def run_claimed_trial(storage, study_name, args, model_name, train, validation, trial_root, number):
    study = optuna.load_study(study_name=study_name, storage=storage)
    objective = trial_runner(args, model_name, train, validation, trial_root)
    trial_id = study._storage.get_trial_id_from_study_id_trial_number(study._study_id, number)
    trial = optuna.trial.Trial(study, trial_id)
    try:
        value = objective(trial)
    except Exception:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    else:
        study.tell(trial, value)


def tune_model(args, model_name, dataset, output):
    model_output = output / model_name.replace("/", "--")
    model_output.mkdir()
    study_path = model_output / "study.db"
    storage = f"sqlite:///{study_path.resolve()}"
    study_name = "subset_validation_macro_f1"
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", sampler=sampler(args.sampler_seed))
    train = select_tuning_records(dataset["training"], args.seed)
    write_json(model_output / "tuning_training_subset.json", {"records": len(train), "ids": [row["id"] for row in train]})
    context = multiprocessing.get_context("spawn")
    for offset in range(0, TRIALS_PER_MODEL, len(args.devices)):
        trials = [study.ask() for _ in args.devices[:TRIALS_PER_MODEL - offset]]
        workers = []
        for device, trial in zip(args.devices, trials):
            worker_args = argparse.Namespace(**vars(args))
            worker_args.device = device
            process = context.Process(target=run_claimed_trial, args=(storage, study_name, worker_args, model_name, train, dataset["validation"], model_output, trial.number))
            process.start()
            workers.append(process)
        for process in workers:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"Trial worker exited with status {process.exitcode}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"All {TRIALS_PER_MODEL} trials failed for {model_name}")
    best = study.best_trial
    summary = {
        "stage": "hyperparameter_tuning", "model": model_name, "trial_budget": TRIALS_PER_MODEL,
        "selection_metric": "validation_macro_f1_after_per_label_threshold_tuning", "best_trial": best.number,
        "best_params": best.params, "best_validation_macro_f1": best.value, "study": str(study_path),
        "trials": [{"number": trial.number, "state": trial.state.name, "value": trial.value, "params": trial.params, "report": trial.user_attrs.get("report")} for trial in study.trials],
    }
    write_json(model_output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Tune candidate backbones on a fixed 5,000-record subset.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hyperparameter-tuning"))
    parser.add_argument("--models", nargs="+", choices=tuple(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.3, .4, .5, .6, .7, .8, .9, .95])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1 or not args.devices:
        parser.error("batch size, accumulation, and devices must be positive")
    if not torch.cuda.is_available() or any(not device.startswith("cuda") for device in args.devices):
        parser.error("This parallel tuner requires CUDA devices")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite tuning output: {args.output}")
    metadata, dataset = load_splits(args.snapshot, args.manifests)
    args.output.mkdir(parents=True)
    configuration = {
        "stage": "hyperparameter_tuning", "dataset_id": metadata["dataset_id"], "snapshot_sha256": metadata["snapshot_sha256"], "split_seed": metadata["split_seed"],
        "models": args.models, "trials_per_model": TRIALS_PER_MODEL, "epochs_per_trial": 1, "search_space": SEARCH_SPACE,
        "tuning_training_records": TUNING_RECORDS, "tuning_minimum_positives_per_label": TUNING_MINIMUM_POSITIVES,
        "validation_records": len(dataset["validation"]), "devices": args.devices, "parallel_trials": len(args.devices),
        "effective_batch_size": args.batch_size * args.gradient_accumulation, "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
        "loss": "weighted_multilabel_loss with unit positive weights", "secondary_label_target": SECONDARY_LABEL_TARGET,
        "max_length": MAX_CONTEXT_LENGTH, "formula_token": FORMULA_TOKEN, "preprocessing": PREPROCESSING_VERSION,
        "threshold_candidates": args.thresholds, "selection_metric": "validation_macro_f1_after_per_label_threshold_tuning",
        "sampler": {"name": "TPESampler", "seed": args.sampler_seed, "n_startup_trials": 4, "constant_liar": True}, "seed": args.seed,
        "precision_policy": "float32", "gradient_clip_norm": 1.0, "checkpoint_selection_rule": "final checkpoint after the required single epoch",
    }
    write_json(args.output / "configuration.json", configuration)
    results = [tune_model(args, name, dataset, args.output) for name in args.models]
    write_json(args.output / "summary.json", {"configuration": "configuration.json", "models": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
