"""Evaluate the frozen-backbone probes on the frozen final holdout."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from classifier.modeling import BACKBONES
from scripts.artifacts import write_json
from scripts.evaluate_holdout import predict, quality_metrics, split_rows
from scripts.hashing import sha256

PROBES = {
    "google-bert/bert-base-uncased": Path("artifacts/frozen-probes-bert/01-bert-base-uncased"),
    "answerdotai/ModernBERT-base": Path("artifacts/frozen-probes-modernbert/01-ModernBERT-base"),
    "google/embeddinggemma-300m": Path("artifacts/frozen-probes-embeddinggemma/01-embeddinggemma-300m"),
    "Qwen/Qwen3-Embedding-0.6B": Path("artifacts/frozen-probes-qwen3/01-Qwen3-Embedding-0.6B"),
}
PROBE_REPORTS = tuple(path.parent / "report.json" for path in PROBES.values())


def validate_probes(dataset):
    if set(PROBES) != set(BACKBONES):
        raise ValueError("Probe mapping does not contain exactly the four candidate backbones")
    expected_splits = dataset["splits"]
    seen_reports = set()
    for report_path in PROBE_REPORTS:
        if report_path in seen_reports:
            continue
        seen_reports.add(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("stage") != "frozen_representation_probe":
            raise ValueError(f"Not a frozen-probe report: {report_path}")
        if report.get("dataset_id") != dataset["dataset_id"]:
            raise ValueError(f"Probe dataset mismatch: {report_path}")
        if report.get("snapshot_sha256") != dataset["snapshot_sha256"]:
            raise ValueError(f"Probe snapshot mismatch: {report_path}")
        for split in ("training", "validation"):
            if report["splits"][split]["manifest_sha256"] != expected_splits[split]["manifest_sha256"]:
                raise ValueError(f"Probe {split} manifest mismatch: {report_path}")
    for model_name, directory in PROBES.items():
        configuration = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
        if configuration.get("base_model") != model_name:
            raise ValueError(f"Probe model mismatch: {directory}")
        if configuration.get("stage") != "frozen_representation_probe" or configuration.get("backbone_frozen") is not True:
            raise ValueError(f"Artifact is not a frozen-backbone probe: {directory}")


def validate_experiment_reference(snapshot, frozen_experiment, final_report):
    freeze = json.loads((frozen_experiment / "freeze.json").read_text(encoding="utf-8"))
    valid_statuses = {"frozen_before_benchmark_evaluation", "frozen_before_holdout_evaluation"}
    if freeze.get("status") not in valid_statuses:
        raise ValueError("Experiment was not frozen before evaluation")
    if sha256(snapshot) != freeze.get("dataset", {}).get("snapshot_sha256"):
        raise ValueError("Snapshot does not match frozen experiment")
    report = json.loads(Path(final_report).read_text(encoding="utf-8"))
    if report.get("split") != "holdout" or report.get("freeze_sha256") != sha256(frozen_experiment / "freeze.json"):
        raise ValueError("Final report does not attest this frozen experiment")


def full_model_metrics(final_report, freeze_digest, dataset_id, records):
    report = json.loads(Path(final_report).read_text(encoding="utf-8"))
    if report.get("split") != "holdout":
        raise ValueError("Full-model comparison is not a holdout report")
    if report.get("freeze_sha256") != freeze_digest:
        raise ValueError("Full-model report belongs to a different frozen experiment")
    if report.get("dataset_id") != dataset_id or report.get("records") != records:
        raise ValueError("Full-model report uses a different holdout")
    return report["models"]


def render_markdown(report):
    lines = [
        "# Frozen probes on the final holdout",
        "",
        "Both frozen-backbone probes and fully fine-tuned models are measured on the same records.",
        "",
        "| Model | Frozen macro F1 | Frozen micro F1 | Full macro F1 | Full micro F1 | Macro difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in PROBES:
        frozen = report["frozen_probes"][model_name]
        full = report["full_fine_tuning"][model_name]
        difference = full["f1_macro"] - frozen["f1_macro"]
        lines.append(
            f"| {model_name} | {frozen['f1_macro']:.4f} | {frozen['f1_micro']:.4f} | "
            f"{full['f1_macro']:.4f} | {full['f1_micro']:.4f} | {difference:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def evaluate(snapshot, frozen_experiment, final_report, output, batch_size, device_name):
    snapshot = Path(snapshot)
    frozen_experiment = Path(frozen_experiment)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite probe holdout report: {output}")

    validate_experiment_reference(snapshot, frozen_experiment, final_report)
    dataset, rows = split_rows(snapshot, frozen_experiment, "holdout")
    validate_probes(dataset)
    labels = np.asarray([row["labels"] for row in rows], dtype=int)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    metrics = {}
    predictions = {}
    for model_name, directory in PROBES.items():
        configuration, logits, probabilities, _ = predict(directory, rows, batch_size, device)
        thresholds = np.asarray(configuration["threshold"], dtype=float)
        metrics[model_name] = quality_metrics(labels, probabilities, thresholds)
        predictions[model_name] = {
            "ids": np.asarray([row["id"] for row in rows]),
            "labels": labels,
            "logits": logits,
            "probabilities": probabilities,
            "predictions": (probabilities >= thresholds).astype(int),
            "thresholds": thresholds,
        }

    freeze_digest = sha256(frozen_experiment / "freeze.json")
    report = {
        "evaluation": "frozen_backbone_probes_on_final_holdout",
        "split": "holdout",
        "dataset_id": dataset["dataset_id"],
        "freeze_sha256": freeze_digest,
        "records": len(rows),
        "metric_note": "Macro and micro F1 use validation-selected per-label thresholds; no holdout tuning.",
        "frozen_probes": metrics,
        "full_fine_tuning": full_model_metrics(final_report, freeze_digest, dataset["dataset_id"], len(rows)),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for model_name, values in predictions.items():
            np.savez_compressed(staging / f"{model_name.replace('/', '--')}.npz", **values)
        write_json(staging / "report.json", report)
        (staging / "report.md").write_text(render_markdown(report), encoding="utf-8")
        staging.replace(output)
    except Exception:
        for path in staging.rglob("*"):
            if path.is_file():
                path.unlink()
        staging.rmdir()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen-backbone probes on the final holdout.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("frozen_experiment", type=Path)
    parser.add_argument("--final-report", type=Path, default=Path("artifacts/final-holdout-evaluation/report.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/frozen-probe-holdout-evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = evaluate(args.snapshot, args.frozen_experiment, args.final_report, args.output, args.batch_size, args.device)
    summary = {
        model: {"macro_f1": values["f1_macro"], "micro_f1": values["f1_micro"]}
        for model, values in report["frozen_probes"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
