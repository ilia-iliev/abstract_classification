"""Archive and attest the inputs to final-holdout evaluation."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from classifier.modeling import BACKBONES
from classifier.preprocessing import PREPROCESSING_VERSION
from scripts.artifacts import write_json
from scripts.data import LABELS
from scripts.hashing import sha256

FREEZE_VERSION = 1
REQUIRED_MANIFESTS = ("training", "validation", "holdout")
REQUIRED_ARTIFACT_FILES = (
    "classifier.pt",
    "configuration.json",
    "metadata.json",
    "metrics.json",
    "history.json",
    "runtime.json",
    "validation_predictions.npz",
)
CODE_FILES = {
    "classification_head_and_pooling": Path("classifier/modeling.py"),
    "preprocessing": Path("classifier/preprocessing.py"),
    "data_loading": Path("scripts/data.py"),
    "metrics": Path("scripts/train.py"),
    "holdout_metrics": Path("scripts/evaluate_holdout.py"),
    "throughput_measurement": Path("scripts/measure_throughput.py"),
}



def file_inventory(root):
    """Return stable checksums for all regular files below root."""
    root = Path(root)
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks cannot be frozen: {path}")
        if path.is_file():
            inventory.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    return inventory


def copy_tree(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"Symlinks cannot be frozen: {source}")
    shutil.copytree(source, destination)


def load_dataset(snapshot, manifests):
    snapshot, manifests = Path(snapshot), Path(manifests)
    metadata_path = manifests / "dataset.json"
    if not snapshot.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Snapshot or dataset.json is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("labels") != LABELS:
        raise ValueError("Dataset labels do not match classifier labels")
    if sha256(snapshot) != metadata.get("snapshot_sha256"):
        raise ValueError("Snapshot hash does not match dataset.json")
    for name in REQUIRED_MANIFESTS:
        path = manifests / f"{name}.jsonl"
        expected = metadata.get("splits", {}).get(name, {})
        if not path.is_file() or sha256(path) != expected.get("manifest_sha256"):
            raise ValueError(f"Manifest hash mismatch: {path}")
        if sum(1 for _ in path.open(encoding="utf-8")) != expected.get("records"):
            raise ValueError(f"Manifest count mismatch: {path}")
    return metadata


def artifact_directory(root, model_name):
    matches = [path for path in Path(root).iterdir() if path.is_dir() and path.name.endswith(model_name.replace("/", "--"))]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected artifact for {model_name}; found {len(matches)}")
    return matches[0]


def validate_artifact(directory, model_name, dataset):
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete final artifact {directory}: {', '.join(missing)}")
    configuration = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "stage": "selected_configuration_training",
        "backend": "pytorch",
        "base_model": model_name,
        "pooling": BACKBONES[model_name].pooling,
        "labels": LABELS,
        "preprocessing": PREPROCESSING_VERSION,
    }
    invalid = [key for key, value in required.items() if configuration.get(key) != value]
    if invalid:
        raise ValueError(f"Invalid final artifact {directory}: {', '.join(invalid)}")
    if metadata.get("threshold") != configuration.get("threshold"):
        raise ValueError(f"Metadata thresholds differ from configuration: {directory}")
    thresholds = configuration.get("threshold")
    if not isinstance(thresholds, list) or len(thresholds) != len(LABELS):
        raise ValueError(f"Invalid per-label thresholds: {directory}")
    if configuration.get("data", {}).get("dataset_id") != dataset.get("dataset_id"):
        raise ValueError(f"Artifact has another dataset identity: {directory}")
    data = configuration.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Artifact has no data configuration: {directory}")
    manifests = data.get("manifests", {})
    for name in ("training", "validation"):
        if manifests.get(name) != dataset["splits"][name]["manifest_sha256"]:
            raise ValueError(f"Artifact has another {name} manifest: {directory}")
    return configuration


def freeze(snapshot, manifests, selected, output, throughput_protocol):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen experiment: {output}")
    dataset = load_dataset(snapshot, manifests)
    selected = Path(selected)
    if not (selected / "summary.json").is_file():
        raise FileNotFoundError(f"Missing selected-training summary: {selected / 'summary.json'}")
    protocol = Path(throughput_protocol)
    if not protocol.is_file():
        raise FileNotFoundError(protocol)

    root = Path(__file__).resolve().parents[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        frozen_models = []
        for model_name in BACKBONES:
            source = artifact_directory(selected, model_name)
            configuration = validate_artifact(source, model_name, dataset)
            destination = staging / "models" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_tree(source, destination)
            frozen_models.append({
                "model": model_name,
                "artifact": destination.relative_to(staging).as_posix(),
                "configuration_sha256": sha256(destination / "configuration.json"),
                "hyperparameters": configuration["hyperparameters"],
                "thresholds": configuration["threshold"],
                "files": file_inventory(destination),
            })

        frozen_manifests = staging / "manifests"
        frozen_manifests.mkdir()
        for name in REQUIRED_MANIFESTS:
            shutil.copy2(Path(manifests) / f"{name}.jsonl", frozen_manifests / f"{name}.jsonl")
        shutil.copy2(Path(manifests) / "dataset.json", frozen_manifests / "dataset.json")
        code = staging / "code"
        code.mkdir()
        code_records = {}
        for purpose, relative in CODE_FILES.items():
            source = root / relative
            destination = code / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            code_records[purpose] = {"path": relative.as_posix(), "sha256": sha256(destination)}
        shutil.copy2(protocol, staging / "throughput_protocol.md")

        manifest = {
            "freeze_version": FREEZE_VERSION,
            "status": "frozen_before_holdout_evaluation",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "snapshot_sha256": dataset["snapshot_sha256"],
                "snapshot_date": dataset["snapshot_date"],
                "split_seed": dataset["split_seed"],
                "manifests": {name: dataset["splits"][name] for name in REQUIRED_MANIFESTS},
                "archive": "manifests",
            },
            "models": frozen_models,
            "preprocessing": {
                "version": PREPROCESSING_VERSION,
                "code": code_records["preprocessing"],
            },
            "classification_head_and_pooling": code_records["classification_head_and_pooling"],
            "data_loading_code": code_records["data_loading"],
            "metric_code": code_records["metrics"],
            "holdout_metric_code": code_records["holdout_metrics"],
            "throughput_measurement_code": code_records["throughput_measurement"],
            "throughput_protocol": {
                "path": "throughput_protocol.md",
                "sha256": sha256(staging / "throughput_protocol.md"),
            },
            "selected_training_summary_sha256": sha256(selected / "summary.json"),
            "evaluation_rule": "Final-holdout evaluation must use only these archived artifacts; retraining, threshold changes, and configuration changes are prohibited.",
        }
        write_json(staging / "freeze.json", manifest)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Freeze final artifacts and evaluation inputs before final-holdout evaluation.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("selected", type=Path, help="artifacts/selected-configurations")
    parser.add_argument("--output", type=Path, default=Path("artifacts/frozen-experiment"))
    parser.add_argument("--throughput-protocol", type=Path, default=Path("THROUGHPUT.md"))
    args = parser.parse_args()
    manifest = freeze(args.snapshot, args.manifests, args.selected, args.output, args.throughput_protocol)
    print(json.dumps({"output": str(args.output), "models": len(manifest["models"])}, sort_keys=True))


if __name__ == "__main__":
    main()
