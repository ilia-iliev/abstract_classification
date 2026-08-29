"""Build immutable benchmark manifests without normalizing the entire snapshot."""

import argparse
import hashlib
import heapq
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from classifier.preprocessing import prepare_abstract
from scripts.data import LABELS, broad_labels, records

SPLIT_SIZES = {"training": 100_000, "validation": 20_000, "benchmark": 20_000, "holdout": 20_000}
MINIMUM_TRAINING_POSITIVES = 8_000
OVERFETCH_RATIO = 0.05
MANIFEST_VERSION = 2


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_rank(seed, value, purpose):
    return hashlib.sha256(f"{seed}\0{purpose}\0{value}".encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def candidate_count(size):
    return int(size * (1 + OVERFETCH_RATIO))


def retain(heap, limit, rank, row):
    """Keep the lowest deterministic ranks using a bounded max heap."""
    item = (-int(rank, 16), row["id"], row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def ordered(heap):
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def scan_candidates(snapshot, seed):
    evaluation = []
    training = []
    by_label = {label: [] for label in LABELS}
    eligible = 0
    for record in records(snapshot):
        labels = broad_labels(record.get("categories", ""))
        abstract = record.get("abstract", "").strip()
        if not labels or not abstract:
            continue
        eligible += 1
        row = {"id": str(record["id"]), "abstract": abstract, "labels": [int(label in labels) for label in LABELS]}
        retain(evaluation, candidate_count(sum(SPLIT_SIZES[name] for name in ("validation", "benchmark", "holdout"))), deterministic_rank(seed, row["id"], "evaluation"), row)
        retain(training, candidate_count(SPLIT_SIZES["training"]), deterministic_rank(seed, row["id"], "training"), row)
        for label in labels:
            retain(by_label[label], candidate_count(MINIMUM_TRAINING_POSITIVES), deterministic_rank(seed, row["id"], f"training-{label}"), row)
    return eligible, ordered(evaluation), ordered(training), {label: ordered(heap) for label, heap in by_label.items()}


def normalize_candidates(rows):
    """Normalize and deduplicate only the 5%-overfetched candidates."""
    groups = {}
    for row in rows:
        normalized = prepare_abstract(row["abstract"])
        if not normalized:
            continue
        digest = content_hash(normalized.casefold())
        labels = tuple(row["labels"])
        group = groups.setdefault(digest, {"labels": labels, "rows": []})
        if group["labels"] != labels:
            group["conflicted"] = True
        group["rows"].append(row)
    result = {}
    for digest, group in groups.items():
        if group.get("conflicted"):
            continue
        # The lowest ID is stable when a duplicate candidate appears more than once.
        row = min(group["rows"], key=lambda value: value["id"])
        value = {"id": row["id"], "content_hash": digest, "labels": list(group["labels"])}
        result.update({candidate["id"]: value for candidate in group["rows"]})
    return result


def select_splits(seed, evaluation_rows, training_rows, label_rows):
    candidates = {row["id"]: row for row in evaluation_rows + training_rows}
    for rows in label_rows.values():
        candidates.update({row["id"]: row for row in rows})
    normalized = normalize_candidates(candidates.values())
    by_id = normalized
    used_hashes = set()

    evaluation = []
    for row in evaluation_rows:
        value = by_id.get(row["id"])
        if value is not None and value["content_hash"] not in used_hashes:
            evaluation.append(value)
            used_hashes.add(value["content_hash"])
    evaluation_total = sum(SPLIT_SIZES[name] for name in ("validation", "benchmark", "holdout"))
    if len(evaluation) < evaluation_total:
        raise ValueError("5% evaluation overfetch was insufficient after candidate deduplication")
    splits = {}
    offset = 0
    for name in ("validation", "benchmark", "holdout"):
        splits[name] = evaluation[offset:offset + SPLIT_SIZES[name]]
        offset += SPLIT_SIZES[name]

    selected, selected_hashes = [], set(used_hashes)
    for label_index, label in enumerate(LABELS):
        positives = 0
        for row in label_rows[label]:
            value = by_id.get(row["id"])
            if value is None or not value["labels"][label_index] or value["content_hash"] in selected_hashes:
                continue
            selected.append(value)
            selected_hashes.add(value["content_hash"])
            positives += 1
            if positives == MINIMUM_TRAINING_POSITIVES:
                break
        if positives < MINIMUM_TRAINING_POSITIVES:
            raise ValueError(f"5% training overfetch was insufficient for {label}")
    for row in training_rows:
        if len(selected) == SPLIT_SIZES["training"]:
            break
        value = by_id.get(row["id"])
        if value is not None and value["content_hash"] not in selected_hashes:
            selected.append(value)
            selected_hashes.add(value["content_hash"])
    if len(selected) != SPLIT_SIZES["training"]:
        raise ValueError("5% training overfetch was insufficient after candidate deduplication")
    splits["training"] = sorted(selected, key=lambda row: deterministic_rank(seed, row["id"], "training-order"))
    return splits


def label_distribution(rows):
    counts = Counter()
    for row in rows:
        counts.update(label for label, value in zip(LABELS, row["labels"]) if value)
    return dict(counts)


def write_manifest(path, rows):
    digest = hashlib.sha256()
    with Path(path).open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            target.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def build_dataset(snapshot, output, snapshot_date, seed=42, snapshot_url=None):
    snapshot, output = Path(snapshot), Path(output)
    snapshot_digest = file_hash(snapshot)
    configuration = {"snapshot_sha256": snapshot_digest, "snapshot_date": snapshot_date, "split_seed": seed}
    metadata_path = output / "dataset.json"
    if output.exists():
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(existing.get(key) == value for key, value in configuration.items()):
                return existing
        raise FileExistsError(f"Refusing to modify immutable dataset directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        eligible, evaluation_rows, training_rows, label_rows = scan_candidates(snapshot, seed)
        splits = select_splits(seed, evaluation_rows, training_rows, label_rows)
        manifest_hashes = {name: write_manifest(staging / f"{name}.jsonl", rows) for name, rows in splits.items()}
        metadata = {
            "manifest_version": MANIFEST_VERSION,
            "dataset_id": f"arxiv-benchmark-{snapshot_digest[:16]}",
            "snapshot": {"path": str(snapshot), "url": snapshot_url, "sha256": snapshot_digest, "date": snapshot_date},
            **configuration,
            "labels": LABELS,
            "normalization": "classifier.preprocessing.prepare_abstract",
            "selection": {"candidate_overfetch_ratio": OVERFETCH_RATIO, "deduplication": "normalized candidate abstracts only", "training": f"tag-aware: {MINIMUM_TRAINING_POSITIVES:,} positives per label, then natural fill"},
            "source_eligible_records": eligible,
            "splits": {name: {"records": len(rows), "label_distribution": label_distribution(rows), "manifest_sha256": manifest_hashes[name]} for name, rows in splits.items()},
        }
        (staging / "dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(output)
        return metadata
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Build immutable manifests from 5%-overfetched random candidates.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--download-url")
    parser.add_argument("--download-to", type=Path)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.download_url:
        if not args.download_to:
            parser.error("--download-to is required with --download-url")
        download_snapshot(args.download_url, args.download_to)
        snapshot = args.download_to
    else:
        snapshot = args.snapshot
    metadata = build_dataset(snapshot, args.output, args.snapshot_date, args.seed, args.download_url)
    print(json.dumps({"dataset_id": metadata["dataset_id"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
