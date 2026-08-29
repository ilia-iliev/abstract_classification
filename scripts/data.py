import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from classifier.labels import LABELS
from classifier.preprocessing import (
    PRIMARY_LABEL_LOSS_WEIGHT,
    SECONDARY_LABEL_LOSS_WEIGHT,
    prepare_abstract,
)

PHYSICS_PREFIXES = ("astro-ph", "cond-mat", "gr-qc", "hep-", "math-ph", "nlin", "nucl-", "physics", "quant-ph")


def broad_labels(categories):
    categories = categories.split() if isinstance(categories, str) else categories
    labels = set()
    for category in categories:
        if category.startswith("q-bio"):
            labels.add("biology")
        if category == "physics.chem-ph":
            labels.add("chemistry")
        elif category.startswith(PHYSICS_PREFIXES):
            labels.add("physics")
        if category.startswith("cs.") or category == "stat.ML":
            labels.add("computer_science")
        if category.startswith(("econ.", "q-fin.", "stat.")) and category != "stat.ML":
            labels.add("social_sciences")
    return labels


def records(path):
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error


def normalized_abstract_hash(abstract):
    """Return the stable SHA-256 identity of the model input text."""
    normalized = " ".join(prepare_abstract(abstract).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _deduplication_key(abstract):
    # Deduplicate the normalized text, not the source TeX. This prevents the
    # same model input from landing in different splits through TeX formatting.
    return bytes.fromhex(normalized_abstract_hash(abstract))


def deduplicated_records(path):
    """Yield one record per abstract and discard duplicate groups with conflicting labels."""
    labels_by_abstract = {}
    for record in records(path):
        labels = broad_labels(record.get("categories", ""))
        abstract = record.get("abstract", "").strip()
        if not labels or not abstract:
            continue
        key = _deduplication_key(abstract)
        values = tuple(int(label in labels) for label in LABELS)
        previous = labels_by_abstract.setdefault(key, values)
        if previous != values:
            labels_by_abstract[key] = None

    yielded = set()
    for record in records(path):
        labels = broad_labels(record.get("categories", ""))
        abstract = record.get("abstract", "").strip()
        if not labels or not abstract:
            continue
        key = _deduplication_key(abstract)
        if key in yielded or labels_by_abstract[key] is None:
            continue
        yielded.add(key)
        yield record, labels, abstract


def _clean_examples(examples):
    return [(prepare_abstract(abstract), values) for _, abstract, values in examples]


def _primary_values(categories):
    primary = categories.split(maxsplit=1)[0]
    primary_labels = broad_labels((primary,))
    return [int(label in primary_labels) for label in LABELS]


def load_manifest_examples(snapshot, manifest_directory, split):
    """Load one immutable manifest split and verify it against its snapshot."""
    manifest_path = Path(manifest_directory) / f"{split}.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    expected = {row["id"]: row for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError(f"Duplicate record IDs in {manifest_path}")

    examples = {}
    for record in records(snapshot):
        record_id = str(record.get("id", ""))
        row = expected.get(record_id)
        if row is None:
            continue
        abstract = record.get("abstract", "").strip()
        normalized = prepare_abstract(abstract)
        digest = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
        labels = [int(label in broad_labels(record.get("categories", ""))) for label in LABELS]
        if digest != row["content_hash"] or labels != row["labels"]:
            # arXiv snapshots can contain multiple versions of one ID. Keep
            # scanning until its manifest-attested version is found.
            continue
        primary = _primary_values(record.get("categories", ""))
        examples[record_id] = {
            "id": record_id,
            "text": normalized,
            "labels": labels,
            "weighted_labels": build_weighted_labels([labels], [primary])[0].tolist(),
        }
    missing = set(expected) - set(examples)
    if missing:
        raise ValueError(f"Snapshot is missing {len(missing)} records matching {manifest_path}")
    return [examples[row["id"]] for row in manifest]


def build_weighted_labels(labels, primary_labels):
    """Encode positive labels with their primary or secondary loss weight."""
    labels = np.asarray(labels, dtype=np.float32)
    primary_labels = np.asarray(primary_labels, dtype=np.float32)
    if labels.shape != primary_labels.shape:
        raise ValueError("Labels and primary labels must have the same shape")
    if np.any((primary_labels > 0) & (labels == 0)):
        raise ValueError("A primary label must also be a mapped label")
    secondary = (labels > 0) & (primary_labels == 0)
    return np.where(
        secondary,
        SECONDARY_LABEL_LOSS_WEIGHT,
        np.where(labels > 0, PRIMARY_LABEL_LOSS_WEIGHT, 0.0),
    ).astype(np.float32)


def _reservoir_add(reservoir, example, seen, limit, rng):
    if len(reservoir) < limit:
        reservoir.append(example)
        return
    position = rng.randrange(seen)
    if position < limit:
        reservoir[position] = example


def _partition(record_id):
    return int.from_bytes(
        hashlib.blake2b(record_id.encode(), digest_size=2).digest(), "big"
    ) % 10


def load_partition_records(path, partition, limit, seed=42):
    """Reproduce a natural split reservoir while retaining source records."""
    if partition not in {"training", "validation", "test"}:
        raise ValueError(f"Unknown partition: {partition}")
    reservoir = []
    seen = 0
    rng = random.Random(f"{seed}-{partition}")
    for record, labels, abstract in deduplicated_records(path):
        bucket = _partition(record["id"])
        split_name = "test" if bucket == 0 else "validation" if bucket == 1 else "training"
        if split_name != partition:
            continue
        seen += 1
        _reservoir_add(
            reservoir,
            {
                "id": record["id"],
                "abstract": abstract,
                "categories": record.get("categories", ""),
                "labels": [int(label in labels) for label in LABELS],
            },
            seen,
            limit,
            rng,
        )
    return reservoir


def load_category_examples(
    path, training_limit, validation_limit=20000, test_limit=20000, seed=42
):
    """Build disjoint natural splits before category-aware training selection."""
    splits = {"training": [], "validation": [], "test": []}
    limits = {
        "training": training_limit,
        "validation": validation_limit,
        "test": test_limit,
    }
    seen = Counter()
    rngs = {name: random.Random(f"{seed}-{name}") for name in splits}
    eligible = 0
    label_counts = Counter()

    for record, labels, abstract in deduplicated_records(path):
        eligible += 1
        bucket = _partition(record["id"])
        split_name = "test" if bucket == 0 else "validation" if bucket == 1 else "training"
        values = [int(label in labels) for label in LABELS]
        example = (record["id"], abstract, values)
        seen[split_name] += 1
        _reservoir_add(
            splits[split_name],
            example,
            seen[split_name],
            limits[split_name],
            rngs[split_name],
        )

    random.Random(seed).shuffle(splits["training"])
    for _, _, values in splits["training"]:
        label_counts.update(label for label, value in zip(LABELS, values) if value)
    return {
        name: _clean_examples(examples)
        for name, examples in splits.items()
    } | {
        "eligible": eligible,
        "training_candidates": seen["training"],
        "training_label_counts": dict(label_counts),
    }


def load_tag_aware_examples(
    path,
    training_limit,
    minimum_per_label,
    validation_limit=20000,
    test_limit=20000,
    seed=42,
):
    """Sample every training tag reservoir from the full training partition."""
    natural = []
    label_reservoirs = {label: [] for label in LABELS}
    validation = []
    test = []
    seen = Counter()
    label_seen = Counter()
    natural_rng = random.Random(f"{seed}-training-natural")
    label_rngs = {label: random.Random(f"{seed}-training-{label}") for label in LABELS}
    validation_rng = random.Random(f"{seed}-validation")
    test_rng = random.Random(f"{seed}-test")
    eligible = 0

    for record, broad, abstract in deduplicated_records(path):
        eligible += 1
        categories = record.get("categories", "")
        values = [int(label in broad) for label in LABELS]
        example = (record["id"], abstract, values)
        bucket = _partition(record["id"])
        if bucket == 0:
            seen["test"] += 1
            _reservoir_add(test, example, seen["test"], test_limit, test_rng)
            continue
        if bucket == 1:
            seen["validation"] += 1
            _reservoir_add(
                validation,
                example,
                seen["validation"],
                validation_limit,
                validation_rng,
            )
            continue

        seen["training"] += 1
        weighted_example = (*example, _primary_values(categories))
        _reservoir_add(
            natural,
            weighted_example,
            seen["training"],
            training_limit,
            natural_rng,
        )
        for label in broad:
            label_seen[label] += 1
            _reservoir_add(
                label_reservoirs[label],
                weighted_example,
                label_seen[label],
                minimum_per_label,
                label_rngs[label],
            )

    candidates_by_id = {
        example[0]: example
        for reservoir in [natural, *label_reservoirs.values()]
        for example in reservoir
    }
    candidates = list(candidates_by_id.values())
    random.Random(seed).shuffle(candidates)
    training = [
        (prepare_abstract(abstract), values)
        for _, abstract, values, _ in candidates
    ]
    primary_values = [values for *_, values in candidates]
    return {
        "training": training,
        "training_primary_values": primary_values,
        "validation": _clean_examples(validation),
        "test": _clean_examples(test),
        "eligible": eligible,
        "training_candidates": seen["training"],
        "sampled_candidates": len(candidates),
        "training_label_counts": dict(label_seen),
    }


def load_uniform_examples(path, per_label, validation_limit=20000, test_limit=20000, seed=42):
    """Build uniform label reservoirs and disjoint natural validation/test sets."""
    label_reservoirs = {label: [] for label in LABELS}
    label_seen = Counter()
    label_rngs = {label: random.Random(f"{seed}-{label}") for label in LABELS}
    validation = []
    test = []
    validation_seen = 0
    test_seen = 0
    validation_rng = random.Random(f"{seed}-validation")
    test_rng = random.Random(f"{seed}-test")
    eligible = 0

    for record, broad, abstract in deduplicated_records(path):
        eligible += 1
        values = [int(label in broad) for label in LABELS]
        example = (record["id"], abstract, values)
        bucket = _partition(record["id"])
        if bucket == 0:
            test_seen += 1
            _reservoir_add(test, example, test_seen, test_limit, test_rng)
            continue
        if bucket == 1:
            validation_seen += 1
            _reservoir_add(
                validation, example, validation_seen, validation_limit, validation_rng
            )
            continue
        for label in broad:
            label_seen[label] += 1
            _reservoir_add(
                label_reservoirs[label],
                example,
                label_seen[label],
                per_label,
                label_rngs[label],
            )

    training_by_id = {
        example[0]: example
        for reservoir in label_reservoirs.values()
        for example in reservoir
    }
    training = list(training_by_id.values())
    random.Random(seed).shuffle(training)
    return {
        "training": _clean_examples(training),
        "validation": _clean_examples(validation),
        "test": _clean_examples(test),
        "eligible": eligible,
        "training_candidates": dict(label_seen),
    }


def load_examples(path, limit, seed=42):
    """Reservoir-sample mapped records to avoid the snapshot's chronological bias."""
    rng = random.Random(seed)
    sample = []
    eligible = 0
    label_counts = Counter()
    for record, labels, abstract in deduplicated_records(path):
        eligible += 1
        example = (abstract, [int(label in labels) for label in LABELS])
        if len(sample) < limit:
            sample.append(example)
        else:
            position = rng.randrange(eligible)
            if position < limit:
                sample[position] = example
    sample = [(prepare_abstract(abstract), values) for abstract, values in sample]
    rng.shuffle(sample)
    for _, values in sample:
        label_counts.update(label for label, value in zip(LABELS, values) if value)
    return sample, eligible, label_counts
