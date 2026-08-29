"""Evaluate the prompted Qwen model on the frozen final holdout."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from scripts.data import LABELS, load_manifest_examples
from scripts.hashing import sha256
from scripts.llm_prompt_experiment import MODEL, PROMPT_TEMPLATE, classify

SNAPSHOT = Path("data/arxiv-metadata-oai-snapshot.json")
MANIFESTS = Path("artifacts/frozen-experiment/manifests")
RESULTS = Path("logs/qwen-llm-final-holdout-results.jsonl")
SUMMARY = Path("logs/qwen-llm-final-holdout-summary.json")
WORKERS = 8
EXPECTED_RECORDS = 20_000


def validate_inputs():
    dataset = json.loads((MANIFESTS / "dataset.json").read_text(encoding="utf-8"))
    holdout = dataset["splits"]["holdout"]
    if sha256(SNAPSHOT) != dataset["snapshot_sha256"]:
        raise ValueError("Snapshot does not match the frozen holdout")
    if sha256(MANIFESTS / "holdout.jsonl") != holdout["manifest_sha256"]:
        raise ValueError("Frozen holdout manifest hash mismatch")
    if holdout["records"] != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} holdout records")
    return dataset


def validate_response(response):
    if set(response) != {"reasoning", "label"}:
        raise ValueError(f"Unexpected response fields: {sorted(response)}")
    if not isinstance(response["reasoning"], str) or not response["reasoning"].strip():
        raise ValueError("Reasoning must be a nonempty string")
    if response["label"] not in LABELS:
        raise ValueError(f"Unknown label: {response['label']!r}")


def classify_record(row):
    _, response, completion = classify(row["text"])
    validate_response(response)
    expected = [label for label, value in zip(LABELS, row["labels"]) if value]
    return {
        "id": row["id"],
        "expected_labels": expected,
        "response": response,
        "matched": response["label"] in expected,
        "usage": completion.get("usage"),
    }


def existing_run():
    if not RESULTS.exists():
        return None, set()
    with RESULTS.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    metadata = rows[0]
    if metadata.get("type") != "metadata" or metadata.get("split") != "holdout":
        raise ValueError(f"Refusing to resume incompatible results: {RESULTS}")
    return metadata, {row["id"] for row in rows[1:]}


def write_metadata(dataset, started_at):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata",
        "model": MODEL,
        "endpoint": "http://127.0.0.1:8081/v1/chat/completions",
        "started_at": started_at,
        "prompt_template": PROMPT_TEMPLATE,
        "labels": LABELS,
        "dataset_id": dataset["dataset_id"],
        "snapshot_sha256": dataset["snapshot_sha256"],
        "manifest_sha256": dataset["splits"]["holdout"]["manifest_sha256"],
        "split": "holdout",
        "metric": "top-1 prediction matches any expected broad label",
        "input": "frozen normalized abstract used by the fine-tuned models",
    }
    RESULTS.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return metadata


def summarize(metadata):
    with RESULTS.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if '"type": "metadata"' not in line]
    matched = sum(row["matched"] for row in rows)
    summary = {
        "model": MODEL,
        "dataset_id": metadata["dataset_id"],
        "split": "holdout",
        "metric": metadata["metric"],
        "started_at": metadata["started_at"],
        "finished_at": datetime.now(UTC).isoformat(),
        "records": len(rows),
        "correct": matched,
        "top1_any_expected_label_accuracy": matched / len(rows),
        "results_path": str(RESULTS),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    dataset = validate_inputs()
    rows = load_manifest_examples(SNAPSHOT, MANIFESTS, "holdout")
    metadata, completed = existing_run()
    if metadata is None:
        metadata = write_metadata(dataset, datetime.now(UTC).isoformat())
    pending = [row for row in rows if row["id"] not in completed]
    with RESULTS.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for index, result in enumerate(executor.map(classify_record, pending), len(completed) + 1):
                output.write(json.dumps(result) + "\n")
                output.flush()
                if index % 100 == 0 or index == len(rows):
                    print(f"Completed {index}/{len(rows)}.", flush=True)
    print(json.dumps(summarize(metadata), indent=2))


if __name__ == "__main__":
    main()
