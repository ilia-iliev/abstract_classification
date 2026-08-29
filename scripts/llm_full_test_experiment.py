import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from scripts.data import LABELS, load_partition_records
from scripts.llm_prompt_experiment import PROMPT_TEMPLATE, classify

DATA_PATH = Path("data/arxiv-metadata-oai-snapshot.json")
RESULT_PATH = Path("logs/llm-full-test-results.jsonl")
SUMMARY_PATH = Path("logs/llm-full-test-summary.json")
WORKERS = 8
TEST_LIMIT = 20_000


def expected_labels(record):
    return [label for label, value in zip(LABELS, record["labels"]) if value]


def validate_response(response):
    if set(response) != {"reasoning", "label"}:
        raise ValueError(f"Expected only reasoning and label fields, got {sorted(response)}")
    if not isinstance(response["reasoning"], str) or not response["reasoning"].strip():
        raise ValueError("Reasoning must be a nonempty string.")
    if response["label"] not in LABELS:
        raise ValueError(f"Unknown label: {response['label']!r}")


def completed_ids():
    if not RESULT_PATH.exists():
        return set()
    with RESULT_PATH.open(encoding="utf-8") as results:
        return {
            row["id"]
            for line in results
            if (row := json.loads(line)).get("type") != "metadata"
        }


def classify_record(record):
    _, response, completion = classify(record["abstract"])
    try:
        validate_response(response)
    except ValueError:
        _, response, completion = classify(record["abstract"], max_tokens=512)
        validate_response(response)
    expected = expected_labels(record)
    return {
        "id": record["id"],
        "expected_labels": expected,
        "response": response,
        "matched": response["label"] in expected,
        "usage": completion.get("usage"),
    }


def main():
    records = load_partition_records(DATA_PATH, "test", limit=TEST_LIMIT)
    if len(records) != TEST_LIMIT:
        raise RuntimeError(f"Expected {TEST_LIMIT} test records, found {len(records)}.")

    started_at = datetime.now(UTC).isoformat()
    completed = completed_ids()
    pending = [record for record in records if record["id"] not in completed]
    if not completed:
        with RESULT_PATH.open("w", encoding="utf-8") as output:
            output.write(json.dumps({
                "type": "metadata",
                "model": "Qwen3.8-27B",
                "endpoint": "http://127.0.0.1:8081/v1/chat/completions",
                "started_at": started_at,
                "prompt_template": PROMPT_TEMPLATE,
                "labels": LABELS,
                "test_partition": "the 20,000-record deterministic test reservoir used by evaluation",
            }) + "\n")
    with RESULT_PATH.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for index, result in enumerate(executor.map(classify_record, pending), len(completed) + 1):
                completed.add(result["id"])
                output.write(json.dumps(result) + "\n")
                output.flush()
                if index % 100 == 0:
                    print(f"Completed {index}/{len(records)}.", flush=True)

    matched = 0
    with RESULT_PATH.open(encoding="utf-8") as results:
        for line in results:
            row = json.loads(line)
            if row.get("type") != "metadata":
                matched += row["matched"]
    summary = {
        "model": "Qwen3.8-27B",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "records": len(records),
        "exact_label_matches": matched,
        "match_rate": matched / len(records),
        "results_path": str(RESULT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
