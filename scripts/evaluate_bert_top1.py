import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from classifier.modeling import MultilabelClassifier, load_tokenizer
from classifier.preprocessing import MAX_CONTEXT_LENGTH, prepare_abstract
from scripts.data import LABELS, load_partition_records

DATA_PATH = Path("data/arxiv-metadata-oai-snapshot.json")
MODEL_PATH = Path("artifacts/model")
RESULT_PATH = Path("logs/bert-top1-results.jsonl")
SUMMARY_PATH = Path("logs/bert-top1-summary.json")
BATCH_SIZE = 1
TEST_LIMIT = 20_000


def expected_labels(record):
    return [label for label, value in zip(LABELS, record["labels"]) if value]


def main():
    records = load_partition_records(DATA_PATH, "test", limit=TEST_LIMIT)
    if len(records) != TEST_LIMIT:
        raise RuntimeError(f"Expected {TEST_LIMIT} test records, found {len(records)}.")

    metadata = json.loads((MODEL_PATH / "metadata.json").read_text())
    tokenizer = load_tokenizer(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultilabelClassifier.load(MODEL_PATH, len(LABELS), map_location=device).to(device).eval()
    started_at = datetime.now(UTC).isoformat()
    matched = 0

    with RESULT_PATH.open("w", encoding="utf-8") as output, torch.inference_mode():
        output.write(json.dumps({
            "type": "metadata",
            "model": metadata["base_model"],
            "artifact": str(MODEL_PATH),
            "labels": LABELS,
            "test_partition": "the 20,000-record deterministic test reservoir used by evaluation",
        }) + "\n")
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start:start + BATCH_SIZE]
            texts = [prepare_abstract(record["abstract"]) for record in batch]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH, return_tensors="pt")
            scores = torch.sigmoid(model(**{key: value.to(device) for key, value in encoded.items()})).cpu().numpy()
            for record, score in zip(batch, scores):
                prediction = LABELS[int(np.argmax(score))]
                expected = expected_labels(record)
                result = {"id": record["id"], "expected_labels": expected, "prediction": prediction, "matched": prediction in expected}
                matched += result["matched"]
                output.write(json.dumps(result) + "\n")
            output.flush()
            completed = start + len(batch)
            if completed % 100 == 0 or completed == len(records):
                print(f"Completed {completed}/{len(records)}.", flush=True)

    summary = {
        "model": metadata["base_model"],
        "records": len(records),
        "top_1_any_expected_label_matches": matched,
        "top_1_any_expected_label_accuracy": matched / len(records),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "results_path": str(RESULT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
