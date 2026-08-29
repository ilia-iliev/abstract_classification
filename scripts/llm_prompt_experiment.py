import json
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.data import LABELS, load_partition_records

DATA_PATH = Path("data/arxiv-metadata-oai-snapshot.json")
ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "Qwen3.8-27B"
RESULT_PATH = Path("logs/llm-prompt-experiment.json")
PROMPT_TEMPLATE = """You are classifying academic abstracts. There will be a prompt below containing one abstract.
Choose exactly one best-fitting label from: biology, chemistry, computer_science, physics, social_sciences.
Return only a JSON object with exactly these fields: \"reasoning\" (one sentence of at most 20 words) and \"label\" (one of the listed labels).

<abstract>
{prompt}
</abstract>"""


def select_examples():
    records = load_partition_records(DATA_PATH, "test", limit=20_000)
    selected = {}
    for record in records:
        expected = [label for label, value in zip(LABELS, record["labels"]) if value]
        if len(expected) == 1 and expected[0] not in selected:
            selected[expected[0]] = record
        if len(selected) == len(LABELS):
            return [selected[label] for label in LABELS]
    raise RuntimeError("Could not find a single-label test example for every label.")


def classify(abstract, max_tokens=256):
    prompt = PROMPT_TEMPLATE.format(prompt=abstract)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "abstract_classification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "label": {"type": "string", "enum": LABELS},
                    },
                    "required": ["reasoning", "label"],
                    "additionalProperties": False,
                },
            },
        },
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        completion = json.load(response)
    content = completion["choices"][0]["message"]["content"]
    try:
        response = json.loads(content)
    except (JSONDecodeError, TypeError):
        if max_tokens == 256:
            return classify(abstract, max_tokens=512)
        raise
    return prompt, response, completion


def main():
    results = []
    for record in select_examples():
        expected = [label for label, value in zip(LABELS, record["labels"]) if value][0]
        prompt, prediction, completion = classify(record["abstract"])
        results.append(
            {
                "id": record["id"],
                "expected_label": expected,
                "prompt": prompt,
                "response": prediction,
                "matched": prediction.get("label") == expected,
                "usage": completion.get("usage"),
            }
        )
    result = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "run_at": datetime.now(UTC).isoformat(),
        "prompt_template": PROMPT_TEMPLATE,
        "labels": LABELS,
        "test_partition": "single-label records from the repository's deterministic test split",
        "results": results,
        "accuracy": sum(item["matched"] for item in results) / len(results),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(results)} results to {RESULT_PATH} (accuracy: {result['accuracy']:.0%}).")


if __name__ == "__main__":
    main()
