import argparse
import json
import random

import numpy as np
from transformers import AutoTokenizer

from classifier.preprocessing import (
    FORMULA_TOKEN,
    MAX_CONTEXT_LENGTH,
    PREPROCESSING_VERSION,
    prepare_abstract,
    register_formula_token,
)
from scripts.data import broad_labels, records

MODEL_NAMES = (
    "google-bert/bert-base-uncased",
    "answerdotai/ModernBERT-base",
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-0.6B",
)


def sample_records(path, size, seed):
    rng = random.Random(seed)
    sample = []
    eligible = 0
    for record in records(path):
        abstract = record.get("abstract", "").strip()
        if not abstract or not broad_labels(record.get("categories", "")):
            continue
        eligible += 1
        item = {
            "id": record["id"],
            "title": record.get("title", "").strip(),
            "abstract": abstract,
        }
        if len(sample) < size:
            sample.append(item)
        else:
            position = rng.randrange(eligible)
            if position < size:
                sample[position] = item
    return sample, eligible


def percentile(values, value):
    return int(np.percentile(values, value))


def analyze(tokenizer, examples, texts, max_lengths, longest_count):
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    lengths = np.asarray([len(ids) for ids in encoded["input_ids"]])
    unknown_id = tokenizer.unk_token_id
    unknown_counts = np.asarray([
        sum(token == unknown_id for token in ids) for ids in encoded["input_ids"]
    ])
    unknown_indices = np.flatnonzero(unknown_counts)
    longest_indices = np.argsort(lengths)[-longest_count:][::-1]
    return {
        "tokens": {
            "min": int(lengths.min()),
            "p50": percentile(lengths, 50),
            "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 2),
        },
        "truncation": {
            str(max_length): {
                "records": int(np.sum(lengths > max_length)),
                "rate": round(float(np.mean(lengths > max_length)), 5),
                "mean_tokens_discarded": round(
                    float(np.maximum(lengths - max_length, 0).mean()), 2
                ),
            }
            for max_length in max_lengths
        },
        "unknowns": {
            "token": tokenizer.unk_token,
            "records": int(len(unknown_indices)),
            "tokens": int(unknown_counts.sum()),
            "examples": [
                {
                    "id": examples[index]["id"],
                    "title": examples[index]["title"],
                    "unknown_tokens": int(unknown_counts[index]),
                }
                for index in unknown_indices[:5]
            ],
        },
        "longest_examples": [
            {
                "id": examples[index]["id"],
                "title": examples[index]["title"],
                "characters": len(texts[index]),
                "whitespace_words": len(texts[index].split()),
                "tokens": int(lengths[index]),
                "tokens_per_word": round(
                    float(lengths[index]) / max(len(texts[index].split()), 1), 2
                ),
                "start": texts[index][:700],
                "end": texts[index][-700:],
            }
            for index in longest_indices
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longest-count", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    args = parser.parse_args()

    examples, eligible = sample_records(args.dataset, args.sample_size, args.seed)
    texts = [prepare_abstract(example["abstract"]) for example in examples]
    reports = []
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        formula_token_id = register_formula_token(tokenizer)
        reports.append({
            "model": model_name,
            "formula_token": FORMULA_TOKEN,
            "formula_token_id": formula_token_id,
            "token_counts": analyze(
                tokenizer, examples, texts, [MAX_CONTEXT_LENGTH], args.longest_count
            ),
        })
    result = {
        "preprocessing": PREPROCESSING_VERSION,
        "max_input_length": MAX_CONTEXT_LENGTH,
        "eligible_records": eligible,
        "sample_size": len(examples),
        "tokenizers": reports,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
