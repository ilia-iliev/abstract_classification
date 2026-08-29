import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.data import LABELS, broad_labels, records


def analyze(path):
    total = 0
    missing_abstract = 0
    unmapped = 0
    raw_categories = Counter()
    broad_counts = Counter()
    years = Counter()
    word_lengths = []
    char_lengths = []
    multilabel = Counter()
    for record in records(path):
        total += 1
        abstract = record.get("abstract", "").strip()
        if not abstract:
            missing_abstract += 1
        else:
            word_lengths.append(len(abstract.split()))
            char_lengths.append(len(abstract))
        raw_categories.update(record.get("categories", "").split())
        labels = broad_labels(record.get("categories", ""))
        broad_counts.update(labels)
        multilabel[len(labels)] += 1
        if not labels:
            unmapped += 1
        update_date = record.get("update_date", "")
        if len(update_date) >= 4:
            years[update_date[:4]] += 1
    return {
        "total": total,
        "missing_abstract": missing_abstract,
        "unmapped": unmapped,
        "raw_categories": raw_categories,
        "broad_counts": broad_counts,
        "years": years,
        "word_lengths": word_lengths,
        "char_lengths": char_lengths,
        "multilabel": multilabel,
    }


def summary(values):
    if not values:
        return "n/a"
    minimum, median, p95, maximum = np.percentile(values, [0, 50, 95, 100])
    mean = np.mean(values)
    return f"min {minimum:.0f}; median {median:.0f}; mean {mean:.1f}; p95 {p95:.0f}; max {maximum:.0f}"


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def render(result, source):
    total = result["total"]
    mapped = total - result["unmapped"]
    label_rows = [(label, result["broad_counts"][label], f"{result['broad_counts'][label] / max(mapped, 1):.2%}") for label in LABELS]
    category_rows = result["raw_categories"].most_common(20)
    year_rows = sorted(result["years"].items())
    multi_rows = sorted(result["multilabel"].items())
    return f"""# Exploratory data analysis

Generated from `{source}` by `uv run python -m scripts.eda {source}`.

## Dataset structure

The Kaggle arXiv snapshot is newline-delimited JSON. Important fields are `id`, `submitter`, `authors`, `title`, `comments`, `journal-ref`, `doi`, `report-no`, `categories`, `license`, `abstract`, `versions`, `update_date`, and `authors_parsed`. `categories` is a space-separated, multi-label arXiv taxonomy; `abstract` is the model input.

- Records: **{total:,}**
- Missing/blank abstracts: **{result['missing_abstract']:,}**
- Records mapped to at least one target: **{mapped:,}**
- Unmapped records (mostly mathematics): **{result['unmapped']:,}**
- Abstract words: {summary(result['word_lengths'])}
- Abstract characters: {summary(result['char_lengths'])}

## Target distribution

Percentages use mapped records as denominator and can sum above 100% because this is multi-label classification.

{table(label_rows, ['Target', 'Records', 'Share of mapped records'])}

Mapping: `q-bio` → biology; `physics.chem-ph` → chemistry; `cs` and `stat.ML` → computer science; arXiv physics archives → physics; `econ`, `q-fin`, and non-ML `stat` → social sciences. Pure mathematics and other unrelated records are excluded from training.

### Number of mapped labels per record

{table(multi_rows, ['Labels', 'Records'])}

## Most frequent original subjects

{table(category_rows, ['Subject', 'Records'])}

## Records by update year

{table(year_rows, ['Year', 'Records'])}

## Modeling implications

- The target is multi-label, so training uses independent sigmoid outputs and binary cross-entropy rather than softmax.
- Class imbalance makes micro/macro precision, recall, and F1 necessary alongside exact-match accuracy.
- Training and inference use a fixed 512-token context.
- Normalized duplicate abstracts are grouped before sampling; groups with conflicting labels are discarded.
- A seeded reservoir sample avoids the chronological bias of taking the first snapshot rows.
- Deterministic ID hashing creates disjoint natural training, validation, and test partitions. A time-based holdout should also be checked for drift.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--output", default="EDA.md")
    args = parser.parse_args()
    Path(args.output).write_text(render(analyze(args.dataset), args.dataset), encoding="utf-8")


if __name__ == "__main__":
    main()
