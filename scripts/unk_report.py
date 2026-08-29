import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from classifier.preprocessing import prepare_abstract
from scripts.tokenizer_eda import MODEL_NAME, sample_records

_MATH_DELIMITER = re.compile(
    r"(?<!\\)\$|\\(?:\(|\[)|\\begin\{(?:equation|align|math|displaymath)"
)


def context(text, start, end, width=50):
    return text[max(0, start - width):min(len(text), end + width)].replace("\n", " ")


def write_report(
    output, sample_size, unknowns, documents, documents_by_value, examples, unknown_records
):
    math_records = [
        record for record in unknown_records.values()
        if _MATH_DELIMITER.search(record["abstract"])
    ]
    non_math_records = [
        record for record in unknown_records.values()
        if not _MATH_DELIMITER.search(record["abstract"])
    ]
    lines = [
        "# TeX-normalization `[UNK]` report",
        "",
        f"Sampled records: **{sample_size:,}**",
        f"Records containing `[UNK]`: **{documents:,}**",
        f"`[UNK]` tokens: **{sum(unknowns.values()):,}**",
        "",
        "## Math-delimiter check",
        "",
        "A record is counted as math-marked when its source abstract contains `$`, "
        "`\\\\(`, `\\\\[`, or a standard TeX math environment.",
        "",
        f"- Math-marked records: **{len(math_records):,}**",
        f"- No detected math delimiter: **{len(non_math_records):,}**",
        "",
        "## Records without a detected math delimiter",
        "",
    ]
    for record in non_math_records:
        excerpt = record["abstract"].replace("\n", " ")[:500]
        lines.extend([
            f"### `{record['id']}` — {record['title'].replace(chr(10), ' ')}",
            "",
            f"`{excerpt}`",
            "",
        ])
    lines.extend([
        "## Unknown source spans",
        "",
        "| Span after TeX normalization | Tokens | Records |",
        "|---|---:|---:|",
    ])
    for value, count in unknowns.most_common():
        rendered = value.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{rendered}` | {count:,} | {len(documents_by_value[value]):,} |")

    lines.extend(["", "## Examples", ""])
    for value, entries in examples.items():
        lines.append(f"### `{value}`")
        for entry in entries[:3]:
            lines.extend([
                f"- `{entry['id']}` — {entry['title'].replace(chr(10), ' ')}",
                f"  - `{entry['context']}`",
            ])
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("logs/unk-report.md"))
    args = parser.parse_args()

    records, _ = sample_records(args.dataset, args.sample_size, args.seed)
    texts = [prepare_abstract(record["abstract"]) for record in records]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    unknowns = Counter()
    examples = defaultdict(list)
    documents_by_value = defaultdict(set)
    unknown_documents = set()
    unknown_records = {}

    for offset in range(0, len(texts), 128):
        batch = texts[offset:offset + 128]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        for local_index, (ids, offsets) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
            record = records[offset + local_index]
            for token, (start, end) in zip(ids, offsets):
                if token != tokenizer.unk_token_id or start == end:
                    continue
                value = batch[local_index][start:end]
                unknowns[value] += 1
                unknown_documents.add(record["id"])
                unknown_records[record["id"]] = record
                documents_by_value[value].add(record["id"])
                if len(examples[value]) < 3:
                    examples[value].append({
                        "id": record["id"],
                        "title": record["title"],
                        "context": context(batch[local_index], start, end),
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        args.output,
        len(records),
        unknowns,
        len(unknown_documents),
        documents_by_value,
        examples,
        unknown_records,
    )


if __name__ == "__main__":
    main()
