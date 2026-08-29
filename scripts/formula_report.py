import argparse
import re
from collections import Counter
from pathlib import Path

from scripts.tokenizer_eda import sample_records

_ENVIRONMENT_PATTERN = re.compile(
    r"\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|math|displaymath)\}"
)
_CURRENCY_PATTERN = re.compile(r"\b(?:usd|dollars?|euros?|pounds?)\b", re.IGNORECASE)
_NUMBER_ONLY_PATTERN = re.compile(r"\s*\d+(?:[.,]\d+)?\s*")


def escaped(text, index):
    slashes = 0
    while index > 0 and text[index - 1] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def next_unescaped(text, marker, start):
    index = text.find(marker, start)
    while index != -1 and escaped(text, index):
        index = text.find(marker, index + 1)
    return index


def formula_spans(text):
    spans = []
    index = 0
    while index < len(text):
        environment = _ENVIRONMENT_PATTERN.match(text, index)
        if environment:
            name = environment.group(1)
            end_marker = f"\\end{{{name}}}"
            end = text.find(end_marker, environment.end())
            if end != -1:
                spans.append((index, end + len(end_marker), f"environment:{name}"))
                index = end + len(end_marker)
                continue
        if text.startswith("\\(", index):
            end = text.find("\\)", index + 2)
            if end != -1:
                spans.append((index, end + 2, "paren"))
                index = end + 2
                continue
        if text.startswith("\\[", index):
            end = text.find("\\]", index + 2)
            if end != -1:
                spans.append((index, end + 2, "bracket"))
                index = end + 2
                continue
        if text[index] == "$" and not escaped(text, index):
            marker = "$$" if text.startswith("$$", index) else "$"
            end = next_unescaped(text, marker, index + len(marker))
            if end != -1:
                spans.append((index, end + len(marker), "double-dollar" if marker == "$$" else "dollar"))
                index = end + len(marker)
                continue
        index += 1
    return spans


def context(text, start, end, width=60):
    return text[max(0, start - width):min(len(text), end + width)].replace("\n", " ")


def is_suspicious(text, start, end, kind):
    if kind != "dollar":
        return False
    content = text[start + 1:end - 1]
    return bool(_CURRENCY_PATTERN.search(content) or _NUMBER_ONLY_PATTERN.fullmatch(content))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("logs/formula-report.md"))
    args = parser.parse_args()

    examples, _ = sample_records(args.dataset, args.sample_size, args.seed)
    span_types = Counter()
    formulas_per_record = Counter()
    suspicious = []
    samples = []

    for example in examples:
        spans = formula_spans(example["abstract"])
        formulas_per_record[len(spans)] += 1
        for start, end, kind in spans:
            span_types[kind] += 1
            if len(samples) < 50:
                samples.append((example, start, end, kind))
            if is_suspicious(example["abstract"], start, end, kind):
                suspicious.append((example, start, end, kind))

    matched_records = args.sample_size - formulas_per_record[0]
    total_spans = sum(span_types.values())
    lines = [
        "# Formula-delimiter report",
        "",
        f"Sampled records: **{args.sample_size:,}**",
        f"Records containing a matched formula: **{matched_records:,}**",
        f"Matched formula spans: **{total_spans:,}**",
        "",
        "## Delimiter types",
        "",
        "| Type | Spans |",
        "|---|---:|",
    ]
    lines.extend(f"| {kind} | {count:,} |" for kind, count in span_types.most_common())
    lines.extend([
        "",
        "## Potential false positives",
        "",
        "Dollar-delimited spans are flagged only when their content is a number or contains "
        "a currency word. These need manual review.",
        "",
        f"Flagged spans: **{len(suspicious):,}**",
        "",
    ])
    for example, start, end, kind in suspicious:
        lines.extend([
            f"- `{example['id']}` ({kind}): `{context(example['abstract'], start, end)}`",
        ])
    lines.extend(["", "## Sample matched spans", ""])
    for example, start, end, kind in samples:
        lines.extend([
            f"- `{example['id']}` ({kind}): `{context(example['abstract'], start, end)}`",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
