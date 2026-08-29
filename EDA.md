# Exploratory data analysis

Generated from `data/arxiv-metadata-oai-snapshot.json` with `uv run python -m scripts.eda data/arxiv-metadata-oai-snapshot.json`.

## Dataset

The snapshot contains 3,141,764 arXiv records. Abstracts are the model input; `categories` supplies the multi-label targets.

| Measure | Value |
|---|---:|
| Records mapped to a target | 2,644,107 |
| Unmapped records | 497,657 |
| Median abstract length | 144 words / 989 characters |
| 95th-percentile abstract length | 255 words / 1,769 characters |

The broad mapping is: `q-bio` → biology; `physics.chem-ph` → chemistry; `cs` and `stat.ML` → computer science; physics archives → physics; and `econ`, `q-fin`, and non-ML `stat` → social sciences. Mathematics and unrelated records are excluded.

## Label distribution

Shares use mapped records and exceed 100% because papers may have several labels.

| Label | Records | Share |
|---|---:|---:|
| Biology | 56,300 | 2.13% |
| Chemistry | 27,879 | 1.05% |
| Computer science | 1,004,051 | 37.97% |
| Physics | 1,591,437 | 60.19% |
| Social sciences | 114,107 | 4.32% |

Most mapped records have one label (2,501,179); 142,928 have two or more. Chemistry is particularly noisy: 47.3% of its records are secondary cross-lists, usually from physics-adjacent subjects. This limits what can be inferred from the abstract alone.

## Input preparation

A seeded sample of 10,000 mapped abstracts found formulas in 25.13% of records. The production preprocessor keeps short inline math, replaces longer or display formulas with `formula`, strips formatting wrappers, and preserves ordinary percentages. This keeps the text vocabulary-safe and avoids the earlier bug that treated `%` as a TeX comment.

BERT's 512-token context covers 99.51% of raw sampled abstracts and 99.97% after normalization.

## Implications

- This is an imbalanced multi-label problem, so evaluation should include micro and macro F1.
- Abstract duplicates are grouped before splitting; conflicting groups are dropped.
- Hash-based splits provide disjoint, naturally distributed validation and test sets.
- A future time-based holdout would be useful for measuring drift.
