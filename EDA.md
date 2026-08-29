# Exploratory data analysis

These figures were generated from the Cornell arXiv snapshot used for the experiment. They are retained as a summary of that analysis; the source snapshot and raw analysis output are not part of this repository.

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

Shares use mapped records and exceed 100% because papers may have several labels. These five labels are project-defined groupings of Cornell's arXiv subject codes, not labels supplied directly by Cornell.

| Label | Records | Share |
|---|---:|---:|
| Biology | 56,300 | 2.13% |
| Chemistry | 27,879 | 1.05% |
| Computer science | 1,004,051 | 37.97% |
| Physics | 1,591,437 | 60.19% |
| Social sciences | 114,107 | 4.32% |

Most mapped records have one label (2,501,179); 142,928 have two or more. Chemistry is particularly noisy: 47.3% of its records are secondary cross-lists, usually from physics-adjacent subjects. This limits what can be inferred from the abstract alone.

The physics count was independently checked against the raw subject codes. Of all 3,141,764 snapshot records, 1,511,713 (48.12%) have a primary physics-archive category and 1,591,437 (50.65%) have physics as either a primary category or a cross-list. The reported 60.19% is higher because its denominator excludes 497,657 unmapped records, mostly mathematics. The largest primary physics archives are condensed matter (353,969), astrophysics (341,102), the `physics.*` archive (200,896), high-energy phenomenology (144,286), and quantum physics (135,288). The high share is therefore present in the snapshot rather than caused by an accidental category match.

## Input preparation

A seeded sample of 10,000 mapped abstracts found formulas in 25.13% of records. The production preprocessor keeps short inline math, replaces longer or display formulas with the dedicated `<FORMULA>` token, strips formatting wrappers, and preserves ordinary percentages. This keeps the text vocabulary-safe and avoids the earlier bug that treated `%` as a TeX comment.

For BERT, normalization reduced the median from 217 to 207 tokens and the 95th percentile from 385 to 357. The share beyond the 512-token context fell from 0.49% to 0.03%, and the normalized sample produced no `[UNK]` tokens.

## Implications

- This is an imbalanced multi-label problem, so evaluation should include micro and macro F1.
- Abstract duplicates are grouped before splitting; conflicting groups are dropped.
- Hash-based splits provide disjoint, naturally distributed validation and test sets.
- A future time-based holdout would be useful for measuring drift.
