# Model evaluation

The active model is evaluated on a disjoint, natural-distribution test split. The test set has also been used to compare experiments, so it is not an untouched final estimate.

## Overall results

| Split | Accuracy | Micro F1 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Validation | 0.9070 | 0.9456 | 0.7752 | 0.9450 |
| Test | 0.9048 | 0.9437 | 0.7662 | 0.9427 |

The test macro-F1 target (>0.76) is met.

## Test results by category

| Category | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Biology | 441 | 0.7619 | 0.5805 | 0.6589 |
| Chemistry | 209 | 0.5000 | 0.5120 | 0.5059 |
| Computer science | 7,658 | 0.9449 | 0.9521 | 0.9485 |
| Physics | 11,990 | 0.9683 | 0.9741 | 0.9712 |
| Social sciences | 865 | 0.7726 | 0.7225 | 0.7467 |

Chemistry is the weakest label, consistent with its frequent secondary cross-listing and overlap with physics.

## Comparison

The promoted formula-token model reaches test macro F1 0.7662, improving slightly on the previous active model (0.7650). Full metrics and settings are in `artifacts/model/metadata.json`.
