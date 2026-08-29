# Model benchmark plan

## Goal

Benchmark four pretrained backbones for five-label arXiv abstract classification:

- `google-bert/bert-base-uncased`
- `answerdotai/ModernBERT-base`
- `google/embeddinggemma-300m`
- `Qwen/Qwen3-Embedding-0.6B`

BERT is one of the measured models, not a control with privileged treatment.

The primary metric is macro F1. The experiment must also measure predictive quality, calibration, training cost, and inference performance on our hardware.

## 1. Build an entirely new dataset

Download a new arXiv metadata snapshot and construct all experiment splits from it. Do not reuse the repository's current training, validation, or test samples.

Persist immutable manifests containing record IDs and content hashes:

| Split | Records | Use |
|---|---:|---|
| Training | 100,000 | Model fitting |
| Validation | 20,000 | Optuna, checkpoint evaluation, and threshold selection |
| Benchmark | 20,000 | Comparison after configurations are frozen |
| Final holdout | 20,000 | One-time final evaluation |

Requirements:

- Splits are deterministic and disjoint.
- Abstracts are normalized before content-hash deduplication.
- Duplicate abstracts cannot cross splits.
- Duplicate groups with conflicting labels are discarded.
- Validation, benchmark, and holdout retain the natural label distribution.
- Training uses the existing tag-aware selection and preserves at least 8,000 positives per label where available.
- Split manifests are generated once and reused by every model and seed-free rerun.
- Dataset identity, snapshot date, split seed, counts, label distributions, and hashes are recorded.

The existing test set is considered development history. It is not part of this benchmark.

## 2. Freeze preprocessing and targets

Every model receives the same normalized abstract text and labels:

- Abstract only
- Existing TeX-safe normalization
- Existing `<FORMULA>` placeholder
- Five independent labels
- Primary label target: `1.0`
- Secondary-only label target: `0.5`
- Maximum input length: 512 model tokens

The secondary target is fixed at `0.5`. It must not be an Optuna parameter.

Register `<FORMULA>` with every tokenizer and train its token embedding during full fine-tuning. Record token-count distributions and truncation rates for every tokenizer.

## 3. Use PyTorch exclusively

Implement one PyTorch path for training, evaluation, artifact loading, and API inference. All four models must use it.

Each classifier has:

1. The checkpoint's tokenizer and backbone
2. The checkpoint's documented pooling strategy
3. Dropout
4. One linear layer producing five logits
5. Independent sigmoid probabilities
6. The shared multilabel loss and thresholding implementation

Use current compatible releases of PyTorch, Transformers, and related libraries.

## 4. Maintain regression coverage

Keep deterministic tests for:

- Record filtering and broad-label mapping
- Primary and secondary target construction
- Deduplication and split assignment
- Preprocessing and formula replacement
- Token truncation and padding
- Weighted multilabel loss
- Sigmoid probabilities
- Per-label threshold selection, including ties
- Exact-match accuracy
- Micro, macro, and weighted precision, recall, and F1
- Per-label metrics

Run the existing API tests for single and batched requests, output schema and label order, threshold fallback, artifact validation, missing or incompatible artifacts, and CPU and GPU inference.

## 5. Run frozen-representation probes

Before full fine-tuning, freeze each backbone and train only the classification head.

This stage serves two purposes:

- Detect integration, pooling, masking, and tokenization bugs cheaply.
- Measure the quality and operating cost of frozen representations.

Use the same training and validation records for every model. Save artifacts, configuration, predictions, metrics, runtime, and peak VRAM. Report these results in a separate frozen-probe table; do not mix them with full fine-tuning results.

## 6. Tune each model

Run exactly 16 Optuna trials per model on the same fixed 5,000-record, rare-label-aware subset of the immutable training manifest. Every trial trains for exactly one epoch.

Search only:

- Learning rate
- Weight decay
- Warmup ratio

Use the same search space, sampler seed, trial count, fixed tuning subset, validation records, effective batch size, loss, and selection metric for every model. Dynamic padding and gradient accumulation may be used to reach the common effective batch size.

Select each model's hyperparameters by validation macro F1 after tuning per-label thresholds on validation predictions.

Optuna must not search:

- Epoch count
- Secondary-label target
- Training sample composition
- Sequence length
- Classification-head design
- Threshold candidate grid

Save every trial's parameters, metrics, predictions, duration, peak VRAM, and failure status. A failed trial remains part of the 16-trial budget unless the failure is an implementation or infrastructure fault affecting the validity of the run.

## 7. Train the selected configurations

Train each selected configuration once, for one epoch, from the original pretrained checkpoint. There are no multi-seed repeats.

Keep constant:

- Training examples and order
- Effective batch size
- Loss
- Maximum sequence length
- Precision policy
- Gradient clipping
- Checkpoint-selection rule
- Validation threshold grid

Save:

- Model and tokenizer
- Model and data configuration
- Dependency versions and git commit
- Validation logits and probabilities
- Per-label thresholds
- Metrics
- Training history
- Wall-clock duration
- GPU-hours
- Peak VRAM

## 8. Freeze the experiment

Before benchmark evaluation, freeze and record:

- Model checkpoints
- Hyperparameters
- Preprocessing version
- Split manifests
- Classification-head and pooling implementations
- Per-label thresholds
- Metric code
- Throughput protocol

No model may be retrained or adjusted in response to benchmark results.

## 9. Evaluate the benchmark

Evaluate all four frozen models on the benchmark split once.

Report:

- Macro F1
- Micro F1
- Weighted F1
- Exact-match accuracy
- Per-label precision, recall, F1, and PR-AUC
- BCE/log loss
- Calibration error
- Top-1 accuracy
- Paired bootstrap confidence intervals for metrics and model differences

Because each model is trained once, report bootstrap uncertainty over examples but do not present it as training-seed uncertainty.

Include subgroup metrics in the same report rather than treating error analysis as a separate model-development step:

- Primary versus secondary-only labels
- Chemistry–physics overlaps
- Multilabel versus single-label records
- Formula-heavy abstracts
- Truncated versus non-truncated abstracts
- Abstract-length buckets
- Model disagreement counts

## 10. Evaluate the final holdout

After the benchmark report and provisional ranking are complete, evaluate every frozen model once on the untouched final holdout. Do not retrain, retune, or alter thresholds afterward.

Produce the final quality table with the same overall, per-label, subgroup, calibration, and bootstrap metrics used for the benchmark.

## 11. Measure throughput on our system

Measure all four frozen models on the repository's actual hardware, including the available RTX 3090 GPUs. Record exact GPU, CPU, driver, CUDA, PyTorch, precision, and software versions.

Measure:

- Single-GPU training examples per second
- End-to-end training wall time and GPU-hours
- Peak training VRAM
- Batch-1 inference latency: p50, p95, and p99
- Batch-32 inference latency and abstracts per second
- Peak inference VRAM
- CPU batch-1 latency
- Model artifact size
- Cold model-load time

Use the same final-holdout texts, preprocessing, sequence-length buckets, warm-up count, measured iteration count, and synchronization rules for every model. Separate tokenization time, model time, and end-to-end time. Run throughput measurements repeatedly and report the median and dispersion.

Training may use both GPUs, but production inference measurements must include a single-GPU result so model comparisons are not distorted by different device placement.

## 12. Deliverables

Commit or archive:

- Dataset and split manifests
- Migration parity fixtures and reports
- Frozen-probe results
- Optuna studies and trial reports
- Final model artifacts
- Validation, benchmark, and holdout predictions
- Quality and subgroup metric reports
- Bootstrap comparison report
- Hardware throughput report
- Reproduction commands
- Updated API and training documentation

The final recommendation must consider macro F1 first, then rare-label behavior, calibration, throughput, VRAM, and artifact size. A larger model wins only if its measured quality gain justifies its additional cost.
