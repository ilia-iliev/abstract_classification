# Throughput protocol

Use the frozen final-holdout texts and the frozen preprocessing and tokenizer from each model artifact. Run all models on the same machine and record GPU model, CPU, driver, CUDA, PyTorch, precision policy, and dependency versions.

- Bucket texts by the tokenized 512-token limit: 1–128, 129–256, 257–511, and truncated (512).
- Use the same fixed warm-up count (20) and measured iteration count (100) for every model and batch size.
- Synchronize CUDA immediately before and after every timed GPU iteration. Do not include model loading in warm inference timings.
- Measure batch 1 and batch 32 separately on one GPU. Report p50, p95, and p99 latency; batch-32 abstracts/second; and peak inference VRAM.
- For each batch size, report tokenization-only, model-only, and end-to-end timings. End-to-end includes preprocessing and tokenization.
- Repeat every measurement five times and report the median and dispersion (minimum and maximum).
- Measure CPU batch-1 with the same texts, warm-up, iteration count, and timing boundaries.
- Measure cold-load time in a fresh process, and report artifact size as the sum of frozen model-artifact files.
- Record single-GPU training examples/second, final training wall time, GPU-hours, and peak training VRAM from the frozen training artifact. Multi-GPU training, if used, is reported separately and never replaces the single-GPU inference result.

Run after the final-holdout report exists:

```bash
uv run --extra training python -m scripts.measure_throughput \
  data/arxiv-benchmark-snapshot.json artifacts/frozen-experiment \
  --holdout-report artifacts/final-holdout-evaluation \
  --output artifacts/throughput
```

The command refuses to overwrite output and verifies the holdout report, frozen artifact hashes, snapshot, and frozen measurement source. It records every repeat as well as median, minimum, and maximum summaries.
