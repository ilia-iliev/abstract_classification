# Throughput methodology

All four fully fine-tuned models were measured on the same machine with an NVIDIA GeForce RTX 3090. This page records the protocol used for the historical measurements; the raw benchmark output is not included in the repository.

- Use the same held-out abstracts and each model's matching preprocessing and tokenizer.
- Bucket texts by tokenized length: 1–128, 129–256, 257–511, and truncated at 512.
- Use 20 warm-up iterations and 100 measured iterations for every model and batch size.
- Synchronize CUDA immediately before and after timed GPU inference.
- Measure batch sizes 1 and 32 separately on one GPU.
- Separate preprocessing and tokenization time from model time and end-to-end time.
- Record p50, p95, and p99 latency, batch-32 abstracts per second, and peak inference VRAM.
- Repeat each measurement five times and use the median, with minimum and maximum as dispersion.
- Measure CPU batch-1 using the same inputs and timing boundaries.
- Measure cold loading in a fresh process.
- Calculate artifact size from the complete serving model.

Model loading was excluded from warm inference timings and included only in the cold-load measurement. Multi-GPU training measurements were kept separate from single-GPU inference results.
