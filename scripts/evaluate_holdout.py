"""Evaluate frozen classifiers once on the final holdout after benchmarking."""

import argparse
import json
from pathlib import Path

from classifier.modeling import BACKBONES
from scripts.evaluate_benchmark import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    evaluate_split,
    sha256,
)


def validate_benchmark_report(path, frozen):
    """Require the completed benchmark report and its provisional ranking."""
    path = Path(path)
    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.is_file():
        raise FileNotFoundError(f"Benchmark report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_models = set(BACKBONES)
    if report.get("split") != "benchmark":
        raise ValueError("The prerequisite report is not a benchmark evaluation")
    if report.get("freeze_sha256") != sha256(Path(frozen) / "freeze.json"):
        raise ValueError("Benchmark report was produced from a different frozen experiment")
    if set(report.get("models", {})) != expected_models:
        raise ValueError("Benchmark report does not contain exactly the four frozen models")
    ranking = report.get("provisional_ranking")
    if not isinstance(ranking, list) or len(ranking) != len(expected_models) or set(ranking) != expected_models:
        raise ValueError("Benchmark report has no complete provisional ranking")
    return report


def evaluate(frozen, snapshot, benchmark_report, output, batch_size=16, device=None, bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES, seed=42):
    frozen = Path(frozen)
    validate_benchmark_report(benchmark_report, frozen)
    return evaluate_split(
        frozen, snapshot, output, "holdout", "holdout_metric_code", batch_size, device,
        bootstrap_samples, seed, additional_metric_code_keys=("benchmark_metric_code",),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen models once on the final holdout.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("frozen", type=Path, help="artifacts/frozen-experiment")
    parser.add_argument("--benchmark-report", type=Path, default=Path("artifacts/benchmark-evaluation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final-holdout-evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("--batch-size and --bootstrap-samples must be positive")
    report = evaluate(**vars(args))
    print(json.dumps({"output": str(args.output), "records": report["records"], "models": list(report["models"])}, sort_keys=True))


if __name__ == "__main__":
    main()
