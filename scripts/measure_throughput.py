"""Measure frozen-model throughput using the final-holdout protocol."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from classifier.modeling import BACKBONES, MultilabelClassifier
from classifier.preprocessing import MAX_CONTEXT_LENGTH, prepare_abstract
from scripts.artifacts import write_json
from scripts.data import LABELS, load_manifest_examples
from scripts.hashing import sha256

WARMUP_ITERATIONS = 20
MEASURED_ITERATIONS = 100
REPEATS = 5
BUCKETS = {
    "tokens_1_128": lambda length: length <= 128,
    "tokens_129_256": lambda length: 129 <= length <= 256,
    "tokens_257_511": lambda length: 257 <= length <= 511,
    "truncated_512": lambda length: length >= MAX_CONTEXT_LENGTH,
}



def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "p99_ms": float(np.percentile(values, 99) * 1000),
        "median_ms": float(np.median(values) * 1000),
        "min_ms": float(values.min() * 1000),
        "max_ms": float(values.max() * 1000),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_iterations(operation, device, warmup=WARMUP_ITERATIONS, iterations=MEASURED_ITERATIONS):
    for _ in range(warmup):
        operation()
    synchronize(device)
    values = []
    for _ in range(iterations):
        synchronize(device)
        started = time.perf_counter()
        operation()
        synchronize(device)
        values.append(time.perf_counter() - started)
    return values


def selected_texts(texts, indices, batch_size, repeat):
    if not len(indices):
        return None
    rng = np.random.default_rng(42 + repeat)
    chosen = rng.choice(indices, size=batch_size, replace=len(indices) < batch_size)
    return [str(texts[index]) for index in chosen]


def measure_mode(model, tokenizer, texts, device, batch_size, repeat):
    prepared = [prepare_abstract(text) for text in texts]
    encoded = tokenizer(prepared, padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH, return_tensors="pt")
    encoded_device = {key: value.to(device) for key, value in encoded.items()}

    def tokenize():
        tokenizer(prepared, padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH, return_tensors="pt")

    def model_only():
        with torch.inference_mode():
            model(**encoded_device)

    def end_to_end():
        values = [prepare_abstract(text) for text in texts]
        tokens = tokenizer(values, padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH, return_tensors="pt")
        with torch.inference_mode():
            model(**{key: value.to(device) for key, value in tokens.items()})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = {
        "tokenization": percentile_summary(time_iterations(tokenize, device)),
        "model": percentile_summary(time_iterations(model_only, device)),
        "end_to_end": percentile_summary(time_iterations(end_to_end, device)),
    }
    if device.type == "cuda":
        result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return result


def aggregate_repeats(repeats):
    keys = ("tokenization", "model", "end_to_end")
    result = {"repeats": repeats}
    for key in keys:
        result[key] = {
            metric: {
                "median": float(statistics.median(run[key][metric] for run in repeats)),
                "min": float(min(run[key][metric] for run in repeats)),
                "max": float(max(run[key][metric] for run in repeats)),
            }
            for metric in ("p50_ms", "p95_ms", "p99_ms", "median_ms", "min_ms", "max_ms")
        }
    if "peak_vram_bytes" in repeats[0]:
        result["peak_vram_bytes"] = {
            "median": int(statistics.median(run["peak_vram_bytes"] for run in repeats)),
            "min": int(min(run["peak_vram_bytes"] for run in repeats)),
            "max": int(max(run["peak_vram_bytes"] for run in repeats)),
        }
    return result


def artifact_size(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


def cold_load_seconds(directory, repeats):
    code = (
        "from pathlib import Path; import sys; from classifier.modeling import MultilabelClassifier; "
        "from scripts.data import LABELS; from transformers import AutoTokenizer; "
        "import time; start=time.perf_counter(); AutoTokenizer.from_pretrained(sys.argv[1]); "
        "MultilabelClassifier.load(Path(sys.argv[1]), len(LABELS)); print(time.perf_counter()-start)"
    )
    values = []
    for _ in range(repeats):
        result = subprocess.run((sys.executable, "-c", code, str(directory)), check=True, text=True, capture_output=True)
        values.append(float(result.stdout.strip()))
    return {"seconds": values, "median": float(statistics.median(values)), "min": float(min(values)), "max": float(max(values))}


def hardware():
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers")}
    result = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        result["nvidia_smi"] = subprocess.run(("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"), check=True, text=True, capture_output=True).stdout.strip().splitlines()
    except FileNotFoundError:
        result["nvidia_smi"] = []
    if torch.cuda.is_available():
        result["gpus"] = [{"name": torch.cuda.get_device_name(index), "capability": torch.cuda.get_device_capability(index), "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory} for index in range(torch.cuda.device_count())]
    return result


def validate_inputs(frozen, holdout_report):
    frozen, holdout_report = Path(frozen), Path(holdout_report)
    report_path = holdout_report / "report.json" if holdout_report.is_dir() else holdout_report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "holdout" or report.get("freeze_sha256") != sha256(frozen / "freeze.json"):
        raise ValueError("Throughput requires the final-holdout report from this frozen experiment")
    freeze = json.loads((frozen / "freeze.json").read_text(encoding="utf-8"))
    if set(item["model"] for item in freeze["models"]) != set(BACKBONES):
        raise ValueError("Frozen experiment does not contain exactly the four models")
    code = freeze.get("throughput_measurement_code")
    root = Path(__file__).resolve().parents[1]
    if not code or sha256(frozen / "code" / code["path"]) != code["sha256"]:
        raise ValueError("Frozen experiment does not attest throughput measurement code")
    if sha256(root / code["path"]) != code["sha256"]:
        raise ValueError("Throughput measurement code has changed since freeze")
    return freeze


def measure(frozen, snapshot, holdout_report, output, device=None, repeats=REPEATS):
    frozen, snapshot, output = Path(frozen), Path(snapshot), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite throughput report: {output}")
    freeze = validate_inputs(frozen, holdout_report)
    if sha256(snapshot) != freeze["dataset"]["snapshot_sha256"]:
        raise ValueError("Snapshot does not match frozen experiment")
    rows = load_manifest_examples(snapshot, frozen / "manifests", "holdout")
    texts = np.asarray([row["text"] for row in rows], dtype=object)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index not in (None, 0):
        raise ValueError("Inference protocol requires one GPU; use cuda:0")
    report = {"protocol": {"warmup_iterations": WARMUP_ITERATIONS, "measured_iterations": MEASURED_ITERATIONS, "repeats": repeats, "batches": [1, 32], "buckets": list(BUCKETS), "gpu_synchronization": "before and after every timed iteration"}, "hardware": hardware(), "models": {}}
    for item in freeze["models"]:
        name, directory = item["model"], frozen / item["artifact"]
        tokenizer = AutoTokenizer.from_pretrained(directory)
        lengths = np.asarray([len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts])
        model = MultilabelClassifier.load(directory, len(LABELS)).to(device).eval()
        measurements = {}
        for bucket, matches in BUCKETS.items():
            indices = np.flatnonzero(np.fromiter((matches(int(length)) for length in lengths), dtype=bool, count=len(lengths)))
            if not len(indices):
                measurements[bucket] = {"records": 0}
                continue
            bucket_result = {"records": int(len(indices))}
            for batch_size in (1, 32):
                runs = [measure_mode(model, tokenizer, selected_texts(texts, indices, batch_size, repeat), device, batch_size, repeat) for repeat in range(repeats)]
                summary = aggregate_repeats(runs)
                if batch_size == 32:
                    summary["abstracts_per_second"] = {
                        "median": float(batch_size / (summary["end_to_end"]["median_ms"]["median"] / 1000)),
                        "min": float(batch_size / (summary["end_to_end"]["median_ms"]["max"] / 1000)),
                        "max": float(batch_size / (summary["end_to_end"]["median_ms"]["min"] / 1000)),
                    }
                bucket_result[f"batch_{batch_size}"] = summary
            measurements[bucket] = bucket_result
        cpu_measurements = {}
        cpu_model = model.to("cpu").eval()
        for bucket, matches in BUCKETS.items():
            indices = np.flatnonzero(np.fromiter((matches(int(length)) for length in lengths), dtype=bool, count=len(lengths)))
            if not len(indices):
                cpu_measurements[bucket] = {"records": 0}
                continue
            runs = [measure_mode(cpu_model, tokenizer, selected_texts(texts, indices, 1, repeat), torch.device("cpu"), 1, repeat) for repeat in range(repeats)]
            cpu_measurements[bucket] = {"records": int(len(indices)), "batch_1": aggregate_repeats(runs)}
        runtime = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
        training_seconds = runtime["training_wall_seconds"]
        report["models"][name] = {
            "artifact_size_bytes": artifact_size(directory),
            "cold_load": cold_load_seconds(directory, repeats),
            "token_length_distribution": {"min": int(lengths.min()), "median": float(np.median(lengths)), "max": int(lengths.max())},
            "inference_device": str(device),
            "inference": measurements,
            "cpu_batch_1_inference": cpu_measurements,
            "reported_training": {**runtime, "examples_per_second": float(json.loads((directory / "configuration.json").read_text(encoding="utf-8"))["data"]["training_records"] / training_seconds), "source": "frozen selected-training artifact"},
        }
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Measure all frozen models using the final-holdout throughput protocol.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("--holdout-report", type=Path, default=Path("artifacts/final-holdout-evaluation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/throughput"))
    parser.add_argument("--device")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.repeats < 1: parser.error("--repeats must be positive")
    report = measure(**vars(args))
    print(json.dumps({"output": str(args.output), "models": list(report["models"])}, sort_keys=True))


if __name__ == "__main__":
    main()
