"""Metrics collection and parsing for MoE inference benchmarks.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import re
import subprocess
from typing import Any, Optional


def parse_bench_output(output: str) -> dict[str, Any]:
    """Parse vllm bench serve output to extract metrics.

    Extracts throughput, latency percentiles, TTFT, ITL from the
    benchmark tool's stdout.
    """
    metrics: dict[str, Any] = {}

    # Common patterns in vllm bench serve output
    patterns = {
        "throughput_tok_per_sec": r"Throughput:\s+([\d.]+)\s+tok(?:ens)?/s",
        "throughput_req_per_sec": r"Request throughput:\s+([\d.]+)\s+req/s",
        "ttft_avg_ms": r"(?:TTFT|Time to first token).*?avg[:\s]+([\d.]+)",
        "ttft_p50_ms": r"(?:TTFT|Time to first token).*?p50[:\s]+([\d.]+)",
        "ttft_p99_ms": r"(?:TTFT|Time to first token).*?p99[:\s]+([\d.]+)",
        "itl_avg_ms": r"(?:ITL|Inter-token latency).*?avg[:\s]+([\d.]+)",
        "itl_p50_ms": r"(?:ITL|Inter-token latency).*?p50[:\s]+([\d.]+)",
        "itl_p99_ms": r"(?:ITL|Inter-token latency).*?p99[:\s]+([\d.]+)",
        "e2e_latency_avg_ms": r"(?:E2E|End-to-end).*?avg[:\s]+([\d.]+)",
        "e2e_latency_p99_ms": r"(?:E2E|End-to-end).*?p99[:\s]+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            metrics[key] = float(match.group(1))

    # Try to parse JSON output if present
    json_match = re.search(r"\{[^{}]*\"throughput\"[^{}]*\}", output)
    if json_match:
        try:
            data = json.loads(json_match.group())
            metrics.update({f"raw_{k}": v for k, v in data.items()})
        except json.JSONDecodeError:
            pass

    metrics["raw_output_tail"] = output[-1000:] if len(output) > 1000 else output

    return metrics


def collect_gpu_metrics() -> dict[str, Any]:
    """Collect current GPU utilization and memory metrics.

    Works on both AMD (rocm-smi) and NVIDIA (nvidia-smi).
    """
    # Try AMD first
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {"vendor": "amd", "raw": data}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Try NVIDIA
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "index": int(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                        "utilization_pct": float(parts[3]),
                    })
            return {"vendor": "nvidia", "gpus": gpus}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {"vendor": "unknown"}


def compute_derived_metrics(
    raw_metrics: dict[str, Any],
    num_gpus: int = 1,
    model_params_b: float = 0,
) -> dict[str, Any]:
    """Compute derived metrics from raw benchmark output."""
    derived = {}

    throughput = raw_metrics.get("throughput_tok_per_sec")
    if throughput and num_gpus > 0:
        derived["throughput_per_gpu"] = throughput / num_gpus

    if throughput and model_params_b > 0:
        # Model FLOPs utilization (approximate)
        derived["tokens_per_param_per_sec"] = throughput / (model_params_b * 1e9)

    return derived
