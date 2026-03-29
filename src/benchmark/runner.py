"""Main benchmark orchestrator for MoE inference experiments.

Launches vLLM server, runs benchmarks with vllm bench serve,
collects metrics, and saves results.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    get_model_config,
    get_experiment_config,
    get_model_dir,
    detect_gpu_vendor,
    resolve_experiment_env,
)
from utils.gpu_info import detect_gpus, print_gpu_summary
from benchmark.metrics import parse_bench_output, collect_gpu_metrics


VLLM_SERVER_PORT = 8000
HEALTH_ENDPOINT = f"http://localhost:{VLLM_SERVER_PORT}/health"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def build_server_cmd(
    model_id: str,
    tp_size: int = 1,
    dp_size: int = 1,
    enable_ep: bool = False,
    all2all_backend: Optional[str] = None,
    max_model_len: int = 4096,
    gpu_mem_util: float = 0.90,
    dtype: str = "auto",
    chunked_prefill: bool = True,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the vllm serve command."""
    cmd = [
        "vllm", "serve", model_id,
        "--port", str(VLLM_SERVER_PORT),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--dtype", dtype,
        "--disable-log-requests",
        "--enforce-eager",
    ]
    if dp_size > 1:
        cmd.extend(["--data-parallel-size", str(dp_size)])
    if enable_ep:
        cmd.append("--enable-expert-parallel")
    if all2all_backend:
        cmd.extend(["--all2all-backend", all2all_backend])
    if chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_bench_cmd(
    num_prompts: int = 200,
    input_len: int = 512,
    output_len: int = 256,
    concurrency: int = 16,
    port: int = VLLM_SERVER_PORT,
) -> list[str]:
    """Build the vllm bench serve command."""
    return [
        "vllm", "bench", "serve",
        "--dataset-name", "random",
        "--endpoint", "/v1/completions",
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--max-concurrency", str(concurrency),
        "--port", str(port),
    ]


def wait_for_server(timeout: int = 300, interval: int = 5) -> bool:
    """Wait for vLLM server to be healthy."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(HEALTH_ENDPOINT, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def run_bench_against_server(
    server_cmd_str: str,
    model_id: str,
    strategy: dict[str, Any],
    workload: dict[str, Any],
    concurrency: int,
    vendor: str,
    env: dict[str, str],
    bench_timeout: int = 600,
) -> dict[str, Any]:
    """Run a single benchmark against an already-running server."""
    bench_cmd = build_bench_cmd(
        num_prompts=workload.get("num_prompts", 200),
        input_len=workload.get("input_len", 512),
        output_len=workload.get("output_len", 256),
        concurrency=concurrency,
    )

    result = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "strategy": strategy,
        "workload": workload,
        "concurrency": concurrency,
        "vendor": vendor,
        "server_cmd": server_cmd_str,
        "bench_cmd": " ".join(bench_cmd),
    }

    try:
        gpu_before = collect_gpu_metrics()

        print(f"  Running benchmark (concurrency={concurrency})...")
        bench_result = subprocess.run(
            bench_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=bench_timeout,
        )

        gpu_after = collect_gpu_metrics()

        if bench_result.returncode == 0:
            metrics = parse_bench_output(bench_result.stdout)
            result["metrics"] = metrics
            result["gpu_before"] = gpu_before
            result["gpu_after"] = gpu_after
            result["success"] = True
        else:
            result["error"] = bench_result.stderr[-500:] if bench_result.stderr else "Unknown error"
            result["success"] = False

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
        result["success"] = False
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result


def run_strategy_benchmarks(
    model_id: str,
    strategy: dict[str, Any],
    runs: list[tuple[dict[str, Any], int]],
    vendor: str,
    env_overrides: dict[str, str],
    profile: bool = False,
    bench_timeout: int = 600,
    run_offset: int = 0,
    total_runs: int = 0,
) -> list[dict[str, Any]]:
    """Start one server for a strategy and run all workload/concurrency combos against it."""
    env = os.environ.copy()
    env.update(env_overrides)
    if profile:
        env["VLLM_TORCH_PROFILER_DIR"] = str(
            RESULTS_DIR / "profiles" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    tp_size = strategy.get("tensor_parallel_size", 1)
    dp_size = strategy.get("data_parallel_size", 1)
    enable_ep = strategy.get("enable_expert_parallel", False)
    all2all = strategy.get("all2all_backend")

    server_cmd = build_server_cmd(
        model_id=model_id,
        tp_size=tp_size,
        dp_size=dp_size,
        enable_ep=enable_ep,
        all2all_backend=all2all,
    )
    server_cmd_str = " ".join(server_cmd)

    results = []
    server_proc = None
    try:
        print(f"  Starting vLLM server: {' '.join(server_cmd[:6])}...")
        server_proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print("  Waiting for server health...")
        if not wait_for_server():
            error = "Server failed to start within timeout"
            print(f"  FAILED: {error}")
            for wl, conc in runs:
                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "model_id": model_id,
                    "strategy": strategy,
                    "workload": wl,
                    "concurrency": conc,
                    "vendor": vendor,
                    "server_cmd": server_cmd_str,
                    "error": error,
                    "success": False,
                })
            return results

        for i, (wl, conc) in enumerate(runs):
            run_num = run_offset + i + 1
            print(f"\n--- Run {run_num}/{total_runs} ---")
            print(f"  Strategy: {strategy['name']}, Workload: {wl['name']}, Concurrency: {conc}")

            result = run_bench_against_server(
                server_cmd_str=server_cmd_str,
                model_id=model_id,
                strategy=strategy,
                workload=wl,
                concurrency=conc,
                vendor=vendor,
                env=env,
                bench_timeout=bench_timeout,
            )
            results.append(result)

            if result.get("success"):
                metrics = result.get("metrics", {})
                print(f"  Throughput: {metrics.get('throughput_tok_per_sec', 'N/A')} tok/s")
                print(f"  TTFT: {metrics.get('ttft_avg_ms', 'N/A')} ms")
            else:
                print(f"  FAILED: {result.get('error', 'unknown')}")

    except Exception as e:
        print(f"  Server error: {e}")
    finally:
        if server_proc:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    return results


def save_results(results: list[dict], experiment_name: str, model_key: str):
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{model_key}_{timestamp}.json"
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")
    return path


@click.command()
@click.option("--model", "-m", required=True, help="Model key from configs/models.yaml")
@click.option("--experiment", "-e", default="single_gpu", help="Experiment name")
@click.option("--num-gpus", "-g", default=None, type=int, help="Number of GPUs (auto-detect if not set)")
@click.option("--concurrency", "-c", default=None, type=int, help="Override concurrency (sweep if not set)")
@click.option("--workload", "-w", default=None, help="Specific workload name (run all if not set)")
@click.option("--strategy", "-s", default=None, help="Specific strategy name (run all if not set)")
@click.option("--profile", is_flag=True, help="Enable torch profiler")
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
@click.option("--bench-timeout", "-t", default=600, type=int, help="Benchmark subprocess timeout in seconds (default 600)")
def main(model, experiment, num_gpus, concurrency, workload, strategy, profile, dry_run, bench_timeout):
    """MoE Inference Benchmark Runner.

    Runs systematic benchmarks across models, placement strategies,
    workloads, and concurrency levels.
    """
    print("=" * 60)
    print("MoE Inference Optimization - Benchmark Runner")
    print("=" * 60)

    # Detect GPU
    vendor = detect_gpu_vendor()
    print(f"\nGPU Vendor: {vendor.upper()}")
    print_gpu_summary()

    # Load configs
    model_cfg = get_model_config(model)
    exp_cfg = get_experiment_config(experiment)
    exp = exp_cfg.get("experiment", exp_cfg)
    env_vars = resolve_experiment_env(exp_cfg, vendor)

    model_id = model_cfg["hf_model_id"]
    print(f"\nModel: {model_id}")
    print(f"  Total params: {model_cfg['total_params_b']}B")
    print(f"  Active params: {model_cfg['active_params_b']}B")
    print(f"  Experts: {model_cfg['num_experts']} (top-{model_cfg['top_k']})")

    # Determine sweep parameters
    strategies = exp.get("placement_strategies", [{"name": "default"}])
    if strategy:
        strategies = [s for s in strategies if s["name"] == strategy]

    workloads = exp.get("workloads", [{"name": "default", "num_prompts": 200, "input_len": 512, "output_len": 256}])
    if workload:
        workloads = [w for w in workloads if w["name"] == workload]

    concurrency_levels = [concurrency] if concurrency else exp.get("concurrency_levels", [16])

    total_runs = len(strategies) * len(workloads) * len(concurrency_levels)
    print(f"\nExperiment: {exp.get('name', experiment)}")
    print(f"  Strategies: {[s['name'] for s in strategies]}")
    print(f"  Workloads: {[w['name'] for w in workloads]}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Total runs: {total_runs}")

    if dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for strat in strategies:
            cmd = build_server_cmd(model_id, tp_size=strat.get("tensor_parallel_size", 1))
            print(f"  Server: {' '.join(cmd)}")
        return

    # Run experiments — group by strategy to reuse server across workloads/concurrency
    all_results = []
    run_idx = 0
    for strat in strategies:
        strat_runs = [(wl, conc) for wl in workloads for conc in concurrency_levels]
        print(f"\n{'=' * 40}")
        print(f"Strategy: {strat['name']} ({len(strat_runs)} runs)")
        print(f"{'=' * 40}")

        results = run_strategy_benchmarks(
            model_id=model_id,
            strategy=strat,
            runs=strat_runs,
            vendor=vendor,
            env_overrides=env_vars,
            profile=profile,
            bench_timeout=bench_timeout,
            run_offset=run_idx,
            total_runs=total_runs,
        )
        all_results.extend(results)
        run_idx += len(strat_runs)

    # Save results
    save_results(all_results, experiment, model)
    print(f"\nCompleted {len(all_results)} benchmark runs.")


if __name__ == "__main__":
    main()
