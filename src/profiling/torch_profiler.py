"""Torch Profiler wrapper for MoE inference profiling.

Works on both ROCm (AMD) and CUDA (NVIDIA) backends.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


def get_profiler_config(
    trace_dir: str = "results/profiles",
    wait_steps: int = 2,
    warmup_steps: int = 2,
    active_steps: int = 5,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    with_flops: bool = True,
) -> dict:
    """Get torch profiler configuration dict.

    This can be used to set VLLM_TORCH_PROFILER_DIR and related
    environment variables before launching vLLM.
    """
    return {
        "trace_dir": trace_dir,
        "wait_steps": wait_steps,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
        "with_flops": with_flops,
    }


def setup_vllm_profiling_env(trace_dir: str = "results/profiles") -> dict[str, str]:
    """Return environment variables to enable vLLM's built-in torch profiler.

    vLLM supports profiling via VLLM_TORCH_PROFILER_DIR.
    The traces can be viewed in chrome://tracing or Perfetto UI.
    """
    trace_path = Path(trace_dir).resolve()
    trace_path.mkdir(parents=True, exist_ok=True)

    return {
        "VLLM_TORCH_PROFILER_DIR": str(trace_path),
    }


@contextmanager
def torch_profile_context(
    output_dir: str = "results/profiles",
    prefix: str = "moe_profile",
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
    with_flops: bool = True,
):
    """Context manager for direct torch profiling of Python code.

    Use this for profiling custom placement/batching code,
    not for vLLM server profiling (use setup_vllm_profiling_env instead).

    Example:
        with torch_profile_context("results/profiles", "my_test") as prof:
            # ... code to profile ...
        # Traces saved automatically
    """
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity, schedule
    except ImportError:
        print("Warning: torch not available, profiling disabled")
        yield None
        return

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as prof:
        yield prof

    # Export chrome trace
    trace_file = output_path / f"{prefix}.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"Chrome trace saved: {trace_file}")

    # Print summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
