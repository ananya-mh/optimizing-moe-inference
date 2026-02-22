"""ROCm profiling wrapper using rocprofv3.

For AMD MI300X GPUs.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def is_rocm_available() -> bool:
    """Check if ROCm profiling tools are available."""
    return shutil.which("rocprofv3") is not None


def run_rocprof(
    command: list[str],
    output_dir: str = "results/profiles/rocm",
    hip_trace: bool = True,
    kernel_trace: bool = True,
    memory_copy_trace: bool = True,
    output_format: str = "csv",
    extra_args: Optional[list[str]] = None,
) -> Optional[str]:
    """Run a command under rocprofv3 profiling.

    Args:
        command: The command to profile (e.g., ["python", "my_script.py"])
        output_dir: Directory for output files
        hip_trace: Enable HIP API tracing
        kernel_trace: Enable kernel dispatch tracing
        memory_copy_trace: Enable memory copy tracing
        output_format: Output format (csv, json, pftrace, otf2)
        extra_args: Additional rocprofv3 arguments

    Returns:
        Path to output directory, or None on failure.
    """
    if not is_rocm_available():
        print("Warning: rocprofv3 not found. Skipping ROCm profiling.")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prof_cmd = ["rocprofv3"]

    if hip_trace:
        prof_cmd.append("--hip-trace")
    if kernel_trace:
        prof_cmd.append("--kernel-trace")
    if memory_copy_trace:
        prof_cmd.append("--memory-copy-trace")

    prof_cmd.extend(["-o", str(out_path / "trace")])
    prof_cmd.extend(["--output-format", output_format])

    if extra_args:
        prof_cmd.extend(extra_args)

    prof_cmd.append("--")
    prof_cmd.extend(command)

    print(f"Running rocprofv3: {' '.join(prof_cmd[:8])}...")

    try:
        result = subprocess.run(
            prof_cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )
        if result.returncode == 0:
            print(f"ROCm profile saved to {out_path}")
            return str(out_path)
        else:
            print(f"rocprofv3 failed: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        print("rocprofv3 timed out")
        return None


def parse_rocprof_csv(csv_path: str) -> list[dict]:
    """Parse rocprofv3 CSV output into a list of event dicts."""
    import csv as csv_mod
    events = []
    try:
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                events.append(dict(row))
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
    return events
