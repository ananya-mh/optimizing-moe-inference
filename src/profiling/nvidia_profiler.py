"""NVIDIA profiling wrapper using Nsight Systems.

For NVIDIA GPUs.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def is_nsight_available() -> bool:
    """Check if Nsight Systems is available."""
    return shutil.which("nsys") is not None


def run_nsight_profile(
    command: list[str],
    output_dir: str = "results/profiles/nvidia",
    trace_cuda: bool = True,
    trace_nvtx: bool = True,
    trace_osrt: bool = False,
    duration: Optional[int] = None,
    extra_args: Optional[list[str]] = None,
) -> Optional[str]:
    """Run a command under Nsight Systems profiling.

    Args:
        command: The command to profile
        output_dir: Directory for output files
        trace_cuda: Enable CUDA API/kernel tracing
        trace_nvtx: Enable NVTX marker tracing
        trace_osrt: Enable OS runtime tracing
        duration: Max profiling duration in seconds
        extra_args: Additional nsys arguments

    Returns:
        Path to .nsys-rep file, or None on failure.
    """
    if not is_nsight_available():
        print("Warning: nsys not found. Skipping Nsight profiling.")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "profile"

    prof_cmd = [
        "nsys", "profile",
        "-o", str(output_file),
        "--force-overwrite=true",
    ]

    traces = []
    if trace_cuda:
        traces.append("cuda")
    if trace_nvtx:
        traces.append("nvtx")
    if trace_osrt:
        traces.append("osrt")
    if traces:
        prof_cmd.extend(["--trace", ",".join(traces)])

    if duration:
        prof_cmd.extend(["--duration", str(duration)])

    if extra_args:
        prof_cmd.extend(extra_args)

    prof_cmd.extend(command)

    print(f"Running nsys profile: {' '.join(prof_cmd[:8])}...")

    try:
        result = subprocess.run(
            prof_cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode == 0:
            print(f"Nsight profile saved to {out_path}")
            return str(output_file) + ".nsys-rep"
        else:
            print(f"nsys failed: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        print("nsys timed out")
        return None
