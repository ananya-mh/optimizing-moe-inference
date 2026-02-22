"""GPU detection and information gathering.

Supports both AMD (ROCm) and NVIDIA (CUDA) GPUs.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    vendor: str  # "amd" or "nvidia"
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_pct: float
    temperature_c: Optional[float] = None
    compute_units: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def get_amd_gpu_info() -> list[GPUInfo]:
    """Get GPU information from AMD ROCm via rocm-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        mem_data = json.loads(result.stdout) if result.returncode == 0 else {}

        result2 = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            capture_output=True, text=True, timeout=10
        )
        use_data = json.loads(result2.stdout) if result2.returncode == 0 else {}

        result3 = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=10
        )
        name_data = json.loads(result3.stdout) if result3.returncode == 0 else {}

        # Parse GPU count from available data
        gpu_keys = [k for k in mem_data if k.startswith("card")]
        if not gpu_keys:
            # Fallback: count GPUs via rocm-smi -i
            result_i = subprocess.run(
                ["rocm-smi", "-i", "--json"],
                capture_output=True, text=True, timeout=10
            )
            if result_i.returncode == 0:
                info = json.loads(result_i.stdout)
                gpu_keys = [k for k in info if k.startswith("card")]

        for i, key in enumerate(sorted(gpu_keys)):
            mem_info = mem_data.get(key, {})
            total = float(mem_info.get("VRAM Total Memory (B)", 0)) / (1024**3)
            used = float(mem_info.get("VRAM Total Used Memory (B)", 0)) / (1024**3)

            use_info = use_data.get(key, {})
            util = float(use_info.get("GPU use (%)", 0))

            name_info = name_data.get(key, {})
            name = name_info.get("Card Series", f"AMD GPU {i}")

            gpus.append(GPUInfo(
                index=i,
                name=name,
                vendor="amd",
                memory_total_gb=round(total, 2),
                memory_used_gb=round(used, 2),
                memory_free_gb=round(total - used, 2),
                utilization_pct=util,
            ))
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return gpus


def get_nvidia_gpu_info() -> list[GPUInfo]:
    """Get GPU information from NVIDIA via nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vendor="nvidia",
                        memory_total_gb=round(float(parts[2]) / 1024, 2),
                        memory_used_gb=round(float(parts[3]) / 1024, 2),
                        memory_free_gb=round(float(parts[4]) / 1024, 2),
                        utilization_pct=float(parts[5]),
                        temperature_c=float(parts[6]),
                    ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Auto-detect GPUs (tries AMD first, then NVIDIA)."""
    gpus = get_amd_gpu_info()
    if gpus:
        return gpus
    return get_nvidia_gpu_info()


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    return len(detect_gpus())


def get_gpu_vendor() -> str:
    """Detect GPU vendor string."""
    gpus = detect_gpus()
    if not gpus:
        raise RuntimeError("No GPUs detected")
    return gpus[0].vendor


def get_gpu_memory_gb() -> float:
    """Get total memory of first GPU in GB."""
    gpus = detect_gpus()
    if not gpus:
        return 0.0
    return gpus[0].memory_total_gb


def print_gpu_summary():
    """Print a summary of detected GPUs."""
    gpus = detect_gpus()
    if not gpus:
        print("No GPUs detected!")
        return
    print(f"Detected {len(gpus)} {gpus[0].vendor.upper()} GPUs:")
    for gpu in gpus:
        print(
            f"  [{gpu.index}] {gpu.name}: "
            f"{gpu.memory_free_gb:.1f}/{gpu.memory_total_gb:.1f} GB free, "
            f"{gpu.utilization_pct:.0f}% util"
        )


if __name__ == "__main__":
    print_gpu_summary()
