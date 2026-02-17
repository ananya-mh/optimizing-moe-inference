"""
Profiling infrastructure for MoE inference on AMD ROCm.

Supports:
  - torch.profiler with ROCm support (HIP events)
  - rocprofv3 integration for kernel-level tracing
  - Expert routing analysis (activation distribution, load balance)
  - Communication profiling (all-to-all latency, bandwidth)

Usage:
    profiler = MoEProfiler(output_dir="./profiles")
    with profiler.profile_inference(model_name="mixtral"):
        # run inference
        pass
    report = profiler.get_report()
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class KernelProfile:
    """Profile of a single GPU kernel execution."""
    name: str
    duration_us: float
    is_moe_kernel: bool = False
    is_attention_kernel: bool = False
    is_communication: bool = False


@dataclass
class StepProfile:
    """Profile of a single inference step (forward pass)."""
    step_index: int
    total_time_ms: float
    model_forward_ms: float
    moe_routing_ms: float = 0.0
    moe_compute_ms: float = 0.0
    attention_ms: float = 0.0
    communication_ms: float = 0.0
    expert_activations: Optional[Dict[int, int]] = None


class MoEProfiler:
    """
    Profiler for MoE inference on AMD ROCm.

    Collects both high-level timing data and kernel-level traces.
    """

    def __init__(self, output_dir: str = "./profiles", enable_torch_profiler: bool = True):
        self.output_dir = output_dir
        self.enable_torch_profiler = enable_torch_profiler
        self.step_profiles: List[StepProfile] = []
        self.kernel_profiles: List[KernelProfile] = []
        self._profiler = None

    @contextmanager
    def profile_inference(self, model_name: str = "model", num_steps: int = 0):
        """Context manager for profiling an inference run."""
        os.makedirs(self.output_dir, exist_ok=True)

        if self.enable_torch_profiler:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            self._profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(self.output_dir, model_name)
                ),
            )
            self._profiler.__enter__()

        try:
            yield self
        finally:
            if self._profiler is not None:
                self._profiler.__exit__(None, None, None)
                # Export chrome trace
                trace_path = os.path.join(self.output_dir, "%s_trace.json" % model_name)
                self._profiler.export_chrome_trace(trace_path)
                self._profiler = None

    def record_step(self, step_profile: StepProfile):
        """Record profiling data for a single step."""
        self.step_profiles.append(step_profile)

    def get_report(self) -> Dict[str, Any]:
        """Generate a profiling report."""
        if not self.step_profiles:
            return {"error": "No steps profiled"}

        total_times = [s.total_time_ms for s in self.step_profiles]
        model_times = [s.model_forward_ms for s in self.step_profiles]
        moe_routing_times = [s.moe_routing_ms for s in self.step_profiles]
        moe_compute_times = [s.moe_compute_ms for s in self.step_profiles]
        comm_times = [s.communication_ms for s in self.step_profiles]

        return {
            "num_steps": len(self.step_profiles),
            "total_time_ms": {
                "mean": round(float(np.mean(total_times)), 2),
                "std": round(float(np.std(total_times)), 2),
                "min": round(float(np.min(total_times)), 2),
                "max": round(float(np.max(total_times)), 2),
            },
            "model_forward_ms": {
                "mean": round(float(np.mean(model_times)), 2),
                "std": round(float(np.std(model_times)), 2),
            },
            "moe_routing_ms": {
                "mean": round(float(np.mean(moe_routing_times)), 2),
            },
            "moe_compute_ms": {
                "mean": round(float(np.mean(moe_compute_times)), 2),
            },
            "communication_ms": {
                "mean": round(float(np.mean(comm_times)), 2),
            },
            "breakdown_pct": {
                "moe_routing": round(float(np.sum(moe_routing_times) / np.sum(total_times) * 100), 1) if np.sum(total_times) > 0 else 0,
                "moe_compute": round(float(np.sum(moe_compute_times) / np.sum(total_times) * 100), 1) if np.sum(total_times) > 0 else 0,
                "communication": round(float(np.sum(comm_times) / np.sum(total_times) * 100), 1) if np.sum(total_times) > 0 else 0,
            },
        }

    def save_report(self, filename: str = "profile_report.json"):
        """Save the profiling report to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        report = self.get_report()
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path


class RocprofWrapper:
    """
    Wrapper for rocprofv3 command-line profiler.

    Generates rocprofv3 commands to capture kernel traces for MoE inference.
    """

    @staticmethod
    def get_profile_command(
        script_command: str,
        output_dir: str = "./profiles",
        prefix: str = "moe",
        trace_hip: bool = True,
        trace_hsa: bool = False,
    ) -> str:
        """Generate a rocprofv3 command to profile the given script."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, prefix)

        parts = ["rocprofv3"]
        if trace_hip:
            parts.append("--hip-trace")
        if trace_hsa:
            parts.append("--hsa-trace")
        parts.append("--output-directory %s" % output_path)
        parts.append("-- %s" % script_command)

        return " ".join(parts)

    @staticmethod
    def generate_rocprof_script(
        model_name: str,
        python_command: str,
        output_dir: str = "./profiles",
    ) -> str:
        """Generate a shell script for rocprofv3 profiling."""
        script = """#!/bin/bash
# Auto-generated rocprofv3 profiling script for {model_name}
set -euo pipefail

OUTPUT_DIR="{output_dir}/{model_name}"
mkdir -p "$OUTPUT_DIR"

echo "Profiling {model_name} with rocprofv3..."
echo "Output: $OUTPUT_DIR"

# HIP API + kernel trace
rocprofv3 \\
    --hip-trace \\
    --output-directory "$OUTPUT_DIR" \\
    -- {python_command}

echo "Profile saved to $OUTPUT_DIR"
echo "View with: chrome://tracing or Perfetto UI"
""".format(
            model_name=model_name,
            output_dir=output_dir,
            python_command=python_command,
        )
        return script
