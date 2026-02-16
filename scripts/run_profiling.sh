#!/usr/bin/env bash
# Profiling script for MoE inference.
#
# Usage:
#   bash scripts/run_profiling.sh --model mixtral_8x7b [--rocprof] [--nsight] [--torch]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL="${1:---help}"
USE_ROCPROF=false
USE_NSIGHT=false
USE_TORCH=true
PROFILE_DIR="${RESULTS_DIR:-./results}/profiles/$(date +%Y%m%d_%H%M%S)"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --rocprof) USE_ROCPROF=true; shift ;;
        --nsight) USE_NSIGHT=true; shift ;;
        --torch) USE_TORCH=true; shift ;;
        --output-dir) PROFILE_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "${PROFILE_DIR}"
echo "=== MoE Inference Profiling ==="
echo "Model: ${MODEL}"
echo "Output: ${PROFILE_DIR}"

# Torch profiler (via vLLM built-in)
if [ "${USE_TORCH}" = true ]; then
    echo "--- Torch Profiler ---"
    export VLLM_TORCH_PROFILER_DIR="${PROFILE_DIR}/torch"
    mkdir -p "${VLLM_TORCH_PROFILER_DIR}"
    cd "${PROJECT_DIR}"
    python -m src.benchmark.runner --model "${MODEL}" --experiment single_gpu --profile
fi

# ROCm profiler
if [ "${USE_ROCPROF}" = true ]; then
    echo "--- ROCm Profiler (rocprofv3) ---"
    if command -v rocprofv3 &>/dev/null; then
        rocprofv3 --hip-trace --kernel-trace --memory-copy-trace             -o "${PROFILE_DIR}/rocm/trace" --output-format csv             -- python -m src.benchmark.runner --model "${MODEL}" --experiment single_gpu
    else
        echo "rocprofv3 not found, skipping"
    fi
fi

# Nsight Systems
if [ "${USE_NSIGHT}" = true ]; then
    echo "--- Nsight Systems ---"
    if command -v nsys &>/dev/null; then
        nsys profile -o "${PROFILE_DIR}/nvidia/profile"             --trace cuda,nvtx --force-overwrite=true             python -m src.benchmark.runner --model "${MODEL}" --experiment single_gpu
    else
        echo "nsys not found, skipping"
    fi
fi

echo "=== Profiling complete ==="
echo "Results in: ${PROFILE_DIR}"
