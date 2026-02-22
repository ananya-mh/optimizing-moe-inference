#!/usr/bin/env bash
# Main benchmark entry point for MoE inference optimization.
#
# Usage:
#   bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu
#   bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment multi_gpu --num-gpus 8
#   bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu --profile
#
# Environment:
#   MODEL_DIR   - Path to model weights (default: ./models)
#   HF_TOKEN    - HuggingFace token (required for gated models)
#   RESULTS_DIR - Path to save results (default: ./results)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Auto-detect GPU vendor
detect_gpu_vendor() {
    if command -v rocm-smi &>/dev/null; then
        echo "amd"
    elif command -v nvidia-smi &>/dev/null; then
        echo "nvidia"
    else
        echo "unknown"
    fi
}

GPU_VENDOR=$(detect_gpu_vendor)
echo "=== MoE Inference Benchmark ==="
echo "GPU Vendor: ${GPU_VENDOR}"
echo "Project Dir: ${PROJECT_DIR}"
echo ""

# Set vendor-specific environment
if [ "${GPU_VENDOR}" = "amd" ]; then
    export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
    export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
    export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
    export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
    export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"
    echo "AMD environment configured"
elif [ "${GPU_VENDOR}" = "nvidia" ]; then
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
    echo "NVIDIA environment configured"
fi

# Run benchmark
cd "${PROJECT_DIR}"
python -m src.benchmark.runner "$@"
