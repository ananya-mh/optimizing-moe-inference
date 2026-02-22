#!/usr/bin/env bash
# Environment setup for MoE inference optimization experiments.
#
# Usage:
#   source scripts/setup_env.sh
set -euo pipefail

echo "=== Setting up MoE Optimization Environment ==="

# Detect GPU vendor
if command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
    echo "Detected: AMD GPU (ROCm)"
elif command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    echo "Detected: NVIDIA GPU (CUDA)"
else
    echo "WARNING: No GPU management tool found"
    GPU_VENDOR="unknown"
fi

# Set common env vars
export MODEL_DIR="${MODEL_DIR:-./models}"
export RESULTS_DIR="${RESULTS_DIR:-./results}"

# GPU-specific env vars
if [ "${GPU_VENDOR}" = "amd" ]; then
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export HIP_FORCE_DEV_KERNARG=1
    export TORCH_BLAS_PREFER_HIPBLASLT=1
    export NCCL_MIN_NCHANNELS=112
    export VLLM_USE_TRITON_FLASH_ATTN=0
fi

# Create Python venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "=== Setup complete ==="
echo "GPU Vendor: ${GPU_VENDOR}"
echo "Model Dir:  ${MODEL_DIR}"
echo "Results:    ${RESULTS_DIR}"
echo ""
echo "Reminder: Set HF_TOKEN before downloading models:"
echo "  export HF_TOKEN=your_token_here"
