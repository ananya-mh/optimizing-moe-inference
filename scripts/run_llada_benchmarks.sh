#!/bin/bash
# Run LLaDA benchmarks on AMD MI300X cluster
#
# Usage:
#   1. Allocate a compute node:
#      srun --partition=amd-rccl --nodes=1 --ntasks=1 \
#        --cpus-per-task=32 --gres=gpu:mi300x:8 \
#        --time=04:00:00 --job-name=llada-bench bash
#
#   2. Run this script:
#      bash /path/to/run_llada_benchmarks.sh
#
# Alternatively, build and use the Docker image:
#   docker build -f docker/Dockerfile.llada -t llada-rocm:latest .
#   (then use docker run as shown in Dockerfile.llada)

set -euo pipefail

MODEL_DIR="/shared_inference/ravgupta_models"
RESULTS_DIR="${MODEL_DIR}/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/inference"

# Use the cluster ROCm PyTorch image (pre-installed)
ROCM_IMAGE="rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch_release_2.4.0"

# Alternatively, use the vLLM image that already has everything:
VLLM_IMAGE="rocm/pytorch-private:miali_vllm_0.14.0rc2_ucx_develop_rixl_develop_20260126_retemadi_added_profile_pr18827"

DOCKER_BASE_ARGS=(
    --rm
    --device=/dev/kfd --device=/dev/dri
    --group-add video --group-add render
    --shm-size=32g --ipc=host
    --security-opt seccomp=unconfined
    -e VLLM_USE_TRITON_FLASH_ATTN=0
    -v "${MODEL_DIR}:/models"
    -v "${RESULTS_DIR}:/results"
    -v "${SCRIPT_DIR}:/app/src/inference"
)

mkdir -p "${RESULTS_DIR}"

echo "================================================"
echo "LLaDA Benchmark Suite on AMD MI300X"
echo "================================================"
echo ""

# --- LLaDA-8B (Dense Diffusion) ---
echo ">>> LLaDA-8B-Instruct (Dense) - Single GPU"
docker run "${DOCKER_BASE_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES=0 \
    "${ROCM_IMAGE}" \
    bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
    python3 /app/src/inference/llada_engine.py \
        --model-path /models/GSAI-ML/LLaDA-8B-Instruct \
        --gen-length 128 --steps 64 \
        --num-prompts 10 \
        --output-json /results/llada_8b_single.json"

echo ""
echo ">>> LLaDA-8B-Instruct (Dense) - Steps sweep"
for STEPS in 32 64 128; do
    echo "  Steps=${STEPS}"
    docker run "${DOCKER_BASE_ARGS[@]}" \
        -e HIP_VISIBLE_DEVICES=0 \
        "${ROCM_IMAGE}" \
        bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
        python3 /app/src/inference/llada_engine.py \
            --model-path /models/GSAI-ML/LLaDA-8B-Instruct \
            --gen-length 128 --steps ${STEPS} \
            --num-prompts 10 \
            --output-json /results/llada_8b_steps${STEPS}.json"
done

echo ""
echo ">>> LLaDA-8B-Instruct (Dense) - Gen length sweep"
for GEN_LEN in 64 128 256; do
    echo "  GenLen=${GEN_LEN}"
    docker run "${DOCKER_BASE_ARGS[@]}" \
        -e HIP_VISIBLE_DEVICES=0 \
        "${ROCM_IMAGE}" \
        bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
        python3 /app/src/inference/llada_engine.py \
            --model-path /models/GSAI-ML/LLaDA-8B-Instruct \
            --gen-length ${GEN_LEN} --steps 64 \
            --num-prompts 10 \
            --output-json /results/llada_8b_gen${GEN_LEN}.json"
done

# --- LLaDA-MoE-7B (MoE Diffusion) ---
echo ""
echo ">>> LLaDA-MoE-7B-A1B-Instruct (MoE) - Single GPU"
docker run "${DOCKER_BASE_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES=0 \
    "${ROCM_IMAGE}" \
    bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
    python3 /app/src/inference/llada_engine.py \
        --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
        --gen-length 128 --steps 64 \
        --num-prompts 10 \
        --output-json /results/llada_moe_single.json"

echo ""
echo ">>> LLaDA-MoE-7B-A1B-Instruct (MoE) - Steps sweep"
for STEPS in 32 64 128; do
    echo "  Steps=${STEPS}"
    docker run "${DOCKER_BASE_ARGS[@]}" \
        -e HIP_VISIBLE_DEVICES=0 \
        "${ROCM_IMAGE}" \
        bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
        python3 /app/src/inference/llada_engine.py \
            --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
            --gen-length 128 --steps ${STEPS} \
            --num-prompts 10 \
            --output-json /results/llada_moe_steps${STEPS}.json"
done

# --- Multi-GPU (distributed with RCCL) ---
echo ""
echo ">>> LLaDA-MoE-7B-A1B-Instruct - Distributed (2 GPUs)"
docker run "${DOCKER_BASE_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES=0,1 \
    "${ROCM_IMAGE}" \
    bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
    torchrun --nproc_per_node=2 /app/src/inference/llada_distributed.py \
        --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
        --gen-length 128 --steps 64 \
        --num-prompts 10 \
        --output-json /results/llada_moe_dist2.json"

echo ""
echo ">>> LLaDA-MoE-7B-A1B-Instruct - Distributed (4 GPUs)"
docker run "${DOCKER_BASE_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    "${ROCM_IMAGE}" \
    bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
    torchrun --nproc_per_node=4 /app/src/inference/llada_distributed.py \
        --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
        --gen-length 128 --steps 64 \
        --num-prompts 10 \
        --output-json /results/llada_moe_dist4.json"

echo ""
echo "================================================"
echo "All LLaDA benchmarks complete!"
echo "Results saved to: ${RESULTS_DIR}"
echo "================================================"
ls -la "${RESULTS_DIR}"/llada_*.json 2>/dev/null || echo "No results found"
