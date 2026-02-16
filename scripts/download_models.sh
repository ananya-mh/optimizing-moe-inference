#!/usr/bin/env bash
# Download MoE model weights from HuggingFace.
#
# Usage:
#   export HF_TOKEN=your_token_here
#   bash scripts/download_models.sh [group]
#
# Groups: single_gpu, multi_gpu, multi_node, all (default: single_gpu)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
GROUP="${1:-single_gpu}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "  export HF_TOKEN=your_huggingface_token"
    exit 1
fi

echo "=== MoE Model Downloader ==="
echo "Model directory: ${MODEL_DIR}"
echo "Group: ${GROUP}"
echo ""

declare -A MODELS
MODELS[qwen_moe_a2.7b]="Qwen/Qwen1.5-MoE-A2.7B"
MODELS[olmoe_1b_7b]="allenai/OLMoE-1B-7B-0924"
MODELS[mixtral_8x7b]="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODELS[qwen2_57b_a14b]="Qwen/Qwen2-57B-A14B-Instruct"
MODELS[dbrx]="databricks/dbrx-instruct"
MODELS[deepseek_v3]="deepseek-ai/DeepSeek-V3"

# Select models by group
case "${GROUP}" in
    single_gpu)
        SELECTED=(qwen_moe_a2.7b olmoe_1b_7b mixtral_8x7b)
        ;;
    multi_gpu)
        SELECTED=(mixtral_8x7b qwen2_57b_a14b dbrx)
        ;;
    multi_node)
        SELECTED=(dbrx deepseek_v3)
        ;;
    all)
        SELECTED=(qwen_moe_a2.7b olmoe_1b_7b mixtral_8x7b qwen2_57b_a14b dbrx deepseek_v3)
        ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Available: single_gpu, multi_gpu, multi_node, all"
        exit 1
        ;;
esac

mkdir -p "${MODEL_DIR}"

for key in "${SELECTED[@]}"; do
    model_id="${MODELS[$key]}"
    echo "--- Downloading: ${model_id} ---"
    huggingface-cli download "${model_id}" \
        --local-dir "${MODEL_DIR}/${model_id}" \
        --token "${HF_TOKEN}" \
        --resume-download
    echo "  Done: ${model_id}"
    echo ""
done

echo "=== All downloads complete ==="
