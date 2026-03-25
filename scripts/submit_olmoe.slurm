#!/bin/bash
#SBATCH --job-name=moe-olmoe
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/olmoe-%j.out
#SBATCH --error=results/logs/olmoe-%j.err

# OLMoE-1B-7B multi-GPU parallelism benchmark
# 6.9B total params (~14 GB), 64 experts, top-8 routing
# Sweeps 2 and 4 GPU configs across all placement strategies

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/results/logs"

echo "=============================="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME}"
echo "Model  : OLMoE-1B-7B"
echo "Start  : $(date)"
echo "=============================="

export NCCL_MIN_NCHANNELS=112
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

for NUM_GPUS in 2 4; do
    echo ""
    echo ">>> OLMoE-1B-7B  |  GPUs: ${NUM_GPUS}  |  $(date)"
    bash scripts/run_benchmark.sh \
        --model olmoe_1b_7b \
        --experiment multi_gpu \
        --num-gpus "${NUM_GPUS}"
done

echo ""
echo "=============================="
echo "Done: $(date)"
echo "=============================="
