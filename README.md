# Optimizing Mixture-of-Experts (MoE) Inference

A systematic framework for studying MoE inference optimization strategies including **expert-aware batching**, **expert placement** (co-located vs. distributed), and **scaling analysis** across single-GPU, multi-GPU, and multi-node configurations.

Targets both **AMD MI300X** (ROCm) and **NVIDIA** (CUDA) GPUs.

## Overview

Mixture-of-Experts models activate only a subset of parameters per token, offering theoretical efficiency gains over dense models. However, deploying MoE models introduces unique bottlenecks:

- **Communication overhead** from dynamic expert routing (all-to-all)
- **GPU under-utilization** from uneven expert activation
- **Irregular memory access** patterns that strain HBM bandwidth

This project provides tools to:
1. **Benchmark** MoE inference across 6 models spanning 6.9B to 671B parameters
2. **Profile** execution with torch profiler, rocprofv3, and Nsight Systems
3. **Compare** placement strategies: Tensor Parallelism, Expert Parallelism, and hybrids
4. **Estimate** optimal expert placement using a lightweight CPU-based ML predictor
5. **Scale** experiments from 1 GPU to 4 nodes (32 GPUs)

## Models

| Model | Total Params | Active/Token | Experts | Top-k | Min GPUs |
|-------|-------------|-------------|---------|-------|----------|
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | 1 |
| OLMoE-1B-7B | 6.9B | 1.3B | 64 | 8 | 1 |
| Mixtral-8x7B | 46.7B | 12.9B | 8 | 2 | 1 |
| Qwen2-57B-A14B | 57.4B | 14.0B | 64 | 8 | 1 |
| DBRX | 132B | 36B | 16 | 4 | 2 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | 8 |

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/ananya-mh/optimizing-moe-inference.git
cd optimizing-moe-inference
source scripts/setup_env.sh
export HF_TOKEN=your_token_here
```

### 2. Download Models

```bash
bash scripts/download_models.sh single_gpu
bash scripts/download_models.sh all
```

### 3. Run Benchmarks

```bash
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment multi_gpu --num-gpus 8 --strategy ep_only
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu --profile
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu --dry-run
```

### 4. Docker

```bash
docker build -f docker/Dockerfile.rocm -t moe-opt:rocm .
docker build -f docker/Dockerfile.cuda -t moe-opt:cuda .
```

### 5. Analyze Results

```bash
python analysis/plot_results.py
python analysis/generate_tables.py
```

## Experiment Design

### Experiment 1: Single-GPU Baseline
- **Goal**: Establish per-model inference characteristics
- **Sweep**: Workloads (short/medium/long) x Concurrency (1-64)

### Experiment 2: Expert Placement
- **Goal**: Compare co-locate vs. distributed (TP-only, EP-only, TP+EP, DP+EP)

### Experiment 3: Expert-Aware Batching
- **Goal**: Optimize queue depth for maximum GPU occupancy

### Experiment 4: Multi-GPU Scaling
- **Goal**: Scaling efficiency from 1 to 8 GPUs

### Experiment 5: Multi-Node Scaling
- **Goal**: Cross-node expert parallelism (2-4 nodes)

## AMD-Specific Optimizations

| Variable | Purpose |
|----------|---------|
| `VLLM_ROCM_USE_AITER=1` | AITer high-performance kernels |
| `VLLM_ROCM_USE_AITER_MOE=1` | AITer fused MoE kernels |
| `HIP_FORCE_DEV_KERNARG=1` | Faster kernel argument passing |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | Prefer hipBLASLt for GEMM |
| `NCCL_MIN_NCHANNELS=112` | Multi-GPU collectives |

## Profiling

```bash
bash scripts/run_profiling.sh --model mixtral_8x7b --torch
bash scripts/run_profiling.sh --model mixtral_8x7b --rocprof
bash scripts/run_profiling.sh --model mixtral_8x7b --nsight
```

## Configuration

Environment variables (set at runtime, never hardcoded):
- `HF_TOKEN` - HuggingFace authentication token
- `MODEL_DIR` - Path to model weights
- `RESULTS_DIR` - Path for output results

## License

MIT License. See [LICENSE](LICENSE) for details.
