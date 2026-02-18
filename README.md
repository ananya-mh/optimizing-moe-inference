# Optimizing Mixture-of-Experts (MoE) Inference

A systematic framework for studying MoE inference optimization strategies including **expert-aware batching**, **expert placement** (co-located vs. distributed), and **scaling analysis** across single-GPU, multi-GPU, and multi-node configurations.

Targets both **AMD MI300X** (ROCm) and **NVIDIA** (CUDA) GPUs using **upstream vLLM** as the inference engine.

> **Paper**: SIEDS 2026 submission

## Overview

Mixture-of-Experts models activate only a subset of parameters per token, offering theoretical efficiency gains over dense models. However, deploying MoE models introduces unique bottlenecks:

- **Communication overhead** from dynamic expert routing (all-to-all dispatching)
- **GPU under-utilization** from uneven expert activation (load imbalance)
- **Irregular memory access** patterns that strain HBM bandwidth

This project provides a complete experimental framework to:

1. **Benchmark** MoE inference across 8 models spanning 6.9B to 671B parameters
2. **Profile** execution with torch profiler, rocprofv3 (AMD), and Nsight Systems (NVIDIA)
3. **Compare** placement strategies: Tensor Parallelism (TP), Expert Parallelism (EP), and hybrids
4. **Analyze** EP load balancing with Gini coefficient, imbalance ratio, and rebalancing recommendations
5. **Estimate** optimal expert placement using a lightweight CPU-based ML predictor
6. **Scale** experiments from 1 GPU to 4 nodes (32 GPUs)
7. **Study** both autoregressive and diffusion-based MoE architectures

## Inference Framework

We use **upstream vLLM** as the inference engine. On AMD MI300X:

| Component | Version | Purpose |
|-----------|---------|---------|
| **vLLM** | >= 0.8.0 | OpenAI-compatible serving with EP support |
| **ROCm** | >= 6.3.1 | GPU compute platform |
| **AITer** | Integrated | AMD-optimized MoE/attention kernels |
| **hipBLASLt** | Integrated | High-performance GEMM |
| **rocprofv3** | System | Hardware profiling (kernel traces, memory, HIP API) |

Docker image: `rocm/vllm-dev:main` (AMD's optimized vLLM build with AITer)
See [AMD's vLLM Docker guide](https://www.amd.com/en/developer/resources/technical-articles/how-to-use-prebuilt-amd-rocm-vllm-docker-image-with-amd-instinct-mi300x-accelerators.html).

## Models

| Model | Total | Active | Experts | Top-k | Type | Min GPUs |
|-------|-------|--------|---------|-------|------|----------|
| **LLaDA-MoE-7B** | 7B | 1.4B | 8 | 2 | Diffusion MoE | 1 |
| LLaDA-8B | 8B | 8B | 1 (dense) | - | Diffusion | 1 |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | Autoregressive MoE | 1 |
| OLMoE-1B-7B | 6.9B | 1.3B | 64 | 8 | Autoregressive MoE | 1 |
| Mixtral-8x7B | 46.7B | 12.9B | 8 | 2 | Autoregressive MoE | 1 |
| Qwen2-57B-A14B | 57.4B | 14B | 64 | 8 | Autoregressive MoE | 1 |
| DBRX | 132B | 36B | 16 | 4 | Autoregressive MoE | 2 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | Autoregressive MoE | 8 |

**LLaDA-MoE-7B** is the first open-source MoE diffusion LLM, enabling comparison of expert parallelism between autoregressive and diffusion inference paradigms.

## Project Structure

```
optimizing-moe-inference/
├── configs/
│   ├── models.yaml                    # Model registry (8 models)
│   └── experiments/                   # Experiment configurations
│       ├── single_gpu.yaml
│       ├── multi_gpu.yaml
│       └── multi_node.yaml
├── docker/
│   ├── Dockerfile.rocm                # AMD ROCm (upstream vLLM)
│   ├── Dockerfile.llada               # ROCm image for LLaDA diffusion models
│   ├── Dockerfile.cuda                # NVIDIA CUDA
│   └── docker-compose.yaml
├── src/
│   ├── inference/
│   │   ├── llada_engine.py            # Custom LLaDA inference (single GPU)
│   │   ├── llada_distributed.py       # Multi-GPU LLaDA with RCCL
│   │   ├── expert_parallel.py         # EP placement strategies + dispatch
│   │   └── profiler.py                # torch.profiler + rocprofv3 hooks
│   ├── benchmark/
│   │   ├── runner.py                  # Benchmark orchestrator (vLLM serve + bench)
│   │   ├── metrics.py                 # Metrics parsing (throughput, TTFT, ITL)
│   │   ├── report.py                  # Rich tables, CSV export
│   │   └── factorial_study.py         # Controlled factorial experiment design
│   ├── placement/
│   │   ├── strategies.py              # 5 placement strategies + memory estimation
│   │   ├── estimator.py               # Placement recommendation engine
│   │   ├── predictor.py               # CPU-based ML predictor (RandomForest)
│   │   └── load_balancing.py          # EP load balance analysis + optimization
│   ├── profiling/
│   │   ├── torch_profiler.py          # Torch profiler (both platforms)
│   │   ├── rocm_profiler.py           # rocprofv3 wrapper (AMD)
│   │   └── nvidia_profiler.py         # Nsight Systems wrapper (NVIDIA)
│   └── utils/
│       ├── config.py                  # YAML config loader, GPU vendor detection
│       └── gpu_info.py                # GPU info (rocm-smi / nvidia-smi)
├── scripts/
│   ├── download_models.sh             # Model downloader (HF_TOKEN from env)
│   ├── run_benchmark.sh               # Main entry point (auto-detects GPU)
│   ├── run_llada_benchmarks.sh        # LLaDA sweep runner (steps, gen length)
│   ├── run_profiling.sh               # Profiling (torch/rocprof/nsight)
│   └── setup_env.sh                   # Environment setup
├── analysis/
│   ├── plot_results.py                # Throughput/latency plots
│   ├── plot_load_balance.py           # EP load balance heatmaps
│   └── generate_tables.py             # LaTeX tables for paper
├── experiments/                       # Per-experiment notes
├── results/                           # Output (gitignored)
├── Report1.md                         # Benchmark results from MI300X experiments
├── requirements.txt
├── setup.py
└── LICENSE                            # MIT
```

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/ananya-mh/optimizing-moe-inference.git
cd optimizing-moe-inference

# Set up environment (auto-detects AMD vs NVIDIA)
source scripts/setup_env.sh

# Set HuggingFace token and model directory
export HF_TOKEN=your_token_here
export MODEL_DIR=/path/to/models  # default: ./models
```

### 2. Download Models

```bash
# Single-GPU models (LLaDA-MoE, Qwen-MoE, OLMoE, Mixtral, LLaDA-8B)
bash scripts/download_models.sh single_gpu

# Diffusion LLM models only
bash scripts/download_models.sh diffusion_llm

# All models
bash scripts/download_models.sh all
```

### 3. Run Benchmarks

```bash
# Single-GPU baseline
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu

# Multi-GPU with expert parallelism
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment multi_gpu --strategy ep_only

# With torch profiling
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu --profile

# Dry run (shows commands without executing)
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu --dry-run
```

### 4. EP Load Balance Analysis

```bash
# Run load balance simulation and analysis
python -m src.placement.load_balancing

# Generate load balance visualizations
python analysis/plot_load_balance.py
```

### 5. Factorial Experiment Design

```bash
# Generate the full factorial design matrix
python -m src.benchmark.factorial_study
```

### 6. Using Docker

```bash
# AMD MI300X
docker build -f docker/Dockerfile.rocm -t moe-opt:rocm .
docker run --device /dev/kfd --device /dev/dri --group-add video \
    -e HF_TOKEN=$HF_TOKEN -v $MODEL_DIR:/models:ro \
    -it moe-opt:rocm

# NVIDIA
docker build -f docker/Dockerfile.cuda -t moe-opt:cuda .
docker run --gpus all -e HF_TOKEN=$HF_TOKEN -v $MODEL_DIR:/models:ro \
    -it moe-opt:cuda
```

### 7. Analyze Results

```bash
python analysis/plot_results.py           # Throughput/latency plots
python analysis/plot_load_balance.py      # EP load balance heatmaps
python analysis/generate_tables.py        # LaTeX tables for paper
```

## Experiment Design

### Experiment 1: Single-GPU Baseline
- **Goal**: Establish per-model inference characteristics
- **Models**: LLaDA-MoE-7B, Qwen-MoE-A2.7B, OLMoE-1B-7B, Mixtral-8x7B, LLaDA-8B
- **Sweep**: Workloads (short/medium/long) x Concurrency (1-64)
- **Metrics**: Throughput (tok/s), TTFT, ITL, GPU memory, CU utilization

### Experiment 2: Expert Placement (Factorial Study)
- **Goal**: Isolate contribution of each strategy via controlled factorial design
- **Strategies**: TP-only, EP-only, TP+EP hybrid, DP+EP
- **Factors**: Model x GPUs x Strategy x Batching x Queue depth x Workload
- **Analysis**: ANOVA to determine which factors most impact throughput

### Experiment 3: Expert-Aware Batching
- **Goal**: Optimize queue depth for maximum GPU (CU) occupancy
- **Method**: Sweep queue depth [4, 8, 16, 32, 64, 128, 256]
- **Metrics**: Throughput, CU occupancy, expert wait time

### Experiment 4: Multi-GPU Scaling
- **Goal**: Scaling efficiency from 1 to 8 GPUs
- **All2All backends**: allgather_reducescatter, pplx
- **AMD-specific**: AITer fused MoE kernels, hipBLASLt GEMM

### Experiment 5: Multi-Node Scaling
- **Goal**: Cross-node EP with RDMA (2-4 nodes, CX-7 NICs)
- **Models**: DBRX (132B), DeepSeek-V3 (671B)

## EP Load Balancing Analysis

The load balancing module (`src/placement/load_balancing.py`) provides:

- **Imbalance metrics**: Load imbalance ratio, coefficient of variation, Gini coefficient
- **Hot/cold expert detection**: Identifies over- and under-utilized experts
- **Routing simulation**: Uniform, Zipfian, and skewed distributions
- **Rebalancing recommendations**: Expert replication, migration, greedy re-mapping
- **Visualization**: GPU load bar charts, expert activation heatmaps

```python
from placement.load_balancing import run_load_balance_study, print_load_balance_summary
from utils.config import get_model_config

model = get_model_config("mixtral_8x7b")
reports = run_load_balance_study(model, num_gpus=8)
print_load_balance_summary(reports["zipf"])
```

## Placement Estimation Framework

```python
from placement.estimator import recommend_placement
from utils.config import get_model_config

model = get_model_config("mixtral_8x7b")
rec = recommend_placement(model, num_gpus=8, gpu_memory_gb=192.0)
print(f"Strategy: {rec.strategy_name}")
print(f"Memory/GPU: {rec.memory_per_gpu_gb:.1f} GB")
print(f"Queue depth: {rec.estimated_queue_depth}")
```

## AMD-Specific Optimizations

| Variable | Purpose |
|----------|---------|
| `VLLM_ROCM_USE_AITER=1` | Enable AITer high-performance kernels |
| `VLLM_ROCM_USE_AITER_MOE=1` | AITer fused MoE kernels (top-k routing, sorting) |
| `HIP_FORCE_DEV_KERNARG=1` | Faster HIP kernel argument passing |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | Prefer hipBLASLt for GEMM operations |
| `NCCL_MIN_NCHANNELS=112` | Optimize multi-GPU NCCL collectives |
| `VLLM_USE_TRITON_FLASH_ATTN=0` | Use CK-based FlashAttention (faster on MI300X) |

## Profiling

| Tool | Platform | Command |
|------|----------|---------|
| Torch Profiler | Both | `bash scripts/run_profiling.sh --model MODEL --torch` |
| rocprofv3 | AMD | `bash scripts/run_profiling.sh --model MODEL --rocprof` |
| Nsight Systems | NVIDIA | `bash scripts/run_profiling.sh --model MODEL --nsight` |

View torch traces at [Perfetto UI](https://ui.perfetto.dev) or `chrome://tracing`.

## Configuration

All configs in YAML under `configs/`. Environment variables (set at runtime, never hardcoded):

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (none) | HuggingFace authentication token |
| `MODEL_DIR` | `./models` | Path to model weights |
| `RESULTS_DIR` | `./results` | Path for benchmark output |

## Custom LLaDA Inference Engine

LLaDA diffusion models are **not supported by vLLM** due to their non-autoregressive masked diffusion architecture. We built a custom inference engine:

| Module | Purpose |
|--------|---------|
| `src/inference/llada_engine.py` | Single-GPU LLaDA inference with block-based denoising |
| `src/inference/llada_distributed.py` | Multi-GPU distributed inference with RCCL |
| `src/inference/expert_parallel.py` | Expert placement strategies + all-to-all dispatch |
| `src/inference/profiler.py` | Profiling for torch.profiler and rocprofv3 |
| `docker/Dockerfile.llada` | Docker image for LLaDA on ROCm |
| `scripts/run_llada_benchmarks.sh` | Automated sweep runner |

See [Report1.md](Report1.md) for full benchmark results.

## Abstract Alignment

This codebase implements the full methodology described in the SIEDS 2026 abstract:

| Abstract Claim | Implementation |
|----------------|----------------|
| Expert-aware batching with tunable queue depth | `src/placement/estimator.py` - `estimate_queue_depth()` |
| Static vs distributed expert placement | `src/placement/strategies.py` - 5 strategies |
| Placement estimation framework | `src/placement/estimator.py` - `recommend_placement()` |
| Lightweight CPU-based ML predictor | `src/placement/predictor.py` - RandomForest on CPU |
| Controlled factorial study | `src/benchmark/factorial_study.py` |
| rocprof + Nsight profiling | `src/profiling/rocm_profiler.py`, `nvidia_profiler.py` |
| Memory bandwidth + CU occupancy analysis | `src/benchmark/metrics.py`, `gpu_info.py` |
| Multi-GPU and multi-node scaling (up to 4 nodes) | `configs/experiments/multi_node.yaml` |
| EP load balance analysis | `src/placement/load_balancing.py` |
| Diffusion MoE inference (LLaDA) | `src/inference/llada_engine.py`, `llada_distributed.py` |
| Expert Parallelism with RCCL | `src/inference/expert_parallel.py` |

## License

MIT License. See [LICENSE](LICENSE).
