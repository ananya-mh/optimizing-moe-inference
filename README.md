# Optimizing Mixture-of-Experts (MoE) Inference

A systematic framework for studying MoE inference optimization strategies including **expert placement** (co-located vs. distributed) and **scaling analysis** across single-GPU and multi-GPU configurations.

Benchmarked on **NVIDIA A100 (80 GB)** GPUs using **upstream vLLM** as the inference engine.

> **Paper**: SIEDS 2026 submission

## Overview

Mixture-of-Experts models activate only a subset of parameters per token, offering theoretical efficiency gains over dense models. However, deploying MoE models introduces unique bottlenecks:

- **Communication overhead** from dynamic expert routing (all-to-all dispatching)
- **GPU under-utilization** from uneven expert activation (load imbalance)
- **Irregular memory access** patterns that strain HBM bandwidth

This project provides a complete experimental framework to:

1. **Benchmark** MoE inference across 5 models spanning 6.9B to 46.7B parameters
2. **Profile** execution with PyTorch Profiler and Nsight Systems
3. **Compare** placement strategies: Tensor Parallelism (TP), Expert Parallelism (EP), and hybrids
4. **Analyze** EP load balancing with Gini coefficient, imbalance ratio, and rebalancing recommendations
5. **Estimate** optimal expert placement using a lightweight CPU-based ML predictor
6. **Scale** experiments from 1 to 4 GPUs
7. **Study** both autoregressive and diffusion-based MoE architectures

## Inference Framework

We use **upstream vLLM** in server mode as the inference engine on NVIDIA A100:

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA A100 (80 GB HBM2e), PCIe interconnect |
| **vLLM** | Server mode (`vllm serve` + `vllm bench serve`) |
| **Profiling** | Nsight Systems 2024.7.1, PyTorch Profiler |
| **GPU allocation** | 1x A100 (baselines), 2x A100 (EP/TP), 4x A100 (hybrid TP+EP) |

## Models

| Model | HF ID | Total | Active | Experts | Top-k | Type |
|-------|--------|-------|--------|---------|-------|------|
| OLMoE-1B-7B | allenai/OLMoE-1B-7B-0924 | 6.9B | 1.3B | 64 | 8 | Autoregressive MoE |
| Qwen1.5-MoE-A2.7B | Qwen/Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | Autoregressive MoE |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-Instruct-v0.1 | 46.7B | 12.9B | 8 | 2 | Autoregressive MoE |
| LLaDA-8B | GSAI-ML/LLaDA-8B-Instruct | 8.0B | 8.0B | 1 (dense) | - | Diffusion Dense |
| LLaDA-MoE-7B | inclusionAI/LLaDA-MoE-7B-A1B-Instruct | 7.0B | 1.4B | 64 | 8 | Diffusion MoE |

The three autoregressive MoE models (OLMoE, Qwen, Mixtral) were used for both single-GPU and multi-GPU experiments. The two LLaDA models were benchmarked on single GPU only, serving as a dense-vs-MoE comparison for diffusion LLMs.

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
├── docs/
│   └── Report.md                      # Full experimental report
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

## Experiments

All benchmark data is consolidated in `results/tables/master_results_clean.csv` (284 rows). See **[experiment_starter.md](experiment_starter.md)** for the end-to-end guide with commands and data capture tables.

| Experiment | Setup | Status |
|------------|-------|--------|
| **1. Single-GPU Baselines** | 1x A100, 3 workloads x 7 concurrency levels, all 5 models | Complete |
| **2. Multi-GPU Strategy Comparison** | EP (2 GPU), TP (2 GPU), Hybrid TP+EP (4 GPU), 3 MoE models | Complete |
| **3. Expert Load Balance Analysis** | Activation heatmaps + GPU load distribution for all MoE models | Complete |
| **4. Kernel-Level Profiling** | PyTorch Profiler with vLLM offline `LLM` class | In progress |

### Workloads

| Workload | Prompts | Input Tokens | Output Tokens |
|----------|---------|-------------|--------------|
| decode_heavy | 100 | 128 | 128 |
| prefill_heavy | 50 | 512 | 256 |
| balanced | 20 | 1024 | 512 |

### Quick Start

```bash
# Download models and run single-GPU baselines
bash scripts/download_models.sh single_gpu
bash scripts/run_benchmark.sh --model mixtral_8x7b --experiment single_gpu
bash scripts/run_llada_benchmarks.sh
```

### Experiment Details

**Experiment 1 — Single-GPU Baselines**: Sweep 3 workloads x 7 concurrency levels (1, 2, 4, 8, 16, 32, 64) for OLMoE, Qwen-MoE, Mixtral, LLaDA-MoE, LLaDA-8B. Metrics: throughput, TTFT, ITL.

**Experiment 2 — Multi-GPU Strategy Comparison**: Three strategies compared across 3 autoregressive MoE models, 3 workloads, and 7 concurrency levels. EP only (2 GPU, TP=1), TP only (2 GPU, TP=2), Hybrid TP+EP (4 GPU, TP=2 + EP). All-to-all backend: `allgather_reducescatter`.

**Experiment 3 — Expert Load Balance Analysis**: Expert activation heatmaps and GPU load distribution charts for all MoE models, analyzing routing skew and its impact on EP efficiency.

**Experiment 4 — Kernel-Level Profiling** (in progress): Single-GPU execution traces to characterize each model as memory-bound or compute-bound using PyTorch Profiler with vLLM's offline `LLM` class.

See [docs/Report.md](docs/Report.md) for the full experimental report.

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

## Profiling

| Tool | Command |
|------|---------|
| PyTorch Profiler | `bash scripts/run_profiling.sh --model MODEL --torch` |
| Nsight Systems | `bash scripts/run_profiling.sh --model MODEL --nsight` |

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
| `src/inference/profiler.py` | Profiling for torch.profiler |
| `docker/Dockerfile.llada` | Docker image for LLaDA |
| `scripts/run_llada_benchmarks.sh` | Automated sweep runner |

See [docs/Report.md](docs/Report.md) for full benchmark results.

## Key Findings

- **Active parameters predict throughput, not total parameters**: Qwen (2.7B active) outperforms Mixtral (12.9B active) on single GPU despite fewer total params.
- **TP generally outperforms EP at this scale**: On 2x A100 with PCIe, TP wins for Mixtral (+24%) and OLMoE (+21%). Exception: Qwen's 60 experts benefit more from EP (+32%).
- **Hybrid TP+EP is counterproductive**: 4-GPU hybrid yields only ~20% of 2-GPU pure strategy throughput due to compounding communication overhead over PCIe.
- **MoE overhead in diffusion LLMs**: LLaDA-MoE is ~10x slower than the dense LLaDA-8B, motivating expert parallelism for diffusion models.

## Code Structure

| Component | Implementation |
|-----------|----------------|
| Placement strategies | `src/placement/strategies.py` - 5 strategies |
| Placement estimation | `src/placement/estimator.py` - `recommend_placement()` |
| CPU-based ML predictor | `src/placement/predictor.py` - RandomForest on CPU |
| EP load balance analysis | `src/placement/load_balancing.py` |
| Profiling | `src/profiling/torch_profiler.py`, `nvidia_profiler.py` |
| Diffusion MoE inference (LLaDA) | `src/inference/llada_engine.py`, `llada_distributed.py` |
| Expert Parallelism dispatch | `src/inference/expert_parallel.py` |

## License

MIT License. See [LICENSE](LICENSE).
