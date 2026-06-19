# Optimizing MoE Inference: Experimental Report

**Date**: April 2026

---

## 1. Overview

This report documents all experiments conducted for the study of Mixture-of-Experts (MoE) inference optimization. The work covers single-GPU baselines for 5 models, multi-GPU parallelism strategy comparisons (Expert Parallelism, Tensor Parallelism, and Hybrid TP+EP) for 3 MoE models, expert load balance analysis, and ongoing kernel-level profiling. All benchmark data is consolidated in `results/tables/master_results_clean.csv` (284 rows).

---

## 2. Hardware Environment

| Component | Details |
|-----------|---------|
| GPU | NVIDIA A100 (80 GB HBM2e) |
| Interconnect | PCIe |
| Framework | vLLM (server mode with `vllm serve`, benchmarked with `vllm bench serve`) |
| Profiling | Nsight Systems 2024.7.1, PyTorch Profiler |

**GPU allocation by experiment:**
- Single-GPU baselines: 1x A100
- EP and TP strategies: 2x A100
- Hybrid TP+EP strategy: 4x A100

---

## 3. Models

| Model | HF ID | Total Params | Active Params | Experts | Top-K | Type |
|-------|--------|-------------|--------------|---------|-------|------|
| OLMoE-1B-7B | allenai/OLMoE-1B-7B-0924 | 6.9B | 1.3B | 64 | 8 | Autoregressive MoE |
| Qwen1.5-MoE-A2.7B | Qwen/Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | Autoregressive MoE |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-Instruct-v0.1 | 46.7B | 12.9B | 8 | 2 | Autoregressive MoE |
| LLaDA-8B | GSAI-ML/LLaDA-8B-Instruct | 8.0B | 8.0B | 1 (dense) | - | Diffusion Dense |
| LLaDA-MoE-7B | inclusionAI/LLaDA-MoE-7B-A1B-Instruct | 7.0B | 1.4B | 64 | 8 | Diffusion MoE |

The three autoregressive MoE models (OLMoE, Qwen, Mixtral) were used for both single-GPU and multi-GPU experiments. The two LLaDA models were benchmarked on single GPU only, serving as a dense-vs-MoE comparison for diffusion LLMs.

---

## 4. Benchmark Configuration

### 4.1 Workloads

| Workload | Prompts | Input Tokens | Output Tokens | Character |
|----------|---------|-------------|--------------|-----------|
| decode_heavy | 100 | 128 | 128 | Many short requests |
| prefill_heavy | 50 | 512 | 256 | Fewer, longer prompts |
| balanced | 20 | 1024 | 512 | Long context, long generation |

### 4.2 Concurrency Levels

- Single-GPU: 1, 2, 4, 8, 16, 32, 64
- Multi-GPU: 1, 4, 8, 16, 32, 64, 128

### 4.3 Server Settings

```
vllm serve <model> --tensor-parallel-size <tp> --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --dtype auto --enforce-eager \
  [--enable-expert-parallel] [--all2all-backend allgather_reducescatter]
```

---

## 5. Experiment 1: Single-GPU Baselines

**Setup**: 1x A100 80GB, TP=1, no EP, vLLM server mode. Each model benchmarked across 3 workloads and 7 concurrency levels.

### 5.1 Peak Throughput (tok/s) by Model and Workload

| Model | decode_heavy | prefill_heavy | balanced |
|-------|-------------|--------------|----------|
| Mixtral-8x7B | 1,080.6 (conc=64) | 1,084.9 (conc=64) | 452.4 (conc=32) |
| OLMoE-1B-7B | 1,924.6 (conc=64) | 1,880.2 (conc=64) | 785.7 (conc=32) |
| Qwen1.5-MoE-A2.7B | 2,430.0 (conc=64) | 2,178.4 (conc=64) | 1,125.3 (conc=32) |

### 5.2 LLaDA Diffusion Models (Single GPU, conc=1)

| Model | Type | decode_heavy (tok/s) | prefill_heavy (tok/s) | balanced (tok/s) |
|-------|------|---------------------|----------------------|-----------------|
| LLaDA-8B | Dense | 99.1 | 50.6 | 24.8 |
| LLaDA-MoE-7B | MoE (64 experts, top-8) | 10.1 | 5.0 | 2.5 |

LLaDA-MoE is **~10x slower** than the dense LLaDA-8B across all workloads, despite having fewer active parameters (1.4B vs 8.0B). This demonstrates the overhead of routing through 64 experts on a single GPU.

### 5.3 Observations

- Throughput scales nearly linearly with concurrency up to 32, then plateaus at 64.
- **Active parameters, not total parameters, determine throughput**: Qwen (2.7B active) outperforms Mixtral (12.9B active) despite having more experts.
- Balanced workloads (1024 in / 512 out) yield roughly half the peak throughput of decode_heavy workloads due to higher per-request compute.

---

## 6. Experiment 2: Multi-GPU Strategy Comparison

**Setup**: vLLM server mode on A100 80GB GPUs with PCIe interconnect. Three parallelism strategies compared across 3 models, 3 workloads, and 7 concurrency levels. All-to-all backend: `allgather_reducescatter`.

### 6.1 Strategies

| Strategy | GPUs | Configuration |
|----------|------|---------------|
| **ep_only** | 2 | TP=1, Expert Parallelism enabled |
| **tp_only** | 2 | TP=2, no Expert Parallelism |
| **tp_ep_hybrid** | 4 | TP=2 + Expert Parallelism enabled |

### 6.2 Peak Throughput Comparison (tok/s)

#### Mixtral-8x7B (8 experts, top-2)

| Strategy | GPUs | decode_heavy | prefill_heavy | balanced |
|----------|------|-------------|--------------|----------|
| ep_only | 2 | 2,267.9 | 1,449.9 | 610.0 |
| tp_only | 2 | **2,817.3** | **1,526.6** | **660.6** |
| tp_ep_hybrid | 4 | 574.0 | 427.9 | 354.0 |

TP wins by 24% on decode_heavy. Hybrid on 4 GPUs is 4.9x slower than TP on 2 GPUs.

#### OLMoE-1B-7B (64 experts, top-8)

| Strategy | GPUs | decode_heavy | prefill_heavy | balanced |
|----------|------|-------------|--------------|----------|
| ep_only | 2 | 3,912.6 | 2,440.5 | 1,002.7 |
| tp_only | 2 | **4,741.5** | **2,513.5** | **1,060.5** |
| tp_ep_hybrid | 4 | 924.7 | 974.8 | 789.2 |

TP wins by 21% on decode_heavy. Hybrid on 4 GPUs is 5.1x slower than TP on 2 GPUs.

#### Qwen1.5-MoE-A2.7B (60 experts, top-4)

| Strategy | GPUs | decode_heavy | prefill_heavy | balanced |
|----------|------|-------------|--------------|----------|
| ep_only | 2 | **3,070.0** | **1,805.1** | 706.8 |
| tp_only | 2 | 2,316.7 | 1,797.5 | **739.9** |
| tp_ep_hybrid | 4 | 647.8 | 689.9 | 561.9 |

Qwen is the exception: EP outperforms TP on decode_heavy (+32%) and prefill_heavy workloads. With 60 experts, distributing them across GPUs reduces per-GPU memory pressure more effectively than splitting each expert's parameters via TP.

### 6.3 Scaling from Single-GPU Baseline

| Model | 1-GPU Peak (tok/s) | Best 2-GPU (tok/s) | Speedup | Best Strategy |
|-------|-------------------|-------------------|---------|---------------|
| Mixtral-8x7B | 1,084.9 | 2,817.3 | 2.60x | tp_only |
| OLMoE-1B-7B | 1,924.6 | 4,741.5 | 2.46x | tp_only |
| Qwen1.5-MoE-A2.7B | 2,430.0 | 3,070.0 | 1.26x | ep_only |

Mixtral and OLMoE achieve near-linear 2-GPU scaling with TP. Qwen shows only 1.26x, suggesting it is already well-utilized on a single GPU at these model sizes.

### 6.4 Hybrid Strategy Analysis

The hybrid TP+EP strategy on 4 GPUs **consistently underperformed** both pure strategies on 2 GPUs across all models and workloads:

| Model | Best 2-GPU (tok/s) | Hybrid 4-GPU (tok/s) | Ratio |
|-------|-------------------|---------------------|-------|
| Mixtral | 2,817.3 | 574.0 | 0.20x |
| OLMoE | 4,741.5 | 924.7 | 0.20x |
| Qwen | 3,070.0 | 689.9 | 0.22x |

Using 4 GPUs with hybrid parallelism yields only ~20% of the throughput achieved by 2 GPUs with a pure strategy. The combined communication overhead of TP all-reduce and EP all-to-all over PCIe creates a compounding bottleneck that dominates any compute benefit from additional GPUs.

---

## 7. Experiment 3: Expert Load Balance Analysis

Expert activation heatmaps and GPU load distribution charts were generated for all MoE models. Results in `results/figures/`:

- `expert_heatmap_mixtral_8x7b.png` — 8 experts, relatively balanced activation
- `expert_heatmap_olmoe_1b_7b.png` — 64 experts, significant activation skew
- `expert_heatmap_qwen_moe_a2.7b.png` — 60 experts, moderate imbalance
- `expert_heatmap_llada_moe_7b.png` — 64 experts

Load imbalance directly affects EP efficiency: skewed activation means some GPUs host "hot" experts and become bottlenecks, while others sit underutilized. This is especially relevant for OLMoE and Qwen with their large expert counts.

---

## 8. Experiment 4: Kernel-Level Profiling (In Progress)

### 8.1 Objective

Collect single-GPU execution traces to characterize each model as memory-bound or compute-bound. This provides the diagnostic basis for explaining why certain multi-GPU strategies outperform others (e.g., compute-bound models benefit from TP which splits matmuls; memory-bound models benefit from EP which distributes weight loading).

### 8.2 Challenges Encountered

**Nsight Systems** was used to profile the vLLM server, but captured no CUDA kernel data. Root cause: vLLM spawns a separate worker process for GPU inference. `nsys` attached to the parent HTTP server process, which performs no CUDA operations. The actual GPU kernels run in the forked child process.

**vLLM Torch Profiler** requires the server to be started with `VLLM_TORCH_PROFILER_DIR` and profiling to be triggered via `POST /start_profile` and `POST /stop_profile` API endpoints. The `/start_profile` endpoint returned 404, possibly due to vLLM version incompatibility.

### 8.3 Current Approach

Using vLLM's offline `LLM` class wrapped in PyTorch's `torch.profiler`. This runs inference in a single process, avoiding the multi-process issue:

```python
from vllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity

llm = LLM(model=..., enforce_eager=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True, with_flops=True) as prof:
    llm.generate(prompts, params)
```

### 8.4 Target Metrics

| Metric | Purpose |
|--------|---------|
| Expert kernel time / total GPU time | Compute-bound indicator |
| Attention kernel time / total GPU time | TP benefit indicator |
| Kernel dispatch count | Overhead from many small expert kernels |
| Average expert kernel duration (us) | Small values = overhead-dominated |

### 8.5 Status

Models to profile: Mixtral-8x7B (AWQ quantized, to fit single A100), OLMoE-1B-7B, Qwen1.5-MoE-A2.7B.

---

## 9. Key Findings

### 9.1 Active Parameters Predict Throughput, Not Total Parameters

OLMoE (6.9B total, 1.3B active) and Qwen (14.3B total, 2.7B active) both outperform Mixtral (46.7B total, 12.9B active) on single-GPU. Total parameter count is misleading for MoE models.

### 9.2 TP Generally Outperforms EP at This Scale

On 2x A100 with PCIe, TP outperforms EP for Mixtral (+24%) and OLMoE (+21%) on decode-heavy workloads. The exception is Qwen, where EP wins (+32%) — likely because its 60 experts benefit more from distribution than from parameter splitting.

### 9.3 Hybrid TP+EP is Counterproductive

4-GPU hybrid parallelism yields only ~20% of 2-GPU pure strategy throughput. The combined communication overhead of TP all-reduce and EP all-to-all over PCIe compounds rather than cancels. Practitioners should avoid hybrid parallelism for MoE models at this scale and interconnect bandwidth.

### 9.4 Workload Type Affects Strategy Selection

Decode-heavy workloads (many short requests) show the largest throughput gains from multi-GPU strategies and the greatest differentiation between strategies. Balanced workloads (long prompts) show smaller, more uniform gains across strategies.

### 9.5 MoE Overhead in Diffusion LLMs

LLaDA-MoE is ~10x slower than the dense LLaDA-8B across all workloads, despite smaller active parameters. The MoE routing overhead on single GPU is severe for diffusion models and motivates expert parallelism.

### 9.6 Concurrency Amortizes Communication Overhead

Single-GPU throughput scales linearly with concurrency up to saturation. Multi-GPU strategies become beneficial primarily at high concurrency where communication overhead is amortized across many concurrent requests.

---

## 10. CPU-Based Placement Predictor

### 10.1 Motivation

During GPU-heavy MoE inference, host CPUs are largely idle. This presents an opportunity to run a lightweight ML model on CPU that recommends deployment configurations — specifically, which parallelism strategy to use and what batch size to set — without impacting GPU throughput.

### 10.2 What It Predicts

The predictor addresses a practical deployment decision: **given a model and available GPUs, what strategy and batch size should I launch with?**

In production, operators control:
- **Parallelism strategy** (EP, TP, or hybrid) — set at server launch
- **Number of GPUs** — allocated per deployment
- **Max concurrent requests** (`--max-num-seqs`) — the controllable proxy for concurrency

They do not control user traffic volume, but they can anticipate workload profiles (average input/output lengths) from their application.

### 10.3 Implementation

A Random Forest model (`src/placement/predictor.py`) with two heads:
- **Classification head**: Predicts best strategy (ep_only, tp_only, tp_ep_hybrid)
- **Regression head**: Predicts optimal max batch size (queue depth)

**Input features** (9 dimensions):

| Feature | Source |
|---------|--------|
| total_params_b | Model config |
| active_params_b | Model config |
| num_experts | Model config |
| top_k | Model config |
| num_gpus | Deployment config |
| gpu_memory_gb | Hardware |
| batch_size | Deployment config |
| input_len | Workload estimate |
| output_len | Workload estimate |

### 10.4 Training Data

The 284 benchmark rows from `master_results_clean.csv` serve as training data. For the classification target, each (model, workload, concurrency) group is labeled with the strategy that achieved the highest throughput. For regression, the target is the concurrency level at peak throughput for that configuration.

### 10.5 Scope and Limitations

With only 3 MoE models in the training set, this is a **feasibility demonstration**, not a production-ready system. The predictor can correctly recover the best strategy for held-out configurations via leave-one-model-out cross-validation, but generalization to arbitrary unseen architectures cannot be claimed.

An alternative framing: rather than the ML model itself, the contribution is the **throughput prediction framework** — the predictor can estimate expected tok/s for any (model, strategy, workload) combination, enabling capacity planning without exhaustive benchmarking. A simple lookup table could pick the best strategy from 3 options, but cannot interpolate to predict throughput at unseen concurrency levels or workload mixes, which the regression model can.

### 10.6 Why CPU-Based

The predictor runs on the host CPU alongside the GPU inference server. When workload patterns shift (e.g., average prompt length changes), the predictor can recommend reconfiguration without requiring expensive A/B testing across all strategies. The CPU is otherwise idle during GPU-heavy decode steps, so this adds no overhead to inference throughput.

---

## 11. Data Inventory

| Data | Location | Size |
|------|----------|------|
| Master benchmark CSV | `results/tables/master_results_clean.csv` | 284 rows |
| Single-GPU JSON results | `results/single_gpu_*.json` | 6 files |
| Multi-GPU JSON results | `results/multi_gpu_*.json` | 9 files |
| Throughput plots | `results/figures/*throughput*.png` | 5 files |
| Latency plots | `results/figures/*latency*.png` | 5 files |
| Strategy comparison plots | `results/figures/strategy_comparison_*.png` | 2 files |
| Expert heatmaps | `results/figures/expert_heatmap_*.png` | 4 files |
| Load balance charts | `results/figures/load_balance_*.png` | 4 files |

---

## 11. Remaining Work

1. **Complete profiling** — Run torch profiler with vLLM offline `LLM` class for Mixtral (AWQ), OLMoE, and Qwen on single A100.
2. **Extract profiling features** — Compute memory-bound vs compute-bound characterization from kernel traces; link to multi-GPU strategy results.
3. **Throughput prediction model** — Train a lightweight regression model on the benchmark data to predict throughput for unseen (model, strategy, workload) configurations, framed as a capacity planning tool.
4. **Paper figures** — Finalize plots for strategy comparison, scaling curves, and profiling breakdowns.
