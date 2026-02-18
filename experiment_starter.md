# Experiment Starter Guide

Complete end-to-end experiment plan: from first run to trained CPU predictor to final multi-node scaling studies.

**7 phases**, each building on the previous. Every phase specifies exactly what to run, what data to capture, and what it feeds into downstream.

---

## Phase 1: Single-GPU Baselines

**Goal**: Establish per-model inference characteristics and collect foundation data.

**Time estimate**: 2-3 hours (model downloads + benchmarks)

### 1.1 Setup

```bash
git clone https://github.com/ananya-mh/optimizing-moe-inference.git
cd optimizing-moe-inference
source scripts/setup_env.sh

export HF_TOKEN=your_token_here
export MODEL_DIR=/path/to/models
export RESULTS_DIR=./results
mkdir -p $RESULTS_DIR
```

### 1.2 Download Models

```bash
# Single-GPU models first (smallest to largest)
bash scripts/download_models.sh single_gpu

# Diffusion LLM models
bash scripts/download_models.sh diffusion_llm

# Large models (need multi-GPU, download when ready for Phase 6)
bash scripts/download_models.sh large
```

### 1.3 Run Autoregressive MoE Baselines (vLLM)

```bash
for model in olmoe_1b_7b qwen_moe_a2.7b mixtral_8x7b; do
  bash scripts/run_benchmark.sh --model $model --experiment single_gpu
done
```

### 1.4 Run LLaDA Diffusion Baselines (Custom Engine)

```bash
bash scripts/run_llada_benchmarks.sh
```

### 1.5 Data to Capture

| Metric | Source | Used By |
|--------|--------|---------|
| Throughput (tok/s) | `bench.py` JSON | CPU predictor (Phase 5), paper figures (Phase 7) |
| Output throughput (tok/s) | `bench.py` JSON | Latency analysis |
| Latency per request (ms) | `bench.py` JSON | SLA modeling |
| Model load time (s) | `bench.py` JSON | Cold start cost analysis |
| GPU memory used (GB) | `rocm-smi` during run | Placement feasibility checks |
| Forward pass time (ms) | LLaDA engine JSON | Diffusion overhead breakdown |
| Sampling time (ms) | LLaDA engine JSON | Denoising cost analysis |
| Step timings (per step) | LLaDA engine JSON | Step-level variance |

### 1.6 Collect Profiling Traces

```bash
# Torch profiler — kernel-level breakdown
bash scripts/run_profiling.sh --model mixtral_8x7b --torch
bash scripts/run_profiling.sh --model qwen_moe_a2.7b --torch

# rocprofv3 — HIP API, kernel dispatch, memory transfers
bash scripts/run_profiling.sh --model mixtral_8x7b --rocprof
```

| Trace Metric | Tool | Why It Matters |
|-------------|------|----------------|
| MoE kernel time (`fused_moe_2stages`) | torch profiler | Routing + expert compute cost |
| Attention kernel time | torch profiler | Baseline non-MoE cost |
| Memory transfer time | rocprofv3 | HBM bandwidth utilization |
| Kernel launch overhead | rocprofv3 | Dispatch latency |
| Expert routing decisions | `ExpertLoadTracker` in `llada_engine.py` | Load balance input data |

### 1.7 Expected Outputs

```
results/
├── olmoe_tp1.json
├── qwen_moe_tp1.json
├── mixtral_tp1.json
├── llada_8b_single.json
├── llada_8b_steps32.json
├── llada_8b_steps64.json
├── llada_8b_steps128.json
├── llada_8b_gen64.json
├── llada_8b_gen128.json
├── llada_8b_gen256.json
├── llada_moe_single.json
├── llada_moe_steps32.json
├── llada_moe_steps64.json
└── llada_moe_steps128.json
```

---

## Phase 2: Expert Routing Analysis

**Goal**: Understand expert activation patterns that drive placement decisions.

**Time estimate**: 1-2 hours

**Depends on**: Phase 1 profiling traces

### 2.1 Run Load Balance Simulation

```bash
# Simulate routing across uniform, Zipfian, and skewed distributions
python -m src.placement.load_balancing

# Generate visualizations
python analysis/plot_load_balance.py
```

### 2.2 Collect Real Routing Data

The `ExpertLoadTracker` in `src/inference/llada_engine.py` captures expert activation
counts automatically for the LLaDA-MoE model. For autoregressive models, extract
routing decisions from torch profiler traces.

### 2.3 Data to Capture

| Metric | Source | Purpose |
|--------|--------|---------|
| Per-expert activation count | `ExpertLoadTracker` / profiler | Load imbalance measurement |
| Gini coefficient | `load_balancing.py` output | Single-number imbalance metric |
| Co-activation matrix | Routing trace analysis | Affinity-aware placement input |
| Hot/cold expert IDs | `load_balancing.py` output | Replication candidates |
| Per-step routing variance | LLaDA step timings | Temporal load patterns |

### 2.4 Key Analysis Questions

1. Are expert activations uniform or skewed? (Gini < 0.1 = balanced)
2. Which experts are consistently hot/cold?
3. Do co-activated experts cluster? (Affinity patterns)
4. Does the distribution change with input length or topic?

---

## Phase 3: Multi-GPU Placement Study (Core Experiment)

**Goal**: Compare TP vs EP vs hybrid strategies across GPU counts. This is the central
experiment for the paper.

**Time estimate**: 8-12 hours (many configurations)

**Depends on**: Phase 1 baselines

### 3.1 Generate Factorial Design

```bash
python -m src.benchmark.factorial_study
# Output: results/factorial_design.json
```

### 3.2 Factorial Design Matrix

| Factor | Levels | Count |
|--------|--------|-------|
| Model | OLMoE, Qwen-MoE, Mixtral, Qwen2-57B, LLaDA-MoE | 5 |
| GPUs | 1, 2, 4, 8 | 4 |
| Strategy | TP-only, EP-only, TP+EP hybrid | 3 |
| Queue depth | 8, 32, 128 | 3 |
| Workload (input/output len) | Short (128/64), Medium (512/128), Long (2048/256) | 3 |

**Full factorial**: 5 x 4 x 3 x 3 x 3 = **540 experiments**

**Practical subset** (skip infeasible combos — e.g., TP=8 for 7B model, or 1-GPU EP): **~200 experiments**

### 3.3 Run Autoregressive Models (vLLM)

```bash
for gpus in 1 2 4 8; do
  for model in mixtral_8x7b qwen2_57b; do

    # TP-only
    bash scripts/run_benchmark.sh --model $model --experiment multi_gpu \
      --strategy tp_only --num-gpus $gpus

    # EP-only (vLLM --enable-expert-parallel flag)
    bash scripts/run_benchmark.sh --model $model --experiment multi_gpu \
      --strategy ep_only --num-gpus $gpus

    # TP+EP hybrid
    bash scripts/run_benchmark.sh --model $model --experiment multi_gpu \
      --strategy tp_ep_hybrid --num-gpus $gpus
  done
done
```

### 3.4 Run LLaDA-MoE (Custom Engine with RCCL)

```bash
export ROCM_IMAGE=rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch_release_2.4.0

for gpus in 1 2 4 8; do
  docker run --rm --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render --shm-size=32g --ipc=host \
    -e HIP_VISIBLE_DEVICES=$(seq -s, 0 $((gpus-1))) \
    -v $MODEL_DIR:/models $ROCM_IMAGE \
    bash -c "pip install -q transformers accelerate sentencepiece protobuf safetensors && \
    torchrun --nproc_per_node=$gpus /models/llada_distributed.py \
      --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
      --gen-length 128 --steps 64 --num-prompts 10 \
      --output-json /models/results/llada_moe_dist${gpus}.json"
done
```

### 3.5 Data to Capture Per Configuration

| Metric | Purpose |
|--------|---------|
| Throughput (tok/s) | Primary comparison metric |
| Scaling efficiency (throughput_N / throughput_1) | Communication overhead cost |
| Per-GPU memory usage (GB) | Feasibility check |
| All-to-all communication time (ms) | EP communication cost |
| Expert load imbalance (Gini coefficient) | Balance quality per strategy |
| Per-GPU idle time (ms) | Utilization gap from imbalance |

### 3.6 Expected Results Structure

```
results/
├── mixtral_tp1.json
├── mixtral_tp2.json
├── mixtral_tp4.json
├── mixtral_tp8.json
├── mixtral_ep2.json
├── mixtral_ep4.json
├── mixtral_ep8.json
├── mixtral_tp2_ep2.json   # hybrid: TP=2 x EP=2 = 4 GPUs
├── mixtral_tp2_ep4.json   # hybrid: TP=2 x EP=4 = 8 GPUs
├── qwen2_57b_tp2.json
├── ...
├── llada_moe_dist1.json
├── llada_moe_dist2.json
├── llada_moe_dist4.json
└── llada_moe_dist8.json
```

---

## Phase 4: Expert-Aware Batching Sweep

**Goal**: Find optimal queue depth per model and strategy.

**Time estimate**: 3-4 hours

**Depends on**: Phase 1 baselines

### 4.1 Run Queue Depth Sweep

```bash
for model in mixtral_8x7b qwen_moe_a2.7b qwen2_57b llada_moe_7b; do
  for queue_depth in 4 8 16 32 64 128 256; do
    bash scripts/run_benchmark.sh --model $model --experiment single_gpu \
      --queue-depth $queue_depth
  done
done
```

### 4.2 Data to Capture

For each (model, queue_depth) pair:

| Queue Depth | Throughput (tok/s) | CU Occupancy (%) | Expert Wait Time (ms) | GPU Memory (GB) |
|------------|-------------------|------------------|----------------------|-----------------|
| 4 | ... | ... | ... | ... |
| 8 | ... | ... | ... | ... |
| 16 | ... | ... | ... | ... |
| 32 | ... | ... | ... | ... |
| 64 | ... | ... | ... | ... |
| 128 | ... | ... | ... | ... |
| 256 | ... | ... | ... | ... |

### 4.3 What to Look For

- **Sweet spot**: Queue depth where throughput plateaus but memory hasn't saturated
- **Knee point**: Where adding more concurrent requests stops helping
- **OOM boundary**: Max queue depth before GPU memory runs out
- **Model differences**: Do models with more experts benefit from larger queues?

---

## Phase 5: Train the CPU Predictor

**Goal**: Build a lightweight ML model that predicts optimal strategy + queue depth
from configuration features alone, without running a full benchmark.

**Time estimate**: 1-2 hours (after Phases 1-4 data is collected)

**Depends on**: Phases 1, 3, and 4 results (need 50+ data points)

### 5.1 Training Data Schema

Each data point for the CPU predictor requires these fields:

```python
training_sample = {
    # ── Model features (from configs/models.yaml) ──
    "model_total_params_b": 46.7,       # Total parameters in billions
    "model_active_params_b": 12.9,      # Active parameters per token
    "num_experts": 8,                    # Number of routed experts
    "top_k": 2,                          # Experts activated per token

    # ── Hardware features (from runtime) ──
    "num_gpus": 4,                       # GPUs used in this config
    "gpu_memory_gb": 192,               # Per-GPU memory (MI300X = 192)

    # ── Workload features (from experiment config) ──
    "batch_size": 50,                    # Concurrent requests / prompts
    "input_len": 512,                    # Average input sequence length
    "output_len": 128,                   # Average output sequence length

    # ── TARGETS (measured from experiments) ──
    "best_strategy": "ep_only",          # Which strategy gave best throughput
    "best_queue_depth": 32,              # Queue depth that maximized throughput
}
```

### 5.2 Build the Training Set

```python
import json
from pathlib import Path
from src.utils.config import get_model_config

results_dir = Path("results")
training_data = []

# Model metadata lookup
model_info = {
    "mixtral": {"total": 46.7, "active": 12.9, "experts": 8, "top_k": 2},
    "qwen_moe": {"total": 14.3, "active": 2.7, "experts": 60, "top_k": 4},
    "olmoe": {"total": 6.9, "active": 1.3, "experts": 64, "top_k": 8},
    "qwen2_57b": {"total": 57.4, "active": 14.0, "experts": 64, "top_k": 8},
    "llada_moe": {"total": 7.0, "active": 1.4, "experts": 64, "top_k": 8},
}

# Group results by (model, num_gpus) and label with the winning strategy
# For each group, the strategy with highest throughput is "best_strategy"
for model_key, info in model_info.items():
    # Collect all results for this model
    model_results = []
    for f in results_dir.glob(f"{model_key}*.json"):
        r = json.loads(f.read_text())
        model_results.append(r)

    # Group by gpu count, find best strategy per group
    from itertools import groupby
    for gpu_count, group in groupby(
        sorted(model_results, key=lambda x: x.get("tp_size", 1)),
        key=lambda x: x.get("tp_size", 1)
    ):
        configs = list(group)
        best = max(configs, key=lambda x: x["throughput_tok_s"])

        training_data.append({
            "model_total_params_b": info["total"],
            "model_active_params_b": info["active"],
            "num_experts": info["experts"],
            "top_k": info["top_k"],
            "num_gpus": gpu_count,
            "gpu_memory_gb": 192,
            "batch_size": best["num_prompts"],
            "input_len": best["in_tokens"] // best["num_prompts"],
            "output_len": best["out_tokens"] // best["num_prompts"],
            "best_strategy": best.get("strategy", "tp_only"),
            "best_queue_depth": best["num_prompts"],
        })

print(f"Training set: {len(training_data)} samples")
```

### 5.3 Train the Predictor

```python
from src.placement.predictor import PlacementPredictor

predictor = PlacementPredictor()
predictor.train(training_data)
predictor.save()
```

### 5.4 Validate with Leave-One-Model-Out Cross-Validation

```python
models = list(model_info.keys())
for holdout in models:
    train_set = [d for d in training_data if not d.get("_model", "").startswith(holdout)]
    test_set = [d for d in training_data if d.get("_model", "").startswith(holdout)]

    if not test_set:
        continue

    p = PlacementPredictor()
    p.train(train_set)

    correct = 0
    for t in test_set:
        pred = p.predict(t)
        if pred["recommended_strategy"] == t["best_strategy"]:
            correct += 1
        print(f"  Model={holdout} GPUs={t['num_gpus']}: "
              f"predicted={pred['recommended_strategy']} "
              f"actual={t['best_strategy']} "
              f"confidence={pred['confidence']:.2f}")

    if test_set:
        print(f"  Accuracy for {holdout}: {correct}/{len(test_set)}")
```

### 5.5 Use at Runtime

```python
# Load saved predictor
predictor = PlacementPredictor()
predictor.load()

# Predict for a new configuration
result = predictor.predict({
    "model_total_params_b": 132.0,
    "model_active_params_b": 36.0,
    "num_experts": 16,
    "top_k": 4,
    "num_gpus": 8,
    "gpu_memory_gb": 192,
    "batch_size": 32,
    "input_len": 512,
    "output_len": 128,
})

print(f"Strategy: {result['recommended_strategy']}")
print(f"Queue depth: {result['recommended_queue_depth']}")
print(f"Confidence: {result['confidence']:.0%}")
```

### 5.6 Validation Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Strategy prediction accuracy | > 80% | Leave-one-model-out CV |
| Queue depth RMSE | < 2x optimal | Regression quality |
| CPU inference time | < 1ms | Must be lightweight |
| Training data points needed | 50+ | From Phases 1, 3, 4 |

---

## Phase 6: Multi-Node Scaling Study

**Goal**: Scale Expert Parallelism across 2-4 nodes with RDMA interconnect.

**Time estimate**: 4-8 hours

**Depends on**: Phase 3 (intra-node results as baseline)

### 6.1 Models for Multi-Node

| Model | Why | Min Nodes | Max Nodes |
|-------|-----|-----------|-----------|
| Qwen2-57B-A14B | Moderate size, 64 experts | 2 (16 GPUs) | 2 |
| DBRX (132B) | Large, 16 experts | 2 (16 GPUs) | 2 |
| DeepSeek-V3 (671B) | Largest, 256 experts | 1 (8 GPUs) | 4 (32 GPUs) |

### 6.2 Run Multi-Node Experiments

```bash
# 1-node baseline (already collected in Phase 3 for DeepSeek-V3)
# Use existing results/deepseek_v3_tp8.json

# 2-node DeepSeek-V3 with EP (16 GPUs)
srun --partition=$PARTITION --nodes=2 --ntasks=2 \
  --cpus-per-task=32 --gres=gpu:mi300x:8 \
  --time=04:00:00 --job-name=dsv3-2node \
  bash scripts/run_benchmark.sh --model deepseek_v3 \
    --experiment multi_node --num-nodes 2 --strategy ep_only

# 2-node with TP+EP hybrid (TP=8 per node, EP=2 across nodes)
srun --partition=$PARTITION --nodes=2 --ntasks=2 \
  --cpus-per-task=32 --gres=gpu:mi300x:8 \
  --time=04:00:00 --job-name=dsv3-2node-hybrid \
  bash scripts/run_benchmark.sh --model deepseek_v3 \
    --experiment multi_node --num-nodes 2 --strategy tp_ep_hybrid

# 4-node DeepSeek-V3 with EP (32 GPUs)
srun --partition=$PARTITION --nodes=4 --ntasks=4 \
  --cpus-per-task=32 --gres=gpu:mi300x:8 \
  --time=04:00:00 --job-name=dsv3-4node \
  bash scripts/run_benchmark.sh --model deepseek_v3 \
    --experiment multi_node --num-nodes 4 --strategy ep_only

# 2-node DBRX
srun --partition=$PARTITION --nodes=2 --ntasks=2 \
  --cpus-per-task=32 --gres=gpu:mi300x:8 \
  --time=04:00:00 --job-name=dbrx-2node \
  bash scripts/run_benchmark.sh --model dbrx \
    --experiment multi_node --num-nodes 2 --strategy ep_only
```

### 6.3 Data to Capture

| Nodes | GPUs | Strategy | Throughput (tok/s) | Inter-node BW (GB/s) | All2All Latency (us) | Scaling Efficiency |
|-------|------|----------|-------------------|---------------------|---------------------|-------------------|
| 1 | 8 | TP=8 | baseline | N/A | N/A | 1.00x |
| 2 | 16 | EP=16 | ... | ... | ... | ... |
| 2 | 16 | TP=8 + EP=2 | ... | ... | ... | ... |
| 4 | 32 | EP=32 | ... | ... | ... | ... |
| 4 | 32 | TP=8 + EP=4 | ... | ... | ... | ... |

### 6.4 Communication Profiling

```bash
# Profile RCCL/NCCL communication during multi-node EP
bash scripts/run_profiling.sh --model deepseek_v3 --rocprof --num-nodes 2

# Key metrics to extract:
# - AllToAll kernel time (us)
# - AllReduce kernel time (us) 
# - Network bandwidth utilization (% of CX-7 peak)
# - Overlap efficiency (compute during communication)
```

---

## Phase 7: Analysis and Paper Figures

**Goal**: Synthesize all results into figures, tables, and insights for the SIEDS 2026 paper.

**Depends on**: All previous phases

### 7.1 Generate Outputs

```bash
# Throughput scaling bar charts
python analysis/plot_results.py

# Expert load balance heatmaps and Gini curves
python analysis/plot_load_balance.py

# LaTeX tables for the paper
python analysis/generate_tables.py

# Factorial study ANOVA (which factors matter most)
python -m src.benchmark.factorial_study --analyze results/
```

### 7.2 Key Figures for the Paper

| Figure | Data Source | Story It Tells |
|--------|-----------|----------------|
| Throughput vs GPU count (per strategy) | Phase 3 | TP hurts MoE; EP helps |
| Throughput vs queue depth curves | Phase 4 | Sweet spot for batching |
| Expert load heatmap (TP vs EP) | Phase 2 + 3 | EP distributes load more evenly |
| Scaling efficiency: 1 to 32 GPUs | Phase 3 + 6 | Multi-node EP is viable with RDMA |
| AR vs Diffusion MoE comparison | Phase 1 | Diffusion MoE needs EP even more |
| CPU predictor accuracy plot | Phase 5 | Lightweight runtime decisions work |
| Step count vs throughput (LLaDA) | Phase 1 | Quality-speed tradeoff for dLLMs |
| ANOVA factor importance chart | Phase 3 | Strategy > GPU count > queue depth |
| Communication breakdown pie chart | Phase 6 | Where time goes in multi-node EP |

### 7.3 Key Claims to Support

| Claim | Evidence |
|-------|---------|
| TP degrades MoE throughput at small scale | Phase 3: Mixtral TP=2 is 40% slower than TP=1 |
| EP improves throughput for high-expert models | Phase 3: EP vs TP comparison |
| Active params predict throughput better than total | Phase 1: OLMoE vs Mixtral |
| Diffusion MoE benefits more from EP than AR MoE | Phase 1 + 3: LLaDA-MoE 10x slower single-GPU |
| CPU predictor achieves >80% accuracy | Phase 5: Cross-validation results |
| Expert load imbalance is the primary bottleneck | Phase 2: Gini coefficients across models |
| Multi-node EP scales with RDMA | Phase 6: Near-linear scaling to 4 nodes |

---

## Phase Summary

| Phase | Input | Output | Feeds Into | Est. Time |
|-------|-------|--------|-----------|-----------|
| **1. Single-GPU Baselines** | Models + 1 GPU | JSON results, profiling traces | Phases 2, 3, 5, 7 | 2-3h |
| **2. Routing Analysis** | Phase 1 traces | Expert activation patterns, Gini | Phase 5, 7 | 1-2h |
| **3. Multi-GPU Placement** | Phase 1 baselines | Strategy comparison (TP/EP/hybrid) | Phase 5, 7 | 8-12h |
| **4. Batching Sweep** | Phase 1 baselines | Queue depth vs throughput curves | Phase 5, 7 | 3-4h |
| **5. CPU Predictor** | Phases 1-4 (50+ samples) | Trained RandomForest model | Phase 7 | 1-2h |
| **6. Multi-Node Scaling** | DBRX, DeepSeek-V3 | Cross-node EP results | Phase 7 | 4-8h |
| **7. Paper Analysis** | All phases | Figures, tables, ANOVA | SIEDS submission | 2-3h |

**Total estimated time**: 20-35 hours of compute (many phases can run in parallel across nodes)

---

## Checklist

- [ ] Phase 1: Single-GPU baselines for all 5 single-GPU models
- [ ] Phase 1: LLaDA step sweep (32, 64, 128 steps)
- [ ] Phase 1: LLaDA gen-length sweep (64, 128, 256 tokens)
- [ ] Phase 1: Torch profiler traces for Mixtral and Qwen-MoE
- [ ] Phase 1: rocprofv3 traces for Mixtral
- [ ] Phase 2: Load balance simulation (uniform, Zipfian, skewed)
- [ ] Phase 2: Real routing data from LLaDA-MoE runs
- [ ] Phase 3: Mixtral TP=1/2/4/8
- [ ] Phase 3: Mixtral EP=2/4/8
- [ ] Phase 3: Qwen2-57B TP=2/4 and EP=2/4/8
- [ ] Phase 3: LLaDA-MoE distributed 1/2/4/8 GPUs
- [ ] Phase 3: Queue depth sweep per strategy
- [ ] Phase 4: Queue depth 4-256 for Mixtral, Qwen-MoE, Qwen2-57B
- [ ] Phase 5: Build training set (50+ samples)
- [ ] Phase 5: Train CPU predictor
- [ ] Phase 5: Leave-one-model-out cross-validation
- [ ] Phase 6: DeepSeek-V3 on 2 nodes (16 GPUs)
- [ ] Phase 6: DeepSeek-V3 on 4 nodes (32 GPUs)
- [ ] Phase 6: DBRX on 2 nodes (16 GPUs)
- [ ] Phase 7: Generate all paper figures
- [ ] Phase 7: Run ANOVA on factorial results
- [ ] Phase 7: Generate LaTeX tables
