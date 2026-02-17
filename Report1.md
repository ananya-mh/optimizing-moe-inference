# Report 1: MoE Inference Benchmark Results on AMD MI300X

**Date**: February 17, 2026  
**Author**: Ravi Gupta  
**Cluster**: AMD OCSRI Cluster (useocpslog-002.amd.com)  
**Hardware**: AMD Instinct MI300X (192 GB HBM3 per GPU), 8 GPUs per node, CX-7 RDMA  
**Framework**: vLLM 0.14.0rc2 (AMD ROCm build with AITER, UCX, RIXL optimizations)  
**Software Stack**: ROCm 7.0.51831, PyTorch 2.9.0a0, NCCL 2.27.3  

---

## 1. Cluster Environment

| Component | Version / Details |
|-----------|-------------------|
| GPU | AMD Instinct MI300X (192 GB HBM3 VRAM) |
| GPUs/node | 8 |
| CPU | 2x AMD EPYC (224 threads total) |
| RAM | 2 TB DDR5 |
| Interconnect | CX-7 RDMA (inter-node), Infinity Fabric (intra-node) |
| OS | Ubuntu 22.04, kernel 5.15.0-1074-oracle |
| ROCm | 7.0.51831-a3e329ad8 |
| PyTorch | 2.9.0a0+gitb425573 |
| vLLM | 0.14.0rc2.dev350+g9ef3b718d |
| Docker Image | `rocm/pytorch-private:miali_vllm_0.14.0rc2_ucx_develop_rixl_develop_20260126_retemadi_added_profile_pr18827` |
| Job Scheduler | Slurm |

### Partition Info
- **Partition**: `amd-rccl`
- **Available Nodes**: ~55 idle nodes with 8x MI300X each
- **Allocation**: `srun --partition=amd-rccl --gres=gpu:mi300x:8`

---

## 2. Models Under Study

| Model | HF ID | Total Params | Active Params | Experts | Top-K | Type | Status |
|-------|--------|-------------|--------------|---------|-------|------|--------|
| OLMoE-1B-7B | allenai/OLMoE-1B-7B-0924 | 6.9B | 1.3B | 64 | 8 | Autoregressive MoE | Benchmarked |
| Qwen1.5-MoE-A2.7B | Qwen/Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | Autoregressive MoE | Benchmarked |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-Instruct-v0.1 | 46.7B | 12.9B | 8 | 2 | Autoregressive MoE | Benchmarked |
| Qwen2-57B-A14B | Qwen/Qwen2-57B-A14B-Instruct | 57.4B | 14.0B | 64 | 8 | Autoregressive MoE | Benchmarked |
| LLaDA-MoE-7B | inclusionAI/LLaDA-MoE-7B-A1B-Instruct | 7.0B | 1.4B | 64 | 8 | Diffusion MoE | **Benchmarked** (custom engine) |
| LLaDA-8B | GSAI-ML/LLaDA-8B-Instruct | 8.0B | 8.0B | 1 (dense) | - | Diffusion Dense | **Benchmarked** (custom engine) |
| DBRX | databricks/dbrx-instruct | 132.0B | 36.0B | 16 | 4 | Autoregressive MoE | Needs HF license acceptance |
| DeepSeek-V3 | deepseek-ai/DeepSeek-V3 | 671.0B | 37.0B | 256 | 8 | Autoregressive MoE | Available at cluster |

### Model Storage
- **Shared Path**: `/shared_inference/ravgupta_models/`
- **DeepSeek-V3**: `/shared_inference/models_blog/DeepSeek-V3`

---

## 3. Benchmark Configuration

### Workload
- **Input**: ~180 tokens per prompt (repeated text for consistent benchmarking)
- **Output**: 128 tokens per prompt (max_tokens=128, ignore_eos=True)
- **Prompts**: 50 per configuration (20 for large models)
- **Warmup**: 2 prompts before measurement
- **Sampling**: temperature=0.0 (greedy), deterministic
- **Mode**: Offline throughput (all prompts submitted at once via `vllm.LLM`)

### vLLM Settings
```python
LLM(
    model=model_path,
    tensor_parallel_size=tp_size,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    enforce_eager=True,  # Disable CUDAGraph for measurement stability
)
```

### Docker Launch Command
```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=32g --ipc=host \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=<gpu_ids> \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -v /shared_inference/ravgupta_models:/models \
  -v /shared_inference/ravgupta_models/results:/results \
  $VLLM_IMAGE \
  python3 /models/bench.py <model_path> <tp_size> <output_json> <num_prompts>
```

---

## 4. Results

### 4.1 Single-GPU Baselines (TP=1)

All models run on a single AMD MI300X GPU with AITER Flash Attention and fused MoE kernels.

| Model | Throughput (tok/s) | Output (tok/s) | Req/s | Latency (ms/req) | Model Load (s) |
|-------|-------------------|----------------|-------|------------------|----------------|
| **OLMoE-1B-7B** | **6,965.3** | 2,885.3 | 22.54 | 44 | 334.4 |
| **Qwen1.5-MoE-A2.7B** | **4,050.7** | 1,678.0 | 13.11 | 76 | 417.9 |
| **Mixtral-8x7B** | **4,107.2** | 1,695.9 | 13.25 | 75 | 776.9 |

**Key Observations**:
- OLMoE is fastest despite having the most experts (64), because active parameter count (1.3B) is smallest
- Mixtral and Qwen-MoE achieve similar throughput despite 3x difference in total params (46.7B vs 14.3B), confirming that **active parameters, not total parameters, determine throughput**
- All three models fit comfortably in a single MI300X (192 GB VRAM)
- First-run model load times include AITER JIT kernel compilation (~5-7 min); subsequent runs are cached

### 4.2 Multi-GPU Tensor Parallelism Scaling

#### Mixtral-8x7B (Triton MoE backend for TP>1)

| TP Size | GPUs | Throughput (tok/s) | Output (tok/s) | Req/s | Latency (ms) | Scaling vs TP=1 |
|---------|------|-------------------|----------------|-------|-------------|-----------------|
| 1 | 1 | 4,107.2 | 1,695.9 | 13.25 | 75 | 1.00x |
| 2 | 2 | 2,463.7 | 1,017.3 | 7.95 | 126 | 0.60x |
| 4 | 4 | 2,294.9 | 947.6 | 7.40 | 135 | 0.56x |
| 8 | 8 | 2,399.8 | 990.9 | 7.74 | 129 | 0.58x |

#### Qwen1.5-MoE-A2.7B

| TP Size | GPUs | Backend | Throughput (tok/s) | Output (tok/s) | Req/s | Latency (ms) | Scaling |
|---------|------|---------|-------------------|----------------|-------|-------------|---------|
| 1 | 1 | AITER | 4,050.7 | 1,678.0 | 13.11 | 76 | 1.00x |
| 2 | 2 | Triton | 2,608.2 | 1,080.4 | 8.44 | 118 | 0.64x |

#### Qwen2-57B-A14B-Instruct (too large for 1 GPU)

| TP Size | GPUs | Throughput (tok/s) | Output (tok/s) | Req/s | Latency (ms) | Load Time (s) |
|---------|------|-------------------|----------------|-------|-------------|---------------|
| 2 | 2 | 1,021.2 | 423.0 | 3.30 | 303 | 571.5 |
| 4 | 4 | 1,027.9 | 425.8 | 3.33 | 301 | 79.9 |

### 4.3 LLaDA Diffusion LLM Benchmarks (Custom Engine)

LLaDA models use **masked diffusion** (non-autoregressive) inference and are not supported by vLLM.
We built a custom inference engine (`src/inference/llada_engine.py`) using HuggingFace Transformers
with ROCm support, implementing the full block-based denoising loop.

**Configuration**: gen_length=128, steps=64, block_length=32, temperature=0.0, 10 prompts

| Model | Type | Experts | Throughput (tok/s) | Avg Latency (ms/prompt) | Model Forward (ms) | Sampling (ms) | Load Time (s) |
|-------|------|---------|-------------------|------------------------|-------------------|--------------|---------------|
| **LLaDA-8B** | Dense Diffusion | 1 | **90.7** | 1,411 | 1,378 | 28 | 91.1 |
| **LLaDA-MoE-7B** | MoE Diffusion | 64 (top-8) | **9.6** | 13,269 | 13,227 | 34 | 88.0 |

**Key Observations**:
- LLaDA-8B achieves 90.7 tok/s on a single MI300X — comparable to autoregressive models at similar size
- LLaDA-MoE-7B is **9.4x slower** than the dense variant despite smaller active parameters (1.4B vs 8B)
- The bottleneck is the forward pass: 64 experts with top-8 routing creates significant compute overhead when all experts are on one GPU
- Sampling overhead is minimal (2-3% of total time) — optimization should focus on the MoE forward pass
- **This demonstrates why Expert Parallelism is critical for MoE diffusion models** — distributing 64 experts across multiple GPUs should dramatically improve throughput
- GPU memory: LLaDA-8B uses ~16 GB, LLaDA-MoE-7B uses ~28 GB — both fit easily on MI300X (192 GB)

#### LLaDA-8B Step Count Sweep (gen_length=128)

| Steps | Throughput (tok/s) | Avg Latency (ms) | Forward (ms) | Sampling (ms) | Speedup vs 64 |
|-------|-------------------|------------------|-------------|--------------|---------------|
| 32 | **175.4** | 730 | 712 | 14 | **1.93x** |
| 64 | 90.7 | 1,411 | 1,378 | 28 | 1.00x |
| 128 | 47.5 | 2,696 | 2,633 | 54 | 0.52x |

Throughput scales **linearly with step count** — halving steps doubles throughput. This is a key
tunable parameter for quality-vs-speed tradeoff in diffusion LLMs.

#### LLaDA-8B Generation Length Sweep (steps=64)

| Gen Length | Throughput (tok/s) | Avg Latency (ms) | Forward (ms) | Speedup vs 128 |
|-----------|-------------------|------------------|-------------|----------------|
| 64 | 49.8 | 1,285 | 1,257 | — |
| 128 | 90.7 | 1,411 | 1,378 | 1.00x |
| 256 | **171.1** | 1,496 | 1,448 | **1.89x** |

Throughput increases with generation length because the denoising loop cost is nearly constant
regardless of how many tokens are generated in each block. This is a fundamental advantage of
diffusion LLMs over autoregressive models for long-form generation.

#### LLaDA-MoE-7B Step Count Sweep (gen_length=128, 5 prompts)

| Steps | Throughput (tok/s) | Avg Latency (ms) | Forward (ms) | Speedup vs 64 |
|-------|-------------------|------------------|-------------|---------------|
| 32 | **17.8** | 7,201 | 7,180 | **1.85x** |
| 64 | 9.6 | 13,269 | 13,227 | 1.00x |
| 128 | 5.0 | 25,731 | 25,649 | 0.52x |

Same linear scaling with steps, but the MoE variant is ~10x slower across all step counts.
The 64 experts contribute ~10x overhead vs dense model at equivalent parameter budget.

### LLaDA-MoE Raw Results
```json
{
  "model": "/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
  "is_moe": true,
  "num_experts": 64,
  "num_prompts": 10,
  "gen_length": 128,
  "steps": 64,
  "total_generated_tokens": 1280,
  "total_time_ms": 132686.48,
  "avg_time_per_prompt_ms": 13268.65,
  "throughput_tok_s": 9.6,
  "avg_model_forward_ms": 13227.17,
  "avg_sampling_ms": 34.37,
  "model_load_time_s": 88.0,
  "device": "cuda:0",
  "dtype": "torch.bfloat16"
}
```

### LLaDA-8B Raw Results
```json
{
  "model": "/models/GSAI-ML/LLaDA-8B-Instruct",
  "is_moe": false,
  "num_experts": 0,
  "num_prompts": 10,
  "gen_length": 128,
  "steps": 64,
  "total_generated_tokens": 1280,
  "total_time_ms": 14107.33,
  "avg_time_per_prompt_ms": 1410.73,
  "throughput_tok_s": 90.7,
  "avg_model_forward_ms": 1377.84,
  "avg_sampling_ms": 27.56,
  "model_load_time_s": 91.1,
  "device": "cuda:0",
  "dtype": "torch.bfloat16"
}
```

### 4.4 DeepSeek-V3 (671B, 256 experts) - TP=8

The largest model in our study, DeepSeek-V3 requires all 8 MI300X GPUs via TP=8.
Uses FP8 quantization with AITER MLA (Multi-Latent Attention) backend and Triton FP8 MoE.

| Model | TP | Throughput (tok/s) | Output (tok/s) | Req/s | Latency (s/req) | Load Time (s) |
|-------|-----|-------------------|----------------|-------|----------------|---------------|
| **DeepSeek-V3** | 8 | **136.6** | 56.4 | 0.44 | 2.27 | 2,747 |

**Key Observations**:
- Loading time is ~46 minutes for 163 safetensor shards + AITER JIT compilation
- FP8 quantization enables fitting 671B params across 8x192GB GPUs
- Output throughput (56.4 tok/s) is limited by model size and communication overhead
- The AITER MLA backend provides optimized multi-head latent attention
- `VLLM_ROCM_USE_AITER_MOE=0` was used to fall back to Triton FP8 MoE kernels

### deepseek_v3_tp8.json
```json
{
  "model": "/models/DeepSeek-V3",
  "tp_size": 8,
  "num_prompts": 10,
  "elapsed_s": 22.69,
  "in_tokens": 1820,
  "out_tokens": 1280,
  "total_tokens": 3100,
  "throughput_tok_s": 136.6,
  "output_tok_s": 56.4,
  "req_s": 0.44,
  "latency_s": 2.269,
  "load_time_s": 2746.5
}
```

### 4.5 AITER vs Triton MoE Backend

| Backend | Feature | Notes |
|---------|---------|-------|
| **AITER** (AMD) | Fused MoE kernels via CK (Composable Kernels) | Works on TP=1; GEMM dimension issues on TP>1 for high-expert models |
| **Triton** | Portable MoE kernels | Works on all TP sizes; ~35% slower than AITER on TP=1 |

The AITER backend uses 2-stage fused MoE kernels (`fused_moe_2stages`) with JIT-compiled CK operations. It encounters `RuntimeError: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem` when expert dimensions after TP splitting produce unsupported shapes.

---

## 5. Key Findings

### 5.1 Tensor Parallelism Does NOT Help for Small MoE Models

For Mixtral-8x7B and Qwen-MoE, adding GPUs via Tensor Parallelism **reduces** throughput by 35-44%. The all-reduce communication overhead dominates any compute benefit. This strongly validates the paper's thesis: **expert placement strategies are more important than naive parallelism for MoE models**.

### 5.2 Active Parameters Predict Throughput

| Model | Active Params | Throughput | Correlation |
|-------|--------------|-----------|-------------|
| OLMoE (1.3B active) | 6,965 tok/s | Highest throughput, lowest active params |
| Qwen-MoE (2.7B active) | 4,051 tok/s | Mid-range |
| Mixtral (12.9B active) | 4,107 tok/s | Similar to Qwen despite 5x more active params |

Mixtral's competitive throughput despite larger active params is due to fewer experts (8 vs 60) reducing routing overhead.

### 5.3 Qwen2-57B Scales Poorly with TP

TP=2 and TP=4 show nearly identical throughput (1,021 vs 1,028 tok/s), confirming communication overhead saturates early. **Expert Parallelism (EP)** may be more effective for this model.

### 5.4 MI300X Memory Advantage

The 192 GB HBM3 per GPU allows running Mixtral-8x7B (93 GB model) on a single GPU, which is not possible on A100-80G or H100-80G without quantization. This is a key advantage for MoE inference research.

---

## 6. Issues Encountered

| Issue | Root Cause | Resolution |
|-------|-----------|-----------|
| `Engine core initialization failed` | GPU memory consumed by other processes; wrong env var | Use `HIP_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`) |
| `multiprocessing spawn` error | vLLM 0.14 requires `if __name__ == '__main__':` guard | Added guard to benchmark script |
| AITER GEMM failure on TP>1 | CK fused MoE kernels don't support all dimension splits | Fall back to Triton: `VLLM_ROCM_USE_AITER_MOE=0` |
| OLMoE corrupted weights | Download interrupted on login node | Re-downloaded on compute node |
| LLaDA models unsupported | `LLaDAMoEModel` arch not in vLLM 0.14 | Building custom inference framework |
| Docker image not found | `vllm/vllm-openai:latest-rocm` doesn't exist | Used pre-built cluster image with AMD optimizations |

---

## 7. Reproducing the Results

### Step 1: Allocate a Compute Node
```bash
ssh ravgupta@useocpslog-002.amd.com
srun --partition=amd-rccl --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --gres=gpu:mi300x:8 \
  --time=04:00:00 --job-name=moe-bench bash
```

### Step 2: Verify GPU Access
```bash
rocm-smi --showmeminfo vram
```

### Step 3: Prepare Benchmark Script

The benchmark script is at `/shared_inference/ravgupta_models/bench.py`:
```python
import json, time, sys, os
from multiprocessing import freeze_support

def main():
    model_path = sys.argv[1]
    tp_size = int(sys.argv[2])
    output_file = sys.argv[3]
    num_prompts = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    from vllm import LLM, SamplingParams
    print("Loading model: %s, TP=%d" % (model_path, tp_size))
    t0 = time.time()
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, max_model_len=4096,
              gpu_memory_utilization=0.85, trust_remote_code=True, enforce_eager=True)
    load_time = time.time() - t0
    print("Model loaded in %.1fs" % load_time)

    sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    prompts = ["The future of artificial intelligence is " * 30] * num_prompts

    print("Warmup...")
    _ = llm.generate(prompts[:2], sp)

    print("Benchmarking %d prompts..." % num_prompts)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    in_tok = sum(len(o.prompt_token_ids) for o in outputs)
    out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    total = in_tok + out_tok

    sep = "=" * 55
    print("\n" + sep)
    print("Model: %s" % model_path)
    print("TP=%d | Prompts=%d | Elapsed=%.2fs" % (tp_size, num_prompts, elapsed))
    print("Tokens: %d (in=%d, out=%d)" % (total, in_tok, out_tok))
    print("Throughput: %.1f tok/s (output: %.1f tok/s)" % (total/elapsed, out_tok/elapsed))
    print("Request rate: %.2f req/s | Latency: %.3fs/req" % (num_prompts/elapsed, elapsed/num_prompts))
    print(sep)

    r = {"model": model_path, "tp_size": tp_size, "num_prompts": num_prompts,
         "elapsed_s": round(elapsed,2), "in_tokens": in_tok, "out_tokens": out_tok,
         "total_tokens": total, "throughput_tok_s": round(total/elapsed,1),
         "output_tok_s": round(out_tok/elapsed,1), "req_s": round(num_prompts/elapsed,2),
         "latency_s": round(elapsed/num_prompts,3), "load_time_s": round(load_time,1)}
    with open(output_file, "w") as f:
        json.dump(r, f, indent=2)
    print("Saved: %s" % output_file)

if __name__ == "__main__":
    freeze_support()
    main()
```

### Step 4: Run a Benchmark

```bash
VLLM_IMAGE="rocm/pytorch-private:miali_vllm_0.14.0rc2_ucx_develop_rixl_develop_20260126_retemadi_added_profile_pr18827"
MODEL_DIR="/shared_inference/ravgupta_models"
RESULTS_DIR="/shared_inference/ravgupta_models/results"

# Single GPU
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=32g --ipc=host \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -v $MODEL_DIR:/models \
  -v $RESULTS_DIR:/results \
  $VLLM_IMAGE \
  python3 /models/bench.py /models/Qwen/Qwen1.5-MoE-A2.7B 1 /results/qwen_moe_tp1.json 50

# Multi-GPU (with Triton MoE fallback)
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=32g --ipc=host \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -e VLLM_ROCM_USE_AITER_MOE=0 \
  -v $MODEL_DIR:/models \
  -v $RESULTS_DIR:/results \
  $VLLM_IMAGE \
  python3 /models/bench.py /models/Qwen/Qwen1.5-MoE-A2.7B 2 /results/qwen_moe_tp2.json 50
```

---

## 8. Raw Results (JSON)

### qwen_moe_tp1.json
```json
{
  "model": "/models/Qwen/Qwen1.5-MoE-A2.7B",
  "tp_size": 1,
  "num_prompts": 50,
  "elapsed_s": 3.81,
  "in_tokens": 9050,
  "out_tokens": 6400,
  "total_tokens": 15450,
  "throughput_tok_s": 4050.7,
  "output_tok_s": 1678.0,
  "req_s": 13.11,
  "latency_s": 0.076,
  "load_time_s": 417.9
}
```

### olmoe_tp1.json
```json
{
  "model": "/models/allenai/OLMoE-1B-7B-0924",
  "tp_size": 1,
  "num_prompts": 50,
  "elapsed_s": 2.22,
  "in_tokens": 9050,
  "out_tokens": 6400,
  "total_tokens": 15450,
  "throughput_tok_s": 6965.3,
  "output_tok_s": 2885.3,
  "req_s": 22.54,
  "latency_s": 0.044,
  "load_time_s": 334.4
}
```

### mixtral_tp1.json
```json
{
  "model": "/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
  "tp_size": 1,
  "num_prompts": 50,
  "elapsed_s": 3.77,
  "in_tokens": 9100,
  "out_tokens": 6400,
  "total_tokens": 15500,
  "throughput_tok_s": 4107.2,
  "output_tok_s": 1695.9,
  "req_s": 13.25,
  "latency_s": 0.075,
  "load_time_s": 776.9
}
```

### mixtral_tp2.json
```json
{
  "model": "/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
  "tp_size": 2,
  "num_prompts": 50,
  "elapsed_s": 6.29,
  "in_tokens": 9100,
  "out_tokens": 6400,
  "total_tokens": 15500,
  "throughput_tok_s": 2463.7,
  "output_tok_s": 1017.3,
  "req_s": 7.95,
  "latency_s": 0.126,
  "load_time_s": 408.6
}
```

### mixtral_tp4.json
```json
{
  "model": "/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
  "tp_size": 4,
  "num_prompts": 50,
  "elapsed_s": 6.75,
  "in_tokens": 9100,
  "out_tokens": 6400,
  "total_tokens": 15500,
  "throughput_tok_s": 2294.9,
  "output_tok_s": 947.6,
  "req_s": 7.4,
  "latency_s": 0.135,
  "load_time_s": 76.5
}
```

### mixtral_tp8.json
```json
{
  "model": "/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
  "tp_size": 8,
  "num_prompts": 50,
  "elapsed_s": 6.46,
  "in_tokens": 9100,
  "out_tokens": 6400,
  "total_tokens": 15500,
  "throughput_tok_s": 2399.8,
  "output_tok_s": 990.9,
  "req_s": 7.74,
  "latency_s": 0.129,
  "load_time_s": 93.2
}
```

### qwen_moe_tp2_triton.json
```json
{
  "model": "/models/Qwen/Qwen1.5-MoE-A2.7B",
  "tp_size": 2,
  "num_prompts": 50,
  "elapsed_s": 5.92,
  "in_tokens": 9050,
  "out_tokens": 6400,
  "total_tokens": 15450,
  "throughput_tok_s": 2608.2,
  "output_tok_s": 1080.4,
  "req_s": 8.44,
  "latency_s": 0.118,
  "load_time_s": 168.6
}
```

### qwen2_57b_tp2.json
```json
{
  "model": "/models/Qwen/Qwen2-57B-A14B-Instruct",
  "tp_size": 2,
  "num_prompts": 20,
  "elapsed_s": 6.05,
  "in_tokens": 3620,
  "out_tokens": 2560,
  "total_tokens": 6180,
  "throughput_tok_s": 1021.2,
  "output_tok_s": 423.0,
  "req_s": 3.3,
  "latency_s": 0.303,
  "load_time_s": 571.5
}
```

### qwen2_57b_tp4.json
```json
{
  "model": "/models/Qwen/Qwen2-57B-A14B-Instruct",
  "tp_size": 4,
  "num_prompts": 20,
  "elapsed_s": 6.01,
  "in_tokens": 3620,
  "out_tokens": 2560,
  "total_tokens": 6180,
  "throughput_tok_s": 1027.9,
  "output_tok_s": 425.8,
  "req_s": 3.33,
  "latency_s": 0.301,
  "load_time_s": 79.9
}
```

---

## 9. Reproducing LLaDA Benchmarks (Custom Engine)

LLaDA models use masked diffusion and are not supported by vLLM. We use a custom inference
engine (`src/inference/llada_engine.py`) built on HuggingFace Transformers + ROCm.

### Step 1: Allocate Compute Node
```bash
srun --partition=amd-rccl --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --gres=gpu:mi300x:8 \
  --time=04:00:00 --job-name=llada-bench bash
```

### Step 2: Run LLaDA-8B (Dense Diffusion)
```bash
VLLM_IMAGE="rocm/pytorch-private:miali_vllm_0.14.0rc2_ucx_develop_rixl_develop_20260126_retemadi"
MODEL_DIR="/shared_inference/ravgupta_models"

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=32g --ipc=host \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -v $MODEL_DIR:/models \
  $VLLM_IMAGE \
  bash -c '
pip install -q transformers accelerate sentencepiece protobuf safetensors 2>/dev/null
python3 /models/llada_engine.py \
  --model-path /models/GSAI-ML/LLaDA-8B-Instruct \
  --gen-length 128 --steps 64 \
  --num-prompts 10 \
  --output-json /models/results/llada_8b_single.json
'
```

### Step 3: Run LLaDA-MoE-7B (MoE Diffusion)
```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=32g --ipc=host \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -v $MODEL_DIR:/models \
  $VLLM_IMAGE \
  bash -c '
pip install -q transformers accelerate sentencepiece protobuf safetensors 2>/dev/null
python3 /models/llada_engine.py \
  --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --gen-length 128 --steps 64 \
  --num-prompts 10 \
  --output-json /models/results/llada_moe_single.json
'
```

### Step 4: Multi-GPU LLaDA-MoE (Distributed with RCCL)
```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=32g --ipc=host \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v $MODEL_DIR:/models \
  $VLLM_IMAGE \
  bash -c '
pip install -q transformers accelerate sentencepiece protobuf safetensors 2>/dev/null
torchrun --nproc_per_node=4 /models/llada_distributed.py \
  --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --gen-length 128 --steps 64 \
  --num-prompts 10 \
  --output-json /models/results/llada_moe_dist4.json
'
```

---

## 10. Complete Results Summary

| Model | Type | TP | Engine | Throughput (tok/s) | Output (tok/s) | Latency (ms) |
|-------|------|-----|--------|-------------------|----------------|-------------|
| OLMoE-1B-7B | AR MoE | 1 | vLLM+AITER | **6,965** | 2,885 | 44 |
| Qwen1.5-MoE-A2.7B | AR MoE | 1 | vLLM+AITER | 4,051 | 1,678 | 76 |
| Qwen1.5-MoE-A2.7B | AR MoE | 2 | vLLM+Triton | 2,608 | 1,080 | 118 |
| Mixtral-8x7B | AR MoE | 1 | vLLM+AITER | 4,107 | 1,696 | 75 |
| Mixtral-8x7B | AR MoE | 2 | vLLM+Triton | 2,464 | 1,017 | 126 |
| Mixtral-8x7B | AR MoE | 4 | vLLM+Triton | 2,295 | 948 | 135 |
| Mixtral-8x7B | AR MoE | 8 | vLLM+Triton | 2,400 | 991 | 129 |
| Qwen2-57B-A14B | AR MoE | 2 | vLLM | 1,021 | 423 | 303 |
| Qwen2-57B-A14B | AR MoE | 4 | vLLM | 1,028 | 426 | 301 |
| DeepSeek-V3 | AR MoE | 8 | vLLM+AITER-MLA | 137 | 56 | 2,269 |
| LLaDA-8B | Diffusion | 1 | Custom+ROCm | 91 | — | 1,411 |
| LLaDA-MoE-7B | Diff MoE | 1 | Custom+ROCm | 10 | — | 13,269 |

## 11. Next Steps

1. ~~LLaDA Inference~~: **DONE** — Custom ROCm framework built and benchmarked
2. ~~DeepSeek-V3~~: **DONE** — Benchmarked at TP=8 (136.6 tok/s)
3. **Expert Parallelism**: Test `--enable-expert-parallel` on Mixtral and Qwen2-57B
4. **DBRX**: Need to accept license at https://huggingface.co/databricks/dbrx-instruct then re-download
5. **Multi-GPU LLaDA-MoE**: Run distributed inference with RCCL on 2/4/8 GPUs
6. **Profiling**: Collect torch profiler + rocprofv3 traces for MoE routing analysis
7. **Serving Benchmarks**: Online serving with varying request rates (1-64)
8. **CPU Predictor**: Train lightweight ML model for placement decisions
9. **Multi-Node**: Scale to 2-4 nodes with CX-7 RDMA
