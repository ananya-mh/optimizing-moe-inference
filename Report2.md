# Report 2 — Related Work, MI355X + AI NIC, MoRI, and 1M–2M Token Contexts for LLaDA

This report surveys work related to our LLaDA experiments (Fast-dLLM, the LLaDA family, dLLM caching
and serving systems), summarizes the AMD MI355X + Pensando Pollara 400 AI NIC (AINIC) platform and the
MoRI communication library (MoRI-EP, MoRI-IO), and analyzes what it takes to run LLaDA-8B and
LLaDA-MoE-7B at 1M and 2M token contexts. It ends with a concrete engine roadmap and experiment plan.

All capacity/latency figures in Section 6 are **analytical estimates** (derived from model configs
and hardware specs), not measurements. They are there to size experiments, not to be quoted as results.

---

## 1. TL;DR

1. **Report1's LLaDA-MoE slowdown is a software artifact, not an MoE property.** The Hugging Face
   `modeling_lladamoe.py` MoE block loops over all 64 experts in Python for each of the 16 layers
   (`for expert_idx in range(self.num_experts)` + `torch.where` + `index_add_`), i.e. ~1,024 small
   kernel groups and host syncs per forward pass. dInfer users report 40–58 tok/s for the same model on
   a single GPU vs our 9.6 tok/s. The claim "9.4x slower than dense validates EP" should not go in the
   paper until we re-measure with a fused MoE kernel (AITER `fused_moe` or Triton).
2. **The repo had LLaDA-MoE-7B wrong** (8 experts / top-2). The real config is 64 experts, top-8,
   16 layers, hidden 2048, 16 KV heads, max 8K positions. Fixed in `configs/models.yaml` and `README.md`.
3. **The dLLM ecosystem has moved past "vLLM doesn't support LLaDA".** SGLang now serves LLaDA2.0/2.1
   (MoE diffusion, up to 100B) with KV cache, TP/EP, CUDA/HIP graphs, and documents MI300X/MI325X/MI355X.
   inclusionAI's dInfer runs LLaDA-MoE at >1,100 tok/s (8x H800, batch 1). These are the baselines a
   reviewer will expect us to compare against.
4. **Our engine is missing the two optimizations that matter most**: KV caching and confidence-threshold
   parallel decoding (Fast-dLLM, up to 27.6x). At long context, these are mandatory — without a prefix
   cache, every denoising step re-encodes the full prompt.
5. **1M–2M contexts are a systems problem we can study, but no dLLM is trained for them.** LLaDA-8B is
   trained at 4K, LLaDA-MoE at 4K/8K; the best published extension is UltraLLaDA at 128K. At 1M/2M we
   can measure memory, prefill time, per-step latency and throughput; quality beyond ~128K must be
   reported honestly as degraded.
6. **Capacity fits on one 8x MI355X node**: LLaDA-8B at 2M tokens needs ~1.05 TB of BF16 KV (0.52 TB in
   FP8); the node has 2.3 TB HBM. LLaDA-MoE at 1M needs ~131 GB of KV and fits on a single MI355X.
7. **Key research hypothesis for long context**: at 1M+ tokens decoding is KV-bandwidth-bound. An AR
   model reads the whole KV cache once per token; a dLLM reads it once per denoising step and commits
   several tokens per step (tokens-per-forward, TPF ≈ 2–5 with threshold decoding). dLLMs should
   therefore gain relative to AR as context grows — something we can measure directly on MI355X.

---

## 2. Diffusion LLM Landscape (What Exists Today)

### 2.1 LLaDA family

| Model | Params (total / active) | Architecture | Context | Notes |
|-------|------------------------|--------------|---------|-------|
| [LLaDA-8B](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | 8B / 8B | Dense, 32 layers, d=4096, **32 KV heads (full MHA)**, RoPE θ=500K | 4K | The original masked-diffusion LLM; fully bidirectional attention |
| [LLaDA 1.5](https://aclanthology.org/2026.acl-long.524/) | 8B / 8B | Same as LLaDA-8B | 4K | VRPO preference optimization; GSM8K +4.7, HumanEval +3.0, IFEval +4.0 (ACL 2026) |
| [LLaDA-V](https://arxiv.org/abs/2505.16933) | ~8B + vision | LLaDA-8B + SigLIP2-so400m + MLP connector | 4K | Pure diffusion multimodal LLM (CVPR 2026); competitive with LLaMA3-V on same data |
| [LLaDA-MoE-7B-A1B](https://arxiv.org/abs/2509.24389) | 7B / 1.4B | 16 layers, d=2048, 16 KV heads, **64 experts, top-8**, expert dim 1024, RoPE θ=50K | 4K pretrain, 8K anneal | First open MoE dLLM; trained on ~20T tokens; beats LLaDA-8B / Dream-7B on average |
| [LLaDA2.0-mini / flash](https://github.com/inclusionAI/dInfer) | 16B / 1.4B and 100B / 6.1B | MoE, block diffusion | 32K | Served by SGLang and dInfer |
| [LLaDA2.1-mini / flash](https://docs.sglang.io/cookbook/autoregressive/InclusionAI/LLaDA-2.1.md) | 16B and 100B | MoE, block diffusion, self-correcting (JointThreshold) | 32K | SGLang recipe lists MI300X / MI325X / MI355X |
| [LongLLaDA](https://arxiv.org/abs/2506.14429) | LLaDA-8B | Training-free NTK RoPE scaling | ~24K (6x) | Finds dLLMs keep stable perplexity beyond training length ("local perception") |
| [UltraLLaDA](https://arxiv.org/abs/2510.10481) | LLaDA-8B | Diffusion-aware NTK + post-training on 64K packed PG19 | **128K** | Uses T_cap ≈ 2·T_train because bidirectional attention sees relative positions in [-(T-1), T-1]; document-boundary masking is important |

### 2.2 Other diffusion LLMs we should know about

- **Dream-7B** — AR-initialized (Qwen2.5) dLLM, standard comparison point with LLaDA.
- **SDAR-8B / SDAR-30B-A3B** (JetLM) — block-diffusion, dense and MoE; supported in SGLang.
- **DiffusionGemma-26B-A4B** — uniform-state (renoising) block-diffusion multimodal MoE; in SGLang.
- **Fast-dLLM v2** — see 3.2; converts Qwen2.5 AR models to block diffusion with ~1B tokens.
- **Fast-dVLM** ([paper](https://arxiv.org/abs/2604.06832)) — Qwen2.5-VL-3B converted to block diffusion;
  6.18x speedup with SGLang + FP8; notes that LLaDA-V / LaViDa / MMaDA / Dimple use full-sequence
  diffusion, which blocks incremental KV caching.
- **D2F (Discrete Diffusion Forcing)** ([paper](https://arxiv.org/abs/2508.09192)) — distills LLaDA/Dream
  into an AR-diffusion hybrid with inter-block pipelining; up to 52.9x over vanilla LLaDA on MBPP and
  2.5x faster than LLaMA3-8B on GSM8K.

**Why block diffusion matters for us:** LLaDA-8B and LLaDA-MoE use *fully bidirectional* attention, so
prompt-token KV technically changes whenever generated tokens change. Block-diffusion models (LLaDA2.x,
SDAR, Fast-dLLM v2) use block-causal attention, so prefix KV is exact and never needs refreshing. That
is the property that makes 1M-token contexts affordable (Section 6).

---

## 3. dLLM Acceleration Work

### 3.1 Fast-dLLM v1 (training-free) — [paper](https://arxiv.org/abs/2505.22618), [code](https://github.com/NVlabs/Fast-dLLM/tree/main/v1)

Two ideas, both directly applicable to our engine:

1. **Block-wise approximate KV cache.** *PrefixCache* caches KV of everything before the current block
   and refreshes it once per block. *DualCache* also caches the masked suffix. Speedup 2–3.6x alone.
2. **Confidence-aware parallel decoding.** Instead of unmasking a fixed top-k per step (what our engine
   does), unmask every token whose max softmax probability exceeds a threshold (default 0.9), always
   committing at least one. Speedup 4–6x alone.

Combined results on LLaDA (A100, their numbers):

| Benchmark | Gen len | LLaDA tok/s | + Cache | + Parallel | Fast-dLLM | Accuracy (base → Fast-dLLM) |
|-----------|---------|-------------|---------|------------|-----------|-----------------------------|
| GSM8K 5-shot | 256 | 6.7 | 21.2 (3.2x) | 16.5 (2.5x) | 54.4 (8.1x) | 79.3 → 78.5 |
| GSM8K 5-shot | 512 | 3.2 | 10.4 (3.3x) | 18.6 (5.8x) | 35.3 (11.0x) | 77.5 → 77.2 |
| HumanEval | 512 | 18.4 | 29.3 (1.6x) | 57.1 (3.1x) | 73.7 (4.0x) | 43.9 → 44.5 |

Defaults: PrefixCache, block size 32, threshold 0.9. Up to 27.6x end-to-end in the best setting.

### 3.2 Fast-dLLM v2 (ICLR 2026) — [project page](https://nvlabs.github.io/Fast-dLLM/v2/), [paper](https://arxiv.org/abs/2509.26328)

- Converts a pretrained **AR model into a block-diffusion LLM with ~1B tokens** of fine-tuning (vs 580B
  for Dream). Models at 1.5B and 7B (Qwen2.5-based).
- **Block-wise causal mask**: attend to all clean tokens of previous blocks plus noisy tokens of the
  current block. **Token shift**: each masked token is predicted from the logit of the preceding token,
  preserving AR representations. **Complementary masks** so every position gets trained.
- **Hierarchical cache**: block-level cache for finished blocks + sub-block DualCache inside the
  current block; sub-blocks decoded in parallel.
- Up to **2.5x faster than AR Qwen2.5-7B-Instruct**; 102.5 tok/s at batch 1, 217.5 tok/s at batch 4
  (A100); highest 7B-class average (60.3) across HumanEval/MBPP/GSM8K/MATH/IFEval/MMLU/GPQA.

**Relevance for long context:** because the recipe starts from an AR model, it could inherit an AR
model's long-context ability. Applying it to a 1M-context AR model (e.g. Qwen2.5-7B-Instruct-1M) is a
plausible route to a *genuinely* 1M-capable dLLM. Nobody has published this; it is a research idea,
not an established result.

### 3.3 Other caching / decoding methods

| Method | Idea | Reported gain | Long-context relevance |
|--------|------|---------------|------------------------|
| [dKV-Cache](https://arxiv.org/abs/2505.15781) (NeurIPS 2025) | Delay caching a token's KV by one step after it is decoded | 2–10x | Greedy variant cuts complexity from O(L³) to O(L²) |
| [dLLM-Cache](https://arxiv.org/abs/2506.06295) | Prompt KV refreshed at long intervals, response adaptively (V-verify); also caches attention/FFN outputs | Up to 9.1x FLOPs reduction (LongBench HotpotQA) | **Directly addresses long prompts** — refresh the 1M prompt rarely |
| [d2Cache](https://arxiv.org/abs/2509.23094) | Fine-grained per-token selection of which KV to refresh | Better speed and quality than dLLM-Cache / Fast-dLLM | — |
| [Sparse-dLLM](https://github.com/OpenMOSS/Sparse-dLLM) | Attention-guided cache eviction + sparse attention; dLLM attention sparsity is stable across steps | Up to 10x, near-baseline peak memory | **Most relevant for 1M+**: bounds KV memory |
| Elastic-Cache | Layer-aware refresh triggered by attention drift | Up to 45.1x (reported) | — |
| D2F | Distillation to AR-diffusion hybrid, inter-block pipelining | Up to 52.9x vs vanilla LLaDA | Requires distillation training |

### 3.4 dLLM serving systems (our real baselines)

- **dInfer** ([paper](https://arxiv.org/abs/2510.08666), [code](https://github.com/inclusionAI/dInfer)) —
  modular framework (model / diffusion iteration manager / decoder / KV-cache manager). On LLaDA-MoE:
  >1,100 tok/s on HumanEval and ~800 tok/s average over six benchmarks at **batch 1 on 8x H800**, 10x
  over Fast-dLLM, 2–3x over Qwen2.5-3B on vLLM. Supports TP and EP, threshold / hierarchical decoders,
  dual cache, "credit" decoding, and trajectory-distilled LLaDA-MoE-TD. Built on PyTorch + vLLM parallel
  layers. Community reproductions on single A100/H100 report 40–58 tok/s, and the 1,000+ tok/s figure
  depends on their dataset script and settings ([issue #8](https://github.com/inclusionAI/dInfer/issues/8)).
  ROCm support is not documented; porting is plausible because its dependencies (PyTorch, vLLM) run on
  ROCm, but this is unverified.
- **SGLang dLLM** ([docs](https://docs.sglang.io/docs/supported-models/diffusion_language_models)) —
  LLaDA2.0/2.1, SDAR, DiffusionGemma. `--dllm-algorithm LowConfidence | JointThreshold`, KV cache,
  streaming, batching, TP/EP, graphs, and **FDFO scheduling** (requests leave the batch as soon as
  their block resolves, avoiding head-of-line blocking). LLaDA2.1 cookbook lists MI300X/MI325X/MI355X.
  Note: SGLang serves the LLaDA2.x (block-diffusion) architecture, not the original LLaDA-8B / LLaDA-MoE
  checkpoints we benchmarked.

---

## 4. AMD MI355X + Pollara 400 AI NIC

### 4.1 MI355X (CDNA4, gfx950) — [product page](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html), [arch docs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)

| Spec | MI300X | MI355X |
|------|--------|--------|
| HBM | 192 GB HBM3 | **288 GB HBM3E** |
| Memory bandwidth | 5.3 TB/s | **8.0 TB/s** |
| Compute units | 304 | 256 (160 KB LDS per CU vs 64 KB) |
| BF16 dense matrix | 1.3 PF | **2.5 PF** |
| FP8 dense matrix | 2.6 PF | **5 PF** (OCP FP8) |
| MXFP4 / MXFP6 | — | **10 PF** (native) |
| Infinity Fabric | 7 links, 896 GB/s P2P ring | 7 links x 153.6 GB/s, 1,075 GB/s P2P ring |
| Node (8 GPUs) | 1.5 TB HBM | **2.3 TB HBM** |
| Power | 750 W | 1,400 W (liquid) |

**Porting gotcha:** MI300X uses FP8 *FNUZ* (`torch.float8_e4m3fnuz`); MI355X uses *OCP* FP8
(`torch.float8_e4m3fn`). Any FP8 KV cache, FP8 MoRI dispatch, or FP8 GEMM code written for MI300X must
switch dtypes. Also set `MORI_GPU_ARCHS=gfx950` when building MoRI, or it may pick gfx942.

### 4.2 Pollara 400 AI NIC — [product brief](https://www.amd.com/content/dam/amd/en/documents/pensando-technical-docs/product-briefs/pensando-pollara-400-product-brief.pdf), [ops guide UG1801](https://docs.amd.com/r/en-US/ug1801-ai-nic-pollara-400-ops-guide/RCCL-ROCm-and-ANP-in-AI-NIC)

- 400 Gbps RDMA Ethernet per NIC; reference MI355X nodes have **8 NICs, 1:1 GPU-to-NIC mapping**
  (3.2 Tbps / ~400 GB/s scale-out per node). Linux RDMA devices appear as `ionic_0..7`.
- **UEC-ready RDMA**: path-aware adaptive **packet spraying** (AMD recommends Mode 2 = spray at the
  sender NIC + ECMP hashing on switches), selective ACK / fast retransmit, programmable congestion
  control. Also RoCEv2-compatible.
- **RCCL** uses the AINIC via the **ANP plugin** (built into ROCm from 7.2.1). UEC-mode RCCL adds:
  `IONIC_RCQ_NUM_PATHS=1 IONIC_PRIVATE_SERVICE_FORCE=1 IONIC_RCQ_SIGN_BIT=15 NCCL_IB_TIMEOUT=5
  NCCL_PXN_DISABLE=1 RCCL_LL128_FORCE_ENABLE=1`, plus `NCCL_IB_HCA=ionic_0,...,ionic_7`.
- **Fabric QoS** must be lossless for RoCE traffic: DSCP classification, PFC no-drop on the RoCE
  priority, DCQCN, with matching switch config. Reference defaults: RoCE on priority 3 / DSCP 26,
  CNP on priority 6 / DSCP 48. MoRI then needs `MORI_RDMA_SL=<no-drop priority>` and
  `MORI_RDMA_TC=<DSCP * 4>`. Check `nicctl show qos` before trusting any inter-node number.
- **Published results**: AMD reports an MI355X **1P2D, EP8** configuration (3 nodes, SGLang + MoRI) with
  higher per-GPU throughput than NVL72 + Dynamo at similar interactivity for DeepSeek-R1 1K/1K
  ([AMD article](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html),
  [SGLang + MoRI recipe](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/distributed/sglang-mori-recipe.html)).

---

## 5. MoRI — Modular RDMA Interface ([repo](https://github.com/ROCm/mori), [docs](https://rocm.github.io/mori/))

MoRI is AMD's GPU-centric communication stack with three parts: **MoRI-SHMEM** (symmetric-heap,
GPU-initiated RDMA/XGMI, the foundation), **MoRI-EP** (MoE dispatch/combine), and **MoRI-IO**
(point-to-point transfer, mainly KV cache). It supports Pollara (`USE_IONIC`, auto-detected via
`libionic.so`), Broadcom Thor2 (`USE_BNXT`), and Mellanox. An optional DeepEP-compatible API exists
(`ENABLE_STANDARD_MOE_ADAPT`).

### 5.1 MoRI-EP — [guide](https://rocm.github.io/mori/MORI-EP-GUIDE.html)

API: `EpDispatchCombineConfig` + `EpDispatchCombineOp` with `dispatch(input, weights, scales, indices)`
→ `(out, weights, scales, indices, recv_count)` and `combine(expert_out, weights, indices)`. Split
`dispatch_send/recv` and `combine_send/recv` exist for overlapping communication with compute.

| Kernel type | Topology | Use case |
|-------------|----------|----------|
| `IntraNode` | XGMI | EP within one 8-GPU node |
| `InterNode` | XGMI + RDMA | Baseline cross-node |
| `InterNodeV1` | XGMI + RDMA | Higher-bandwidth cross-node (prefill, large batches) |
| `InterNodeV1LL` | XGMI + RDMA | Low-latency (decode, small batches) |
| `AsyncLL` | XGMI + RDMA | Async low-latency, pipelined |

Reference bandwidths from the guide: EP8 IntraNode ~307 GB/s dispatch / 330 GB/s combine (XGMI);
EP16 InterNodeV1 ~63 GB/s RDMA dispatch; EP32 InterNodeV1LL ~57 GB/s RDMA dispatch.

Key knobs: `block_num`, `warp_num_per_block`, `rdma_block_num`, `num_qp_per_pe`,
`max_num_inp_token_per_rank`; env `MORI_SHMEM_HEAP_SIZE` (default 4G; 16G needed for large dispatch
budgets), `MORI_RDMA_DEVICES`, `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` (uses pre-tuned launch configs; a
tuner writes JSON configs per kernel/dtype/token count). Integrated in AITER (`MoriAll2AllManager`),
SGLang, and vLLM.

**Mapping to LLaDA-MoE-7B** (64 experts, top-8, hidden 2048): EP8 gives 8 experts per rank,
`hidden_dim=2048, num_experts_per_rank=8, num_experts_per_token=8`, `IntraNode` kernel. Honest caveat:
the whole model is ~14 GB and fits on one GPU, so at batch 1 with 32-token blocks EP adds 32
dispatch/combine round-trips per forward (16 layers x 2) for very little compute per rank; it will
likely *hurt* latency. EP pays off for (a) large batches, (b) long-context prefill where each forward
carries up to ~1M tokens, and (c) bigger diffusion MoEs such as **LLaDA2.0/2.1-flash (100B)**, which
we should add as the EP target.

### 5.2 MoRI-IO — [guide](https://github.com/ROCm/mori/blob/main/docs/MORI-IO-GUIDE.md)

`IOEngine` with RDMA / XGMI / TCP backends; `register_torch_tensor`, one-sided `read`/`write`,
`batch_read/batch_write`, reusable sessions, async `TransferStatus`. In vLLM it is the ROCm-only
**`MoRIIOConnector`** for prefill/decode (PD) disaggregation
([usage](https://docs.vllm.ai/en/latest/features/moriio_connector_usage/)):

- **Write mode** (default): the proxy sends the request to prefill and decode at the same time; prefill
  pushes KV **layer by layer** into decode's pre-allocated blocks with RDMA WRITE.
- **Read mode** (`VLLM_MORIIO_CONNECTOR_READ_MODE=1`): decode pulls KV after prefill completes.
- vLLM reports **2.5x goodput** on a single 8x MI300X node using 1P+1D (TP4 each) vs two collocated
  TP4 instances (Qwen3-235B-A22B-FP8, 2K in / 1K out, 8 req/s, SLO TTFT < 1 s and ITL < 50 ms)
  ([blog](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)).

**Relevance to 1M contexts:** a 1M-token LLaDA-8B prompt takes about a minute of prefill on 8 GPUs
(Section 6), and the resulting 524 GB of KV (BF16) would take ~1.3 s to move over one node's 8x400G
NICs. So PD disaggregation is cheap relative to prefill and keeps long prefills from stalling other
users' denoising steps.

### 5.3 MoRI-SHMEM vs rocSHMEM vs RCCL

| Library | Model | When to use |
|---------|-------|-------------|
| **RCCL** (+ ANP for AINIC) | Host-launched collectives | TP all-reduce, all-gather, baseline all-to-all |
| **MoRI-SHMEM** | GPU-initiated, symmetric heap; powers MoRI-EP/IO | Custom fused comm+compute kernels in our stack; same heap as MoRI-EP |
| **rocSHMEM** ([docs](https://rocm.docs.amd.com/projects/rocSHMEM/en/develop/introduction.html)) | OpenSHMEM-like; IPC, reverse-offload, and **GDA** backends (`ROCSHMEM_BACKEND=gda`, `ROCSHMEM_GDA_PROVIDER=ionic`); multi-NIC per PE in development | Standard OpenSHMEM semantics; GDA backend is "as-is / limited support" per AMD |

Recommendation: RCCL for TP collectives, MoRI-EP for expert dispatch, MoRI-IO for KV movement, and
MoRI-SHMEM (not rocSHMEM) if we write custom device-initiated kernels, so everything shares one
symmetric heap and one NIC setup.

---

## 6. Running LLaDA at 1M and 2M Tokens

### 6.1 Model facts that drive the math (from the HF `config.json` files)

| | LLaDA-8B | LLaDA-MoE-7B |
|--|----------|--------------|
| Layers / hidden | 32 / 4096 | 16 / 2048 |
| Attention heads / KV heads | 32 / **32** (no GQA) | 16 / **16** (no GQA) |
| KV bytes per token (BF16) | **512 KiB** | **128 KiB** |
| Trained context | 4K | 4K pretrain, 8K anneal, 4K SFT |
| RoPE θ | 500,000 | 50,000 |

Neither model uses GQA, so the KV cache is large, and TP can split it by head (32 or 16 ways) without
duplication.

### 6.2 Memory

| Context | LLaDA-8B KV BF16 / FP8 | LLaDA-MoE KV BF16 / FP8 | Minimum MI355X count (KV + weights, BF16) |
|---------|------------------------|-------------------------|---------------------------------------------|
| 128K | 69 GB / 34 GB | 17 GB / 9 GB | 1 / 1 |
| **1M** | **524 GB / 262 GB** | **131 GB / 66 GB** | 2–4 (8B), **1** (MoE) |
| **2M** | **1,049 GB / 524 GB** | **262 GB / 131 GB** | 4–8 (8B), 1–2 (MoE) |

Both models at 2M fit on **one 8x MI355X node** (2.3 TB). Multi-node is needed only to cut prefill
latency (context parallelism across nodes) or to serve several million-token requests at once.

### 6.3 Compute: prefill is the wall

dLLM prompt attention is bidirectional, so there is **no causal-mask 2x saving** that AR prefill gets.
Attention FLOPs ≈ 4·L²·d·layers:

| Context | LLaDA-8B prefill FLOPs | Time @ ~1 PF/s effective (1 GPU / 8 GPUs) | LLaDA-MoE prefill FLOPs | Time (1 / 8 GPUs) |
|---------|------------------------|--------------------------------------------|-------------------------|-------------------|
| 128K | 1.1e16 | 11 s / 1.4 s | 2.6e15 | 3 s / 0.3 s |
| 1M | 5.4e17 | ~9 min / ~68 s | 1.3e17 | ~2.2 min / ~17 s |
| 2M | 2.1e18 | ~36 min / ~4.4 min | 5.3e17 | ~9 min / ~66 s |

(~1 PF/s ≈ 40% of MI355X BF16 dense peak for flash attention; FP8 attention could roughly halve these.)

**Consequence 1: the current engine cannot run long contexts at all.** It re-runs the full sequence on
every denoising step, so a 1M prompt with 64 steps would cost 64 prefills (~10 hours on one GPU for
LLaDA-8B). A prefix KV cache is a hard requirement.

**Consequence 2: Fast-dLLM's per-block cache refresh does not scale to 1M.** Fast-dLLM PrefixCache
recomputes prefix KV at every block boundary because, in a bidirectional model, prompt KV depends on
the generated tokens. At 1M context that is a full prefill per block. Options, in order of preference:

1. **Freeze prompt KV** after one prefill, refreshing only the generated region (dLLM-Cache style with an
   infinite prompt-refresh interval). This is an approximation for LLaDA-8B / LLaDA-MoE; measure its
   accuracy impact at short contexts where exact recomputation is affordable.
2. **Use block-causal dLLMs** (LLaDA2.x, SDAR, Fast-dLLM v2) where prefix KV is exact by construction.
3. **Bound the KV set** with Sparse-dLLM-style eviction / sparse attention.

### 6.4 Decode: where dLLMs should win at long context

With a frozen prefix cache, each denoising step processes one block (e.g. 32 query tokens) against
the full KV. Compute is small; the step is bound by **reading the KV cache from HBM**:

| Context | LLaDA-8B per-step KV read (1 GPU / 8 GPUs, ~6 TB/s achieved) | LLaDA-MoE (1 / 8 GPUs) |
|---------|------------------------------------------------------------------|------------------------|
| 1M | ~87 ms (doesn't fit) / ~11 ms | ~22 ms / ~2.7 ms |
| 2M | ~175 ms (doesn't fit) / ~22 ms | ~44 ms / ~5.5 ms |

An AR model at the same context also pays one full KV read, but per token. A dLLM pays it per step
and commits TPF tokens per step (our fixed schedule: 2; Fast-dLLM threshold decoding: ~3–5; dInfer
reports ~3 on average). So long-context decode throughput ≈ TPF / step_time, which gives the dLLM a
TPF-fold advantage when KV reads dominate. **This is the headline experiment for the long-context part
of the paper**: plot tokens/s vs context (32K → 2M) for LLaDA vs an AR baseline on the same MI355X
node, and show the crossover.

### 6.5 Parallelism for 1M–2M

- **Head-parallel TP** (TP8 for LLaDA-8B: 4 KV heads per GPU; LLaDA-MoE: 2 per GPU) already shards
  KV with no duplication, since there is no GQA. Simplest first step.
- **Decode context parallelism (DCP)** shards KV along the sequence: each rank holds 1/N of the
  tokens, computes partial attention for the query block, then ranks merge `(output, log-sum-exp)`.
  The merge message is tiny for dLLMs (32 tokens x heads x 128 x layers). This is the scheme vLLM and
  AMD ATOM expose as `-dcp`
  ([vLLM](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/),
  [ATOM](https://rocm.docs.amd.com/projects/atom/en/latest/context_parallel_guide.html)).
- **Prefill context parallelism (ring attention, pass-KV)**: Meta reached 1M-token Llama3-405B
  prefill in 77 s on 128 H100s with 93% efficiency, over RDMA and even TCP
  ([paper](https://arxiv.org/abs/2411.01783)). For dLLMs, **bidirectional ring attention is naturally
  load-balanced** (every chunk attends to every chunk), so the zig-zag reordering causal models need
  isn't required. Cross-node rings run over Pollara RDMA.
- **Chunked prefill** is needed for activation memory: Qwen2.5-1M reports that 32K-token chunks cut MLP
  activation memory by 96.7% at 1M. For bidirectional models, chunk the *queries* while keeping all
  keys visible.
- **FP8 KV** (OCP e4m3 on MI355X) halves every memory and bandwidth number above.

### 6.6 How AR models reach 1M+ (for context and baselines)

| Model / technique | Mechanism | Notes |
|-------------------|-----------|-------|
| [Qwen2.5-7B/14B-Instruct-1M](https://qwenlm.github.io/blog/qwen2.5-1m/) | Dual Chunk Attention + YaRN; MInference vertical-slash sparse prefill; chunked prefill | 3.2–6.7x faster 1M prefill; 7B needs ~120 GB total VRAM. **Best AR baseline for us** (same size class as LLaDA-8B) |
| Llama 4 Scout (10M claimed) | iRoPE (interleaved no-RoPE layers) | vLLM validated ~1.2–1.6M on 8x H100 |
| MiniMax-M1 | Hybrid lightning (linear) + softmax attention, 456B MoE | Native 1M |
| DeepSeek-V3.2 (DSA) | Lightning indexer + top-2048 token selection; O(L·k) main attention | 128K supported; ROCm image `lmsysorg/sglang:dsv32-rocm` |
| Qwen3-Next, Kimi Linear | Hybrid linear attention (Gated DeltaNet / KDA) | Linear-time layers keep KV small |

### 6.7 Quality caveat

No published dLLM has been shown to work at 1M tokens. LongLLaDA reaches ~24K training-free;
UltraLLaDA reaches 128K after post-training. Beyond that we expect "local perception" behavior (the
model mostly uses the most recent window). The 1M/2M experiments should therefore be framed as
**systems experiments** (memory, latency, throughput, scaling) with NIAH accuracy reported as
measured, not as a claim of 1M-token understanding. A Fast-dLLM-v2-style conversion of a 1M-context
AR model is the most plausible path to a quality-preserving 1M dLLM, as future work.

---

## 7. Engine Roadmap (Custom ROCm dLLM Engine v2)

Ordered by expected impact per unit of effort. Items 1–4 are prerequisites for any fair comparison.

| # | Change | Why | Expected effect |
|---|--------|-----|-----------------|
| 1 | **Fused MoE kernel** (AITER `fused_moe` / Triton grouped GEMM) replacing the HF per-expert loop | Removes ~1,024 kernel groups + host syncs per forward | Large LLaDA-MoE speedup; required before any EP claim |
| 2 | **Flash attention** via SDPA / AITER CK FA; `attn_implementation="sdpa"` | HF LLaDA-8B config has `flash_attention: false` | Needed for >32K contexts |
| 3 | **Prefix + Dual KV cache** (Fast-dLLM), with a **frozen-prompt** mode for long context | Avoid re-encoding the prompt every step | 2–3.6x at short context; mandatory at long context |
| 4 | **Threshold parallel decoding** (0.9 default) + record TPF | Replaces fixed top-k per step | 2–6x; TPF is the key long-context metric |
| 5 | **HIP graphs** for fixed-shape denoise steps | Denoise steps have static shapes | Cuts launch overhead at small blocks |
| 6 | **Batching with FDFO scheduling** | Throughput under load; matches SGLang | Serving-level comparison |
| 7 | **MoRI-EP dispatch/combine** (IntraNode, then InterNodeV1/V1LL) | Real EP for LLaDA-MoE and LLaDA2.x-flash; replaces the all-reduce EP in `llada_distributed.py` | Enables the placement study for diffusion MoE |
| 8 | **Sequence-sharded KV (DCP) + LSE merge** over RCCL / MoRI-SHMEM; **bidirectional ring-attention prefill** | 1M–2M contexts across 8 GPUs / nodes | Makes 2M feasible, near-linear prefill scaling |
| 9 | **FP8 KV cache** (OCP e4m3) | Halves memory and bandwidth | 2M LLaDA-8B in 524 GB |
| 10 | **MoRI-IO PD disaggregation** | Isolate minute-long prefills from decode | Stable per-step latency under mixed load |

Baselines to run alongside: SGLang dLLM (LLaDA2.x) on MI355X; dInfer (if it ports to ROCm); vLLM +
Qwen2.5-7B-Instruct-1M as the AR long-context baseline; Fast-dLLM v1 reference code on LLaDA-8B.

---

## 8. Proposed Experiments

**E1 — Fix the LLaDA-MoE baseline.** Re-run Report1's LLaDA-8B vs LLaDA-MoE comparison with the
fused MoE kernel (roadmap item 1). Profile with rocprofv3 before and after to show where time went.
This decides whether the "EP is critical for diffusion MoE" claim survives.

**E2 — Fast-dLLM on ROCm.** LLaDA-8B and LLaDA-MoE with {no cache, prefix, dual} x {fixed top-k,
threshold 0.7/0.8/0.9}; GSM8K + HumanEval accuracy and tok/s at gen length 256/512. Compare to the
Fast-dLLM A100 table in 3.1.

**E3 — Diffusion MoE placement.** LLaDA-MoE and LLaDA2.x-flash under TP vs EP (MoRI-EP IntraNode) vs
TP+EP at batch {1, 8, 32, 128}. Feeds the factorial study and the CPU placement predictor.

**E4 — Long-context scaling (headline).** Context in {32K, 128K, 256K, 512K, 1M, 2M} x {LLaDA-8B,
LLaDA-MoE, Qwen2.5-7B-Instruct-1M} x {TP8, TP8+DCP} x {BF16, FP8 KV}. Measure prefill time, per-step
latency, TPF, decode tok/s, peak HBM, and NIAH accuracy. Expected: prefill ~O(L²), per-step ~O(L), dLLM
decode advantage ≈ TPF.

**E5 — Multi-node over AINIC.** Ring-attention prefill across 2–4 MI355X nodes (RCCL + ANP, UEC
mode vs RoCEv2), MoRI-EP InterNodeV1 vs V1LL for diffusion MoE, and MoRI-IO PD disaggregation for 1M
prompts. Validate fabric QoS and run `rccl-tests` / MoRI EP benchmarks first.

---

## 9. Reproduction Notes

```bash
# Model configs used for the sizing in Section 6
curl -sL https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/raw/main/config.json
curl -sL https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/raw/main/config.json

# Evidence for the per-expert Python loop (look for `for expert_idx in range(self.num_experts)`)
curl -sL https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/raw/main/modeling_lladamoe.py \
  | grep -n "for expert_idx"
```

KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element.
Bidirectional prefill attention FLOPs ≈ 4 x L² x hidden x layers; linear FLOPs ≈ 2 x active_params x L.

MI355X / MoRI environment reminders (no cluster-specific values):

```bash
export MORI_GPU_ARCHS=gfx950            # MI355X; avoids building for gfx942
export MORI_SHMEM_HEAP_SIZE=16G         # needed for large dispatch budgets
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO  # use tuned launch configs
export MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export MORI_SOCKET_IFNAME=$BACKEND_IFACE GLOO_SOCKET_IFNAME=$BACKEND_IFACE
# MORI_RDMA_SL / MORI_RDMA_TC must match the fabric's PFC no-drop priority and DSCP (TC = DSCP * 4)
```
