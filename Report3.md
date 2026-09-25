# Report 3 — Paper Plan: Scaling Diffusion LLM Inference to Multi-Million-Token Contexts

A proposed story, contributions, evaluation plan, baselines, and schedule for an MLSys 2027 (primary)
or IPDPS (secondary) submission. Builds on [Report1.md](Report1.md) (MI300X measurements) and
[Report2.md](Report2.md) (related work, MI355X + AINIC, MoRI, long-context analysis).

Numbers marked *estimate* are analytical (from model configs and published MoRI/hardware figures),
not measurements. The paper must replace every one of them with measured data.

---

## 1. Verdict

**Yes, there is a credible, novel story**, provided we (a) measure real multi-node gains over strong
existing dLLM systems (SGLang dLLM, dInfer), (b) run a controlled AR-vs-diffusion comparison, and
(c) deliver at least two techniques beyond engineering integration.

What makes it novel (as of Sep 2026, from our survey):

- Existing dLLM serving work is single-GPU (dLLM-Serve), single-node batch-1 (dInfer), or cluster
  scheduling with TP only (DiLaServe). SGLang serves LLaDA2.x with TP/EP but has no dLLM-specific
  long-context or disaggregation design.
- Context-parallel serving work (vLLM DCP, NanoCP, Meta CP) targets autoregressive models.
- No published dLLM work goes beyond 256K context (DiffusionGemma) or 128K (LLaDA2.2, UltraLLaDA),
  and none studies multi-node prefill/decode disaggregation or GPU-initiated KV transfer for dLLMs.

**Venue fit.** MLSys is the better fit: the core idea is algorithm–system co-design (exploit dLLM
decoding properties in the distributed system). IPDPS fits the scaling/communication half. See
Section 8 for deadlines — IPDPS (full paper Oct 8, 2026) is not realistic for this scope.

---

## 2. The Story

**Working title:** *dScale: Exploiting Diffusion Decoding to Scale LLM Inference to Multi-Million-Token
Contexts on GPU Clusters*

**Thesis.** At million-token contexts, autoregressive decoding is bound by reading the KV cache: every
generated token streams the entire cache from HBM. Diffusion LLMs have three properties that change
the distributed-systems trade-offs:

1. **Multi-token commits per KV pass.** One denoising step reads the KV cache once and commits
   TPF tokens (≈2–5 with confidence-threshold decoding). KV bandwidth is amortized TPF-fold.
2. **Immutable prefix KV in block-diffusion models.** In LLaDA2.x / SDAR, a committed block's KV never
   changes, so it can be streamed, sharded, cached and reused exactly.
3. **Step-to-step stability.** Attention saliency and (hypothesis) expert routing of committed tokens
   change little across denoising steps, so communication and memory decisions can be reused.

We build a distributed dLLM inference system that turns these properties into multi-node efficiency
for both prefill and decode, using GPU-initiated communication (MoRI / IBGDA) on AMD MI355X clusters
with Pollara AI NICs, and scale LLaDA-family models from 4K to 2M+ token contexts.

**One-sentence pitch:** *Diffusion LLMs convert long-context decoding from memory-bound to
amortizable, and GPU-initiated communication makes the resulting multi-node system nearly free of
transfer overhead.*

---

## 3. Contributions

**C1 — Characterization and cost model (must-have).** First multi-node characterization of dLLM
inference from 4K to 2M+ tokens across small and large models. Per-step cost model:
`T_step = T_KV_read + T_expert_weights + T_EP_comm + T_CP_merge + T_sampling`, validated against
rocprofv3 traces, and a closed-form crossover point where dLLM beats AR as context grows.

**C2 — Denoise-step context parallelism (must-have).** Sequence-sharded KV for decode with a
partial-attention (output, log-sum-exp) merge sized for one block of queries (~32 tokens), plus
two-level ring-attention prefill (XGMI inside a node, AINIC across nodes). Bidirectional attention is
naturally load-balanced; block-causal attention reuses the standard zig-zag trick.

**C3 — Shard-aligned, commit-driven KV transfer for PD disaggregation (must-have, IBGDA as stretch).**
Prefill and decode use the same sequence sharding, so each prefill GPU sends its shard to exactly one
decode GPU over its own NIC (8 NICs in parallel, no reshuffle). Committed blocks are streamed layer by
layer while prefill runs. Version 1 uses MoRI-IO (host-posted RDMA with chunking); version 2 issues
the RDMA writes from the GPU via MoRI-SHMEM/IR (IBGDA), removing the CPU from the critical path.

**C4 — Step-aware expert parallelism (should-have).** Reuse expert routing for committed tokens across
denoising steps and dispatch only still-masked tokens through MoRI-EP; overlap dispatch/combine with
attention using the split send/recv APIs. Report traffic reduction and accuracy impact.

**C5 — TPF-aware long-context scheduling (should-have).** Choose block size / confidence threshold per
request based on context length (larger commits when KV reads dominate), on top of FDFO batching.

**Stretch — Saliency-guided KV tiering** (HBM → peer HBM → remote HBM / host via MoRI-UMBP) using the
previous step's attention to prefetch, for contexts beyond one node's HBM.

---

## 4. Models

| Role | Diffusion LLM | Matched AR baseline | Why |
|------|---------------|---------------------|-----|
| **Small** | LLaDA2.2-mini (16B / 1.4B active, 128K native) | **Ling-mini-2.0** | Identical shape: 20 layers, d=2048, 4 KV heads, 256 experts top-8. LLaDA2.x is continued-trained from Ling 2.0, so the only difference is AR vs diffusion decoding |
| **Large** | LLaDA2.2-flash (103B / 6.1B active, 128K native) | **Ling-flash-2.0** | Identical shape: 32 layers, d=4096, 4 KV heads, 256 experts top-8 |
| Dense long-context AR reference | — | Qwen2.5-7B-Instruct-1M, Qwen2.5-14B-Instruct-1M | Natively ~1M context; strongest available long-context AR baselines |
| Other dLLM families (generality) | SDAR-30B-A3B, DiffusionGemma-26B-A4B (256K, sliding window) | — | Show the techniques are not LLaDA-specific |
| Legacy (from Report1) | LLaDA-8B, LLaDA-MoE-7B | — | Fully bidirectional models; used for the frozen-prefix study |

KV cache at 2M tokens (BF16, *estimate*): LLaDA2.2-mini ~82 GB, LLaDA2.2-flash ~131 GB,
SDAR-30B ~197 GB, LLaDA-8B ~1,049 GB. The GQA models make 2M–4M contexts affordable on one to four
MI355X nodes.

Beyond 128K all LLaDA2.2 runs need RoPE scaling (NTK / diffusion-aware NTK per UltraLLaDA). We report
NIAH / RULER accuracy as measured and frame 2M+ results as systems results.

---

## 5. Things to Test Against (Baselines)

### 5.1 End-to-end systems

| Baseline | What it tells us |
|----------|------------------|
| **SGLang dLLM** (LLaDA2.x, `LowConfidence` / `JointThreshold`, FDFO, TP/EP, MoRI-EP) on MI355X | Strongest open dLLM serving stack on the same hardware — our main system baseline |
| **dInfer** (LLaDA-MoE, LLaDA2.x) | State-of-the-art batch-1 dLLM speed; run on ROCm if it ports, otherwise compare against published H800 numbers with caveats |
| **Fast-dLLM v1** reference (LLaDA-8B) | Training-free caching + parallel decoding baseline |
| **vLLM / SGLang AR** serving Ling-mini/flash-2.0 and Qwen2.5-1M, with DCP and MoRIIOConnector PD | The AR world at the same context length and hardware |
| HF reference implementations | Lower bound; reproduces Report1 |

### 5.2 Component baselines

| Component | Baselines |
|-----------|-----------|
| Expert dispatch/combine | RCCL all-to-all vs MoRI-EP (IntraNode, InterNodeV1, InterNodeV1LL, AsyncLL) vs ours (step-aware) |
| KV transfer for PD | RCCL send/recv vs MoRI-IO (no chunking, chunking, multi-QP) vs shard-aligned vs IBGDA streaming |
| Context parallelism | TP-only (head sharding) vs fixed DCP (vLLM-style) vs ours; ring prefill 1/2/4 nodes |
| Parallel layouts | TP, TP+EP, DP-attention+EP, +CP |
| Fabric | AINIC UEC mode vs RoCEv2 mode (all-to-all tail latency, KV transfer) |
| KV precision | BF16 vs FP8 (OCP e4m3) |

### 5.3 Optional cross-vendor point

If an H200/B200 node is available: SGLang dLLM with DeepEP/NVSHMEM on the same models. Otherwise,
compare against published numbers only, clearly labeled.

---

## 6. Evaluation Plan (Figures a Reviewer Expects)

| # | Experiment | Metric | Expected shape (hypothesis) |
|---|------------|--------|-----------------------------|
| F1 | Step-time breakdown vs context (4K → 2M), LLaDA2.2-mini/flash | ms per step by component | KV read and EP comm dominate beyond ~128K |
| F2 | Decode tok/s vs context: LLaDA2.2 vs Ling 2.0 (same shape) | tok/s per request | Crossover; dLLM advantage ≈ TPF at long context |
| F3 | Prefill strong scaling at 1M / 2M: 8 → 16 → 32 GPUs | TTFT, parallel efficiency | Near-linear (target ≥ 80%) |
| F4 | PD disaggregation: collocated vs MoRI-IO vs shard-aligned vs IBGDA | Goodput at SLO, KV transfer time, exposed transfer | Transfer hidden behind prefill |
| F5 | Step-aware EP | Bytes dispatched per step, latency, GSM8K / HumanEval accuracy | Traffic drops with committed fraction; accuracy unchanged |
| F6 | Ablation: add C2 → C3 → C4 → C5 one at a time | End-to-end tok/s and TTFT | Each adds measurable gain |
| F7 | Max context vs node count (BF16 / FP8) | Largest context served | 2M on 1 node (flash), 4M+ on 2–4 nodes |
| F8 | Generality: SDAR-30B, DiffusionGemma, LLaDA-8B (frozen prefix) | Speedup over SGLang / HF | Techniques transfer |
| T1 | Quality vs context | NIAH, RULER at 32K–2M | Honest: good ≤128K, degraded beyond without post-training |

Report tokens/s/GPU and joules per token where possible (MI355X is a 1,400 W part).

---

## 7. Implementation Strategy

**Build on SGLang for LLaDA2.x, keep the custom engine for LLaDA-8B / LLaDA-MoE.** SGLang already
runs LLaDA2.x on MI355X with KV cache, threshold decoding, TP/EP and MoRI integrations. Writing a new
engine for 100B MoE models in five weeks is not realistic; extending SGLang is. Our contributions
(DCP for dLLM steps, shard-aligned PD, step-aware EP, TPF scheduler) are implemented as SGLang
changes plus MoRI-level code.

The existing custom engine (`src/inference/llada_engine.py`) still needs Report2's roadmap items 1–4
(fused MoE, flash attention, KV cache, threshold decoding) for the legacy-model experiments.

**Hardware needed:** 4 nodes x 8 MI355X with 8x Pollara 400 each, lossless fabric configured (PFC,
DCQCN), for about three weeks of experiments.

---

## 8. Venues and Schedule

| Venue | Abstract | Full paper | Feasible? |
|-------|----------|-----------|-----------|
| [IPDPS 2027](https://www.ipdps.org/ipdps2027/2027-call-for-papers.html) | Oct 1, 2026 | Oct 8, 2026 (firm) | **No** for this scope (under 2 weeks, no system built yet) |
| [MLSys 2027](https://mlsys.org/Conferences/2027/Dates) | — | **Oct 30, 2026** | **Tight but possible** with the must-have scope (C1–C3, IBGDA as stretch) |
| Fallbacks | — | Spring 2027 systems / HPC venues | Full scope incl. IBGDA streaming and KV tiering |

**Five-week plan to MLSys (Sep 28 → Oct 30):**

| Week | Dates | Goals |
|------|-------|-------|
| 1 | Sep 28 – Oct 4 | SGLang LLaDA2.2-mini/flash and Ling 2.0 AR baselines running on MI355X; Qwen2.5-1M on vLLM; rocprofv3 step breakdown (F1); fabric + MoRI-EP/IO micro-benchmarks |
| 2 | Oct 5 – Oct 11 | Long context on one node: RoPE scaling, chunked prefill, FP8 KV, decode CP (C2); first 1M–2M runs (F2, F7) |
| 3 | Oct 12 – Oct 18 | Multi-node ring prefill (F3); shard-aligned PD via MoRI-IO (C3, F4); step-aware EP prototype (C4) |
| 4 | Oct 19 – Oct 25 | Full experiment sweep on 1–4 nodes, ablations (F5, F6, F8), quality runs (T1); IBGDA streaming only if C1–C3 are done |
| 5 | Oct 26 – Oct 30 | Writing, figures, artifact cleanup |

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| "dLLMs don't actually work at 2M" | Frame as systems paper; report NIAH/RULER honestly; show quality at ≤128K; cite UltraLLaDA as the path to quality |
| "Incremental over SGLang" | Contributions C2–C4 are dLLM-specific and absent from SGLang; ablation F6 isolates each |
| "AMD-only" | Techniques are vendor-neutral (MoRI supports CX-7 too); optional NVIDIA data point; controlled AR-vs-dLLM comparison on the same hardware is the main result |
| Routing-reuse hypothesis fails | Drop C4; C1–C3 still carry the paper |
| Cluster time / fabric issues | Validate RCCL + MoRI micro-benchmarks in week 1; keep single-node 2M results as a fallback story |
| 5-week timeline slips | Submit the must-have scope; move IBGDA streaming and tiering to the fallback venue |
