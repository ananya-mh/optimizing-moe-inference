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

**C6 — Multi-node profiling methodology and tooling (must-have).** A cross-node, cross-layer
profiling pipeline for dLLM + MoE inference on AMD clusters: application step markers, GPU kernel and
RCCL traces, MoRI device-side (warp-level) traces, and NIC/fabric counters, merged into one
clock-aligned timeline per experiment. It produces the step-time breakdown for C1, exposes stragglers
and un-overlapped communication, and is released as an artifact (details in Section 6.1).

**C7 — Cluster-wide tiered KV with MoRI-UMBP (should-have).** MoRI-UMBP is a distributed KV pool over
HBM, host DRAM and SSD: each node's peer process owns its blocks, a master acts only as a routing
advisor (global block index built from heartbeat events, capacity-aware placement, eviction by
watermark), and blocks move by RDMA. SGLang already integrates it as a hierarchical-cache storage
backend (`--hicache-storage-backend mori`) and for PD transfer, but only for AR models. For dLLMs we
use it for (a) contexts beyond aggregate HBM, with saliency-guided prefetch (dLLM attention saliency is
stable across steps), (b) exact reuse of immutable committed prefixes across requests and nodes, and
(c) heterogeneous disaggregation where MI300X nodes act as decode and KV-capacity tier for MI355X
prefill (Section 7.2).

---

## 4. Models

| Role | Diffusion LLM | Matched AR baseline | Why |
|------|---------------|---------------------|-----|
| **Small** | LLaDA2.2-mini (16B / 1.4B active, 128K native) | **Ling-mini-2.0** | Identical shape: 20 layers, d=2048, 4 KV heads, 256 experts top-8. LLaDA2.x is continued-trained from Ling 2.0, so the only difference is AR vs diffusion decoding |
| **Large** | LLaDA2.2-flash (103B / 6.1B active, 128K native) | **Ling-flash-2.0** | Identical shape: 32 layers, d=4096, 4 KV heads, 256 experts top-8 |
| Dense long-context AR reference | — | Qwen2.5-7B-Instruct-1M, Qwen2.5-14B-Instruct-1M | Natively ~1M context; strongest available long-context AR baselines |
| Other dLLM families (generality) | SDAR-30B-A3B, DiffusionGemma-26B-A4B (256K, sliding window) | — | Show the techniques are not LLaDA-specific |
| Legacy (from Report1), optional | LLaDA-8B, LLaDA-MoE-7B | — | Not supported by SGLang (it only ships `llada2.py`). Kept as Report1 motivation; port to SGLang only if time allows (frozen-prefix study) |

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
| **SGLang AR** serving Ling-mini/flash-2.0 (`bailing_moe`) and Qwen2.5-1M (`qwen2`), with `--dcp-size`, `--moe-a2a-backend mori`, `--disaggregation-transfer-backend mori` | The AR world at the same context length, hardware and engine |
| dInfer, Fast-dLLM (published numbers only) | Cross-platform reference points, clearly labeled; not re-run |

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
| F3 | Prefill strong scaling at 1M / 2M / 4M: 8 → 16 → 32 → 64 GPUs, MI355X+AINIC and MI300X+CX-7 | TTFT, parallel efficiency | Near-linear (target ≥ 80%) |
| F4 | PD disaggregation: collocated vs MoRI-IO vs shard-aligned vs IBGDA | Goodput at SLO, KV transfer time, exposed transfer | Transfer hidden behind prefill |
| F5 | Step-aware EP | Bytes dispatched per step, latency, GSM8K / HumanEval accuracy | Traffic drops with committed fraction; accuracy unchanged |
| F6 | Ablation: add C2 → C3 → C4 → C5 one at a time | End-to-end tok/s and TTFT | Each adds measurable gain |
| F7 | Max context vs node count (BF16 / FP8) | Largest context served | 2M on 1 node (flash), 4M–8M on 2–8 nodes |
| F8 | Generality: SDAR-30B, DiffusionGemma, LLaDA-8B (frozen prefix) | Speedup over SGLang / HF | Techniques transfer |
| T1 | Quality vs context | NIAH, RULER at 32K–2M | Honest: good ≤128K, degraded beyond without post-training |

Report tokens/s/GPU and joules per token where possible (MI355X is a 1,400 W part).

### 6.1 Multi-node profiling (C6)

**What to capture, by layer.** Every source is written per rank (host name + rank in the file name)
and merged afterwards.

| Layer | Tool | What it gives us |
|-------|------|------------------|
| Application | ROCTx ranges emitted by our SGLang changes: request id, denoising step, block, layer phase (attention / MoE / dispatch / combine / CP merge / KV transfer); SGLang's built-in torch-profiler hooks | Semantic spans to attribute every kernel to a step and phase |
| GPU kernels, copies, collectives | `rocprofv3 --kernel-trace --memory-copy-trace --rccl-trace --marker-trace` (or `--sys-trace`); default `rocpd` database per rank, converted with `rocpd convert` / `rocpd2pftrace` to Perfetto | Kernel timelines, RCCL calls, SDMA copies, ROCTx markers |
| Hardware counters | `rocprofv3 --pmc` on selected kernels; `rocprof-compute` (Omniperf) roofline for attention and fused-MoE kernels | Achieved HBM bandwidth vs 5.3 / 8 TB/s, MFMA utilization, occupancy |
| MoRI-EP device side | **MORI-VIZ** (build MoRI with `ENABLE_PROFILER=ON`): warp-level spans in dispatch/combine kernels, exported to Perfetto | Where GPU-initiated RDMA time goes: packing, RDMA post, wait, XGMI copy |
| MoRI-IO host side | `MORI_ROCTX=1`: ranges around batch-write dispatch and RDMA posting, plus post-to-completion markers | Exposed vs hidden KV-transfer time in PD disaggregation |
| rocSHMEM (if used) | `rocprofv3 --rocshmem-trace` | GPU-initiated OpenSHMEM calls |
| NIC / fabric | Sample every ~100 ms: `rdma statistic`, `ethtool -S`; Pollara: `nicctl` port / QoS / DCQCN statistics; CX-7: congestion-notification (CNP) and out-of-sequence counters; switch telemetry where available | Per-NIC bytes/s vs line rate, PFC pause frames, CNPs, retransmits, packet-spray behavior |
| Node | `amd-smi` metrics sampling (power, clocks, HBM temperature) | Energy per token, throttling detection |

**Aligning clocks across nodes.** Run PTP (or chrony at minimum) on all nodes. At the start of each
run, every rank records a host timestamp and a GPU timestamp immediately after a global barrier; the
merge step uses these to compute per-rank offsets and produces one Perfetto trace covering all
64 GPUs, with one track group per node.

**Derived metrics (computed automatically from merged traces):**

- **Per-step critical path** across ranks and its split into KV read, expert weights, EP
  dispatch/combine, CP merge, sampling (feeds the C1 cost model).
- **Exposed communication:** communication time not overlapped by compute, per phase.
- **Stragglers:** slowest rank per step and why (EP load imbalance from MoRI-EP `recv_count`, longer
  KV shard, NIC congestion).
- **EP load imbalance per step:** tokens received per rank and expert; Gini coefficient over time.
  This also feeds the CPU placement predictor from the SIEDS abstract.
- **KV-transfer overlap:** fraction of PD transfer hidden behind prefill (MoRI-IO vs IBGDA).
- **Fabric health:** NIC utilization vs 400 Gb/s, PFC pause fraction, CNP rate, UEC vs RoCEv2.

**Overhead control.** Trace a fixed window (for example, 20 denoising steps after warmup), filter
kernels by name, keep counter collection to separate runs, and report measured profiler overhead
(target under 5% on throughput).

**Profiling figures:**

| # | Figure |
|---|--------|
| F9 | Critical-path breakdown per step at 1M / 2M across 8 → 64 GPUs |
| F10 | Rank x step heatmap of step time and EP tokens received (stragglers, imbalance) |
| F11 | NIC utilization and congestion events over time: AINIC UEC vs RoCEv2 vs CX-7 |
| F12 | Communication–compute overlap efficiency before and after C2–C4 |

**Tooling to build in this repo** (extends `scripts/run_profiling.sh` and `src/profiling/`):
a multi-node launcher that wraps each rank with `rocprofv3` and starts NIC/`amd-smi` samplers; a
clock-offset recorder; `analysis/merge_traces.py` to align and merge per-rank traces into one
Perfetto file; and `analysis/critical_path.py` for the derived metrics above.

---

## 7. Implementation Strategy

### 7.1 Engine choice

**Decision (Sep 25): SGLang for every model and baseline, dLLM and AR alike.** One engine removes
engine differences from every comparison. The table below records why.

| Engine | dLLM support (Sep 2026) | Multi-node / MoRI | Verdict |
|--------|-------------------------|-------------------|---------|
| **SGLang** | Native: LLaDA2.x (`llada2.py`), SDAR (`sdar.py`, `sdar_moe.py`), DiffusionGemma (`gemma4_diffusion.py`); `LowConfidence` / `JointThreshold` / `Gemma4Renoise`; FDFO batching; TP/EP | `--moe-a2a-backend mori`, `--disaggregation-transfer-backend mori`, `--hicache-storage-backend mori` (UMBP), `--dcp-size`, `--attn-cp-size` | **The engine for everything** |
| vLLM | DiffusionGemma native (Jun 2026, via ModelState / spec-decode path). LLaDA2 only through [`vllm-project/dllm-plugin`](https://github.com/vllm-project/dllm-plugin): MVP, `--max-num-seqs 1` required, TP only | MoRIIOConnector, DCP, DP+EP mature for AR | Not used (decision above) |
| dInfer | LLaDA, LLaDA-MoE, LLaDA2.x; strongest batch-1 dLLM speed | TP/EP on one node; ROCm undocumented | Baseline only |
| Fast-dLLM v2 reference | Plain PyTorch/HF loop, block + sub-block cache, single-GPU; vLLM support still a TODO | None | Algorithm reference, not an engine |
| Custom engine (this repo) | LLaDA-8B, LLaDA-MoE | RCCL only | Retired for the paper; Report1 only |

NVlabs' own follow-ups (Fast-dVLM, Fast-dDrive) moved to a customized SGLang fork for serving, which
supports the same choice.

**What SGLang's dLLM path disables today** (from `python/sglang/srt/arg_groups/dllm_hook.py`, main
branch, Sep 2026). These are exactly the features the paper needs, so enabling them is the concrete
engineering core of C2–C4 and C7:

| Feature | Status for dLLMs | Paper contribution |
|---------|------------------|--------------------|
| PD disaggregation (`--disaggregation-mode`) | Forced off ("not supported by diffusion LLM inference") | C3 shard-aligned KV streaming over MoRI-IO / IBGDA |
| Hierarchical cache (`--enable-hierarchical-cache`, hence UMBP/Mooncake/NIXL storage) | Forced off | C7 UMBP tiering for dLLMs |
| Pipeline parallelism | Forced off | Not needed (TP/EP/CP instead) |
| CUDA/HIP graphs on AMD | Forced off | Enable for fixed-shape denoise steps (Report2 roadmap item 5) |
| Attention backend on AMD | Forced to `triton` or `aiter` | Use AITER |
| Decode / attention context parallelism (`--dcp-size`, `--attn-cp-size`) | Not blocked, but untested with dLLMs | C2; add a MoRI (GPU-initiated) option to `--dcp-comm-backend`, which today offers `ag_rs`, `a2a` (NCCL/RCCL), and `fi_a2a` (NVIDIA MNNVL only) |

### 7.2 Hardware available

| Cluster | Nodes x GPUs | HBM total | NICs |
|---------|--------------|-----------|------|
| MI300X + ConnectX-7 | 8 x 8 = 64 GPUs | 12.3 TB | CX-7 (RoCEv2) |
| MI355X + Pollara 400 AINIC | 8 x 8 = 64 GPUs | 18.4 TB | Pollara (UEC-ready / RoCEv2) |

Both clusters are reported to share a backend network. This enables three things beyond Section 6:

1. **Scale to 64 GPUs per cluster.** EP64 for LLaDA2.x-flash (4 of 256 experts per GPU); ring-CP
   prefill across 8 nodes. *Estimate* for LLaDA2.2-flash prefill (block-causal, ~2·L²·d·layers FLOPs,
   ~1 PF/s effective per GPU, 64 GPUs): 2M ≈ 16 s, 4M ≈ 65 s, 8M ≈ 4.4 min. KV at 8M ≈ 525 GB BF16,
   far below cluster HBM, so **2M–8M token contexts are feasible at the systems level**.
2. **Cross-generation, cross-NIC study.** Identical software on MI300X/CX-7 and MI355X/AINIC separates
   GPU-generation effects from NIC effects and shows MoRI is not tied to one NIC vendor, which answers
   part of the "AMD-only" concern.
3. **Heterogeneous PD disaggregation (new angle, needs validation).** Prefill is compute-bound
   (O(L²) attention), so run it on MI355X (2.5 PF BF16); run decode on MI300X nodes and move KV with
   MoRI-IO across the shared fabric. Open issues: CX-7 ↔ Pollara RoCEv2 interoperability must be
   tested (`ib_write_bw`, `rccl-tests` across one host of each); FP8 KV must be converted between
   MI355X OCP e4m3 and MI300X FNUZ formats; MoRI QoS settings (`MORI_RDMA_SL` / `MORI_RDMA_TC`) must
   match on both fabrics.

### 7.3 Build strategy

**SGLang for all models.** SGLang already runs LLaDA2.x, SDAR and DiffusionGemma, Ling 2.0 and Qwen2.5
on MI355X with KV cache, threshold decoding, TP/EP and MoRI integrations. Writing a new engine for
100B MoE models in five weeks is not realistic; extending SGLang is. All contributions (C2–C7) are
SGLang changes plus MoRI-level code. The custom engine in `src/inference/` is retired for the paper;
its Report1 numbers remain as motivation.

The paper skeleton lives in [`paper/`](paper/) (MLSys style, builds with `latexmk -pdf main.tex`).

**Hardware:** 8 nodes x 8 MI355X with Pollara 400 AINICs, plus 8 nodes x 8 MI300X with CX-7 on a
shared backend (Section 7.2). Validate lossless QoS (PFC, DCQCN) and cross-cluster RDMA in week 1.

---

## 8. Venues and Schedule

| Venue | Abstract | Full paper | Feasible? |
|-------|----------|-----------|-----------|
| [IPDPS 2027](https://www.ipdps.org/ipdps2027/2027-call-for-papers.html) | Oct 1, 2026 | Oct 8, 2026 (firm) | **No** for this scope (under 2 weeks, no system built yet) |
| [MLSys 2027](https://mlsys.org/Conferences/2027/Dates) (research or industrial track; cannot switch after deadline) | — | **Oct 30, 2026** | **Tight but possible** with the must-have scope (C1–C3 + C6; C4, C5, C7 as should-have; IBGDA as stretch) |
| Fallbacks | — | Spring 2027 systems / HPC venues | Full scope incl. IBGDA streaming and KV tiering |

**Five-week plan to MLSys (Sep 28 → Oct 30):**

| Week | Dates | Goals |
|------|-------|-------|
| 1 | Sep 28 – Oct 4 | SGLang LLaDA2.2-mini/flash and Ling 2.0 AR baselines running on MI355X; Qwen2.5-1M on vLLM; multi-node profiling pipeline (rocprofv3 per rank, NIC/amd-smi samplers, clock alignment, trace merge) and first step breakdown (F1); fabric + MoRI-EP/IO micro-benchmarks |
| 2 | Oct 5 – Oct 11 | Long context on one node: RoPE scaling, chunked prefill, FP8 KV, decode CP (C2); first 1M–2M runs (F2, F7) |
| 3 | Oct 12 – Oct 18 | Multi-node ring prefill (F3); shard-aligned PD via MoRI-IO (C3, F4); step-aware EP prototype (C4) |
| 4 | Oct 19 – Oct 25 | Full experiment sweep on 1–8 nodes, ablations (F5, F6, F8), profiling figures (F9–F12), quality runs (T1); IBGDA streaming only if C1–C3 are done |
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
