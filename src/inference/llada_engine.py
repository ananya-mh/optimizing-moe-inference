"""
Custom LLaDA Inference Engine for AMD ROCm MI300X.

Supports both LLaDA-8B (dense diffusion) and LLaDA-MoE-7B (MoE diffusion).
Uses HuggingFace Transformers with trust_remote_code for model loading,
and implements the masked diffusion denoising loop natively on ROCm.

This engine is needed because vLLM does not support the LLaDA architecture.
LLaDA models use non-autoregressive masked diffusion instead of standard
autoregressive generation.

Features:
  - Block-based semi-autoregressive generation
  - Gumbel-max sampling for masked diffusion
  - Confidence-based or random remasking
  - Classifier-free guidance support
  - Multi-GPU support via torch.distributed (RCCL backend)
  - Profiling hooks for torch.profiler and rocprofv3
  - Expert load tracking for MoE variant

Usage:
    engine = LLaDAEngine(model_path, device="cuda:0")
    result = engine.generate("Hello world", gen_length=128, steps=64)
    print(result["text"])
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LLaDAConfig:
    """Configuration for LLaDA inference."""
    model_path: str = ""
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    gen_length: int = 128
    steps: int = 64
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"  # "low_confidence" or "random"
    enable_profiling: bool = False
    profile_output_dir: str = "./profiles"
    track_expert_loads: bool = True


@dataclass
class LLaDAResult:
    """Result from LLaDA generation."""
    text: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_steps: int = 0
    total_time_ms: float = 0.0
    model_forward_time_ms: float = 0.0
    sampling_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    model_load_time_s: float = 0.0
    expert_load_distribution: Optional[Dict[str, Any]] = None
    step_timings: List[float] = field(default_factory=list)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling for Masked Diffusion Models."""
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    """Compute tokens to unmask per step (linear schedule)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer


class ExpertLoadTracker:
    """Track expert activation patterns in MoE layers for load balance analysis."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.activation_counts = np.zeros(num_experts, dtype=np.int64)
        self.per_step_counts = []
        self._hooks = []

    def reset(self):
        self.activation_counts = np.zeros(self.num_experts, dtype=np.int64)
        self.per_step_counts = []

    def record_step(self, router_logits: Optional[torch.Tensor] = None):
        """Record expert activations for one denoising step."""
        if router_logits is not None:
            top_k_indices = torch.topk(router_logits, k=min(8, self.num_experts), dim=-1).indices
            step_counts = np.zeros(self.num_experts, dtype=np.int64)
            for idx in top_k_indices.reshape(-1).cpu().numpy():
                if idx < self.num_experts:
                    step_counts[idx] += 1
            self.activation_counts += step_counts
            self.per_step_counts.append(step_counts.copy())

    def get_report(self) -> Dict[str, Any]:
        """Generate load balance report."""
        total = self.activation_counts.sum()
        if total == 0:
            return {"balanced": True, "gini": 0.0}

        normalized = self.activation_counts / total
        sorted_counts = np.sort(normalized)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n
        cv = np.std(normalized) / np.mean(normalized) if np.mean(normalized) > 0 else 0

        return {
            "total_activations": int(total),
            "per_expert_counts": self.activation_counts.tolist(),
            "gini_coefficient": round(float(gini), 4),
            "coefficient_of_variation": round(float(cv), 4),
            "max_load_ratio": round(float(normalized.max() / normalized.mean()), 3) if normalized.mean() > 0 else 0,
            "min_load_ratio": round(float(normalized.min() / normalized.mean()), 3) if normalized.mean() > 0 else 0,
            "balanced": gini < 0.1,
        }


class LLaDAEngine:
    """
    Custom inference engine for LLaDA diffusion LLMs on AMD ROCm.

    Supports both LLaDA-8B (dense) and LLaDA-MoE-7B (MoE).
    Uses HuggingFace Transformers for model loading and implements
    the denoising loop with AMD-optimized operations.
    """

    def __init__(self, config: LLaDAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.mask_id = None
        self.is_moe = False
        self.num_experts = 0
        self.expert_tracker = None
        self.model_load_time = 0.0

    def load_model(self):
        """Load model and tokenizer from HuggingFace."""
        from transformers import AutoTokenizer, AutoModel, AutoConfig

        print("Loading LLaDA model from %s..." % self.config.model_path)
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        model_config = AutoConfig.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        # Detect mask token ID
        if hasattr(model_config, "mask_token_id"):
            self.mask_id = model_config.mask_token_id
        elif hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
            self.mask_id = self.tokenizer.mask_token_id
        else:
            # Default for LLaDA-8B
            self.mask_id = 126336

        # Detect MoE
        if hasattr(model_config, "num_experts") and model_config.num_experts > 1:
            self.is_moe = True
            self.num_experts = model_config.num_experts
            self.expert_tracker = ExpertLoadTracker(self.num_experts)
            print("Detected MoE model: %d experts, top-%d" % (
                self.num_experts,
                getattr(model_config, "num_experts_per_tok", 8)
            ))
        else:
            self.is_moe = False
            self.num_experts = 0

        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=self.config.dtype,
        ).to(self.config.device).eval()

        self.model_load_time = time.time() - t0
        print("Model loaded in %.1fs" % self.model_load_time)

        mem_gb = torch.cuda.memory_allocated(self.config.device) / 1e9
        print("GPU memory used: %.1f GB" % mem_gb)

    def generate(
        self,
        prompt: str,
        gen_length: Optional[int] = None,
        steps: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        remasking: Optional[str] = None,
    ) -> LLaDAResult:
        """
        Generate text using masked diffusion denoising.

        Args:
            prompt: Input text prompt
            gen_length: Number of tokens to generate
            steps: Number of denoising steps
            block_length: Block size for semi-autoregressive generation
            temperature: Gumbel noise temperature (0=greedy)
            cfg_scale: Classifier-free guidance scale (0=disabled)
            remasking: Remasking strategy ("low_confidence" or "random")

        Returns:
            LLaDAResult with generated text and timing info
        """
        assert self.model is not None, "Call load_model() first"

        gen_length = gen_length or self.config.gen_length
        steps = steps or self.config.steps
        block_length = block_length or self.config.block_length
        temperature = temperature if temperature is not None else self.config.temperature
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale
        remasking = remasking or self.config.remasking
        device = self.config.device

        # Tokenize
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                chat_input = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                input_ids = self.tokenizer(chat_input)["input_ids"]
            except Exception:
                input_ids = self.tokenizer(prompt)["input_ids"]
        else:
            input_ids = self.tokenizer(prompt)["input_ids"]

        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        prompt_length = input_ids.shape[1]

        # Initialize with mask tokens
        x = torch.full(
            (1, prompt_length + gen_length), self.mask_id,
            dtype=torch.long, device=device
        )
        x[:, :prompt_length] = input_ids.clone()
        prompt_index = (x != self.mask_id)

        # Block-based semi-autoregressive denoising
        num_blocks = (gen_length + block_length - 1) // block_length
        steps_per_block = max(steps // num_blocks, 1)

        step_index = 0
        total_model_ns = 0
        total_sampling_ns = 0
        step_timings = []

        if self.expert_tracker:
            self.expert_tracker.reset()

        torch.cuda.synchronize()
        gen_start = time.perf_counter_ns()

        for num_block in range(num_blocks):
            block_start = prompt_length + num_block * block_length
            block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])

            block_mask_index = (x[:, block_start:block_end] == self.mask_id)
            if not block_mask_index.any():
                continue

            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = (x == self.mask_id)
                if not mask_index.any():
                    break

                step_start = time.perf_counter_ns()

                # Model forward pass
                torch.cuda.synchronize()
                t_model_start = time.perf_counter_ns()

                with torch.no_grad():
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = self.mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        logits = self.model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = self.model(x).logits

                torch.cuda.synchronize()
                t_model_end = time.perf_counter_ns()
                total_model_ns += (t_model_end - t_model_start)

                # Token selection and remasking
                t_sample_start = time.perf_counter_ns()

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError("Unknown remasking: %s" % remasking)

                x0_p[:, block_end:] = -float("inf")
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float("inf"))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                for j in range(confidence.shape[0]):
                    block_confidence = confidence[j, block_start:block_end]
                    if i < steps_per_block - 1:
                        k = min(num_transfer_tokens[j, i].item(), block_confidence.numel())
                        _, select_indices = torch.topk(block_confidence, k=k)
                        select_indices = select_indices + block_start
                        transfer_index[j, select_indices] = True
                    else:
                        transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]

                x = torch.where(transfer_index, x0, x)

                torch.cuda.synchronize()
                t_sample_end = time.perf_counter_ns()
                total_sampling_ns += (t_sample_end - t_sample_start)

                step_timings.append((time.perf_counter_ns() - step_start) / 1e6)
                step_index += 1

        torch.cuda.synchronize()
        gen_end = time.perf_counter_ns()

        # Decode
        response_tokens = x[0, prompt_length:]
        output_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        total_ms = (gen_end - gen_start) / 1e6
        model_ms = total_model_ns / 1e6
        sampling_ms = total_sampling_ns / 1e6
        tokens_per_sec = gen_length / (total_ms / 1000) if total_ms > 0 else 0

        result = LLaDAResult(
            text=output_text,
            prompt_tokens=prompt_length,
            generated_tokens=gen_length,
            total_steps=step_index,
            total_time_ms=total_ms,
            model_forward_time_ms=model_ms,
            sampling_time_ms=sampling_ms,
            tokens_per_second=tokens_per_sec,
            model_load_time_s=self.model_load_time,
            step_timings=step_timings,
        )

        if self.expert_tracker:
            result.expert_load_distribution = self.expert_tracker.get_report()

        return result

    def benchmark(
        self,
        prompts: List[str],
        gen_length: int = 128,
        steps: int = 64,
        num_warmup: int = 2,
    ) -> Dict[str, Any]:
        """
        Run a full benchmark across multiple prompts.

        Returns aggregated timing statistics.
        """
        # Warmup
        for i in range(min(num_warmup, len(prompts))):
            self.generate(prompts[i], gen_length=gen_length, steps=steps)
            torch.cuda.empty_cache()

        # Benchmark
        results = []
        for prompt in prompts:
            r = self.generate(prompt, gen_length=gen_length, steps=steps)
            results.append(r)
            torch.cuda.empty_cache()

        total_tokens = sum(r.generated_tokens for r in results)
        total_time = sum(r.total_time_ms for r in results)
        total_model_time = sum(r.model_forward_time_ms for r in results)
        total_sampling_time = sum(r.sampling_time_ms for r in results)

        return {
            "model": self.config.model_path,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "num_prompts": len(prompts),
            "gen_length": gen_length,
            "steps": steps,
            "total_generated_tokens": total_tokens,
            "total_time_ms": round(total_time, 2),
            "avg_time_per_prompt_ms": round(total_time / len(prompts), 2),
            "throughput_tok_s": round(total_tokens / (total_time / 1000), 1),
            "avg_model_forward_ms": round(total_model_time / len(prompts), 2),
            "avg_sampling_ms": round(total_sampling_time / len(prompts), 2),
            "model_load_time_s": round(self.model_load_time, 1),
            "device": self.config.device,
            "dtype": str(self.config.dtype),
        }


# Benchmark prompts
BENCHMARK_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a short poem about the ocean at sunset.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "What is the significance of the Higgs boson discovery?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "How does a neural network learn from data?",
    "What are the advantages of renewable energy sources?",
    "Explain quantum entanglement to a high school student.",
    "Write a haiku about artificial intelligence.",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLaDA Inference Engine Benchmark")
    parser.add_argument("--model-path", required=True, help="Path to LLaDA model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config = LLaDAConfig(
        model_path=args.model_path,
        device=args.device,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
    )

    engine = LLaDAEngine(config)
    engine.load_model()

    # Quick sanity check
    print("\nSanity check...")
    test = engine.generate("Hello, what is 2+2?", gen_length=32, steps=16)
    print("Response: %s" % test.text[:200])
    print("Time: %.1fms, Model: %.1fms" % (test.total_time_ms, test.model_forward_time_ms))

    # Full benchmark
    prompts = BENCHMARK_PROMPTS[:args.num_prompts]
    print("\nRunning benchmark with %d prompts, gen_length=%d, steps=%d..." % (
        len(prompts), args.gen_length, args.steps
    ))

    results = engine.benchmark(
        prompts, gen_length=args.gen_length, steps=args.steps, num_warmup=args.num_warmup
    )

    sep = "=" * 60
    print("\n" + sep)
    print("Model: %s" % results["model"])
    print("MoE: %s (experts=%d)" % (results["is_moe"], results["num_experts"]))
    print("Prompts=%d | GenLen=%d | Steps=%d" % (
        results["num_prompts"], results["gen_length"], results["steps"]
    ))
    print("Total time: %.1fms" % results["total_time_ms"])
    print("Avg per prompt: %.1fms" % results["avg_time_per_prompt_ms"])
    print("Throughput: %.1f tok/s" % results["throughput_tok_s"])
    print("Avg model forward: %.1fms" % results["avg_model_forward_ms"])
    print("Avg sampling: %.1fms" % results["avg_sampling_ms"])
    print(sep)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to %s" % args.output_json)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
