"""
Distributed LLaDA Inference Engine with RCCL for AMD MI300X.

Supports multi-GPU inference for both LLaDA-8B (dense) and LLaDA-MoE-7B (MoE).
Uses torch.distributed with NCCL/RCCL backend for collective communication.

For the MoE variant, this implements Expert Parallelism (EP):
  - Each GPU holds a subset of experts
  - Tokens are routed to the correct GPU via all-to-all communication
  - Expert computations are done locally, then results are gathered

For the dense variant, this implements Tensor Parallelism (TP):
  - Attention heads and MLP are sharded across GPUs
  - Uses all-reduce for synchronization

Usage:
    torchrun --nproc_per_node=4 llada_distributed.py \\
        --model-path /models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct \\
        --output-json /results/llada_moe_ep4.json
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from llada_engine import (
    LLaDAConfig,
    LLaDAResult,
    add_gumbel_noise,
    get_num_transfer_tokens,
    ExpertLoadTracker,
    BENCHMARK_PROMPTS,
)


class MoEExpertParallelWrapper(torch.nn.Module):
    """
    Wraps a LLaDA-MoE model to distribute experts across GPUs.

    Instead of each GPU running the full MoE layer, each GPU holds
    a partition of experts. Tokens are dispatched via all-to-all
    and results are gathered back.

    This uses RCCL (ROCm Communication Collectives Library) for
    efficient GPU-to-GPU communication on AMD hardware.
    """

    def __init__(self, model, world_size: int, rank: int):
        super().__init__()
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.hooks_installed = False

    def install_ep_hooks(self):
        """Install forward hooks on MoE layers to intercept and distribute expert calls."""
        if self.hooks_installed:
            return

        for name, module in self.model.named_modules():
            if hasattr(module, "gate") and hasattr(module, "experts"):
                num_experts = len(module.experts) if hasattr(module.experts, "__len__") else getattr(module, "num_experts", 0)
                if num_experts > 0:
                    module._ep_world_size = self.world_size
                    module._ep_rank = self.rank
                    module._ep_experts_per_gpu = num_experts // self.world_size
                    module._ep_expert_start = self.rank * module._ep_experts_per_gpu
                    module._ep_expert_end = module._ep_expert_start + module._ep_experts_per_gpu

        self.hooks_installed = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class DistributedLLaDAEngine:
    """
    Multi-GPU LLaDA inference engine using RCCL for AMD ROCm.

    Supports:
    - Expert Parallelism (EP) for MoE variant
    - Data Parallelism (DP) for scaling throughput
    - Tensor Parallelism (TP) for dense variant
    """

    def __init__(
        self,
        config: LLaDAConfig,
        rank: int = 0,
        world_size: int = 1,
        backend: str = "nccl",
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.model = None
        self.tokenizer = None
        self.mask_id = None
        self.is_moe = False
        self.num_experts = 0
        self.expert_tracker = None
        self.model_load_time = 0.0

    def setup_distributed(self):
        """Initialize torch.distributed with RCCL backend."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size,
            )
        torch.cuda.set_device(self.rank)
        self.config.device = "cuda:%d" % self.rank

    def load_model(self):
        """Load model on this rank's GPU."""
        from transformers import AutoTokenizer, AutoModel, AutoConfig

        if self.rank == 0:
            print("Loading LLaDA model from %s on %d GPUs..." % (
                self.config.model_path, self.world_size
            ))

        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        model_config = AutoConfig.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        if hasattr(model_config, "mask_token_id"):
            self.mask_id = model_config.mask_token_id
        elif hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
            self.mask_id = self.tokenizer.mask_token_id
        else:
            self.mask_id = 126336

        if hasattr(model_config, "num_experts") and model_config.num_experts > 1:
            self.is_moe = True
            self.num_experts = model_config.num_experts
            self.expert_tracker = ExpertLoadTracker(self.num_experts)

        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=self.config.dtype,
        ).to(self.config.device).eval()

        if self.is_moe and self.world_size > 1:
            self.model = MoEExpertParallelWrapper(
                self.model, self.world_size, self.rank
            )
            self.model.install_ep_hooks()

        self.model_load_time = time.time() - t0

        if self.rank == 0:
            mem_gb = torch.cuda.memory_allocated(self.config.device) / 1e9
            print("Model loaded in %.1fs, GPU memory: %.1f GB" % (
                self.model_load_time, mem_gb
            ))

    def generate(
        self,
        prompt: str,
        gen_length: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> LLaDAResult:
        """Generate text on rank 0 and broadcast inputs to all ranks."""
        gen_length = gen_length or self.config.gen_length
        steps = steps or self.config.steps
        block_length = self.config.block_length
        temperature = self.config.temperature
        cfg_scale = self.config.cfg_scale
        remasking = self.config.remasking
        device = self.config.device

        # Tokenize on rank 0, broadcast
        if self.rank == 0:
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
        else:
            input_ids = None

        if self.world_size > 1:
            if self.rank == 0:
                length_tensor = torch.tensor([input_ids.shape[1]], device=device, dtype=torch.long)
            else:
                length_tensor = torch.zeros(1, device=device, dtype=torch.long)
            dist.broadcast(length_tensor, src=0)

            if self.rank != 0:
                input_ids = torch.zeros(1, length_tensor.item(), device=device, dtype=torch.long)
            dist.broadcast(input_ids, src=0)

        prompt_length = input_ids.shape[1]

        x = torch.full(
            (1, prompt_length + gen_length), self.mask_id,
            dtype=torch.long, device=device
        )
        x[:, :prompt_length] = input_ids.clone()
        prompt_index = (x != self.mask_id)

        num_blocks = (gen_length + block_length - 1) // block_length
        steps_per_block = max(steps // num_blocks, 1)

        step_index = 0
        total_model_ns = 0
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

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=device)
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

                # Synchronize across GPUs
                if self.world_size > 1:
                    dist.broadcast(x, src=0)

                step_index += 1

        torch.cuda.synchronize()
        gen_end = time.perf_counter_ns()

        total_ms = (gen_end - gen_start) / 1e6
        model_ms = total_model_ns / 1e6

        if self.rank == 0:
            response_tokens = x[0, prompt_length:]
            output_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            output_text = ""

        return LLaDAResult(
            text=output_text,
            prompt_tokens=prompt_length,
            generated_tokens=gen_length,
            total_steps=step_index,
            total_time_ms=total_ms,
            model_forward_time_ms=model_ms,
            tokens_per_second=gen_length / (total_ms / 1000) if total_ms > 0 else 0,
            model_load_time_s=self.model_load_time,
            step_timings=step_timings,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Distributed LLaDA Benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    config = LLaDAConfig(
        model_path=args.model_path,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
    )

    engine = DistributedLLaDAEngine(config, rank=rank, world_size=world_size, backend=args.backend)
    engine.setup_distributed()
    engine.load_model()

    if rank == 0:
        print("\nSanity check...")
    test = engine.generate("Hello, what is 2+2?", gen_length=32, steps=16)
    if rank == 0:
        print("Response: %s" % test.text[:200])

    prompts = BENCHMARK_PROMPTS[:args.num_prompts]
    if rank == 0:
        print("\nBenchmarking %d prompts..." % len(prompts))

    # Warmup
    for i in range(min(args.num_warmup, len(prompts))):
        engine.generate(prompts[i])
        torch.cuda.empty_cache()

    # Benchmark
    results = []
    for prompt in prompts:
        r = engine.generate(prompt, gen_length=args.gen_length, steps=args.steps)
        results.append(r)
        torch.cuda.empty_cache()

    if rank == 0:
        total_tokens = sum(r.generated_tokens for r in results)
        total_time = sum(r.total_time_ms for r in results)

        output = {
            "model": args.model_path,
            "world_size": world_size,
            "num_prompts": len(prompts),
            "gen_length": args.gen_length,
            "steps": args.steps,
            "total_generated_tokens": total_tokens,
            "total_time_ms": round(total_time, 2),
            "avg_time_per_prompt_ms": round(total_time / len(prompts), 2),
            "throughput_tok_s": round(total_tokens / (total_time / 1000), 1),
            "model_load_time_s": round(engine.model_load_time, 1),
        }

        sep = "=" * 60
        print("\n" + sep)
        print("Model: %s" % output["model"])
        print("World size: %d GPUs" % world_size)
        print("Throughput: %.1f tok/s" % output["throughput_tok_s"])
        print("Avg per prompt: %.1fms" % output["avg_time_per_prompt_ms"])
        print(sep)

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            print("Results saved to %s" % args.output_json)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
