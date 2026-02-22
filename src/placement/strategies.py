"""Expert placement strategy definitions and memory estimation.

Defines placement strategies for MoE models across GPU configurations
and provides memory/communication cost estimation.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlacementStrategy:
    """A strategy for placing experts across GPUs."""
    name: str
    description: str
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    all2all_backend: str | None = None

    def total_gpus(self) -> int:
        """Total GPUs required."""
        if self.enable_expert_parallel:
            return self.tensor_parallel_size * self.data_parallel_size
        return self.tensor_parallel_size


# Pre-defined strategies
STRATEGIES = {
    "colocate_single": PlacementStrategy(
        name="colocate_single",
        description="All experts on a single GPU (no parallelism)",
        tensor_parallel_size=1,
    ),
    "tp_only": PlacementStrategy(
        name="tp_only",
        description="Tensor parallelism only; experts weight-sharded across TP group",
        tensor_parallel_size=8,
    ),
    "ep_only": PlacementStrategy(
        name="ep_only",
        description="Expert parallelism only; each GPU holds a subset of experts",
        tensor_parallel_size=1,
        data_parallel_size=8,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    ),
    "tp_ep_hybrid": PlacementStrategy(
        name="tp_ep_hybrid",
        description="TP for attention + EP for MoE layers",
        tensor_parallel_size=2,
        data_parallel_size=4,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    ),
    "dp_ep": PlacementStrategy(
        name="dp_ep",
        description="Data parallelism + Expert parallelism for throughput",
        tensor_parallel_size=1,
        data_parallel_size=8,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    ),
}


def estimate_expert_memory_gb(
    total_params_b: float,
    num_experts: int,
    active_params_b: float,
    dtype_bytes: int = 2,  # bf16 = 2 bytes
) -> dict[str, float]:
    """Estimate memory breakdown for a MoE model.

    Returns dict with:
      - expert_params_gb: Total expert parameter memory
      - non_expert_params_gb: Attention/embedding parameter memory
      - single_expert_gb: Memory per expert
    """
    total_bytes = total_params_b * 1e9 * dtype_bytes
    total_gb = total_bytes / (1024**3)

    # Approximate: expert params = total - non-expert
    # Non-expert params ~ active params (attention + embeddings)
    non_expert_gb = (active_params_b * 1e9 * dtype_bytes) / (1024**3)
    expert_gb = total_gb - non_expert_gb
    single_expert_gb = expert_gb / max(num_experts, 1)

    return {
        "total_model_gb": round(total_gb, 2),
        "expert_params_gb": round(expert_gb, 2),
        "non_expert_params_gb": round(non_expert_gb, 2),
        "single_expert_gb": round(single_expert_gb, 3),
    }


def estimate_per_gpu_memory(
    model_config: dict[str, Any],
    strategy: PlacementStrategy,
    kv_cache_gb: float = 0,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate per-GPU memory usage for a placement strategy.

    Args:
        model_config: Model config dict from models.yaml
        strategy: Placement strategy
        kv_cache_gb: Estimated KV cache memory
        dtype_bytes: Bytes per parameter (2 for bf16)

    Returns:
        Dict with per-GPU memory breakdown.
    """
    mem = estimate_expert_memory_gb(
        total_params_b=model_config["total_params_b"],
        num_experts=model_config["num_experts"],
        active_params_b=model_config["active_params_b"],
        dtype_bytes=dtype_bytes,
    )

    n_gpus = strategy.total_gpus()
    n_experts = model_config["num_experts"]

    if strategy.enable_expert_parallel:
        # EP: experts distributed, non-expert replicated or TP-sharded
        experts_per_gpu = n_experts / n_gpus
        expert_mem_per_gpu = experts_per_gpu * mem["single_expert_gb"]
        non_expert_per_gpu = mem["non_expert_params_gb"] / strategy.tensor_parallel_size
    else:
        # TP only: all weights sharded
        expert_mem_per_gpu = mem["expert_params_gb"] / strategy.tensor_parallel_size
        non_expert_per_gpu = mem["non_expert_params_gb"] / strategy.tensor_parallel_size

    total_per_gpu = expert_mem_per_gpu + non_expert_per_gpu + kv_cache_gb

    return {
        "expert_mem_per_gpu_gb": round(expert_mem_per_gpu, 2),
        "non_expert_per_gpu_gb": round(non_expert_per_gpu, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "total_per_gpu_gb": round(total_per_gpu, 2),
        "experts_per_gpu": round(n_experts / n_gpus if strategy.enable_expert_parallel else n_experts, 1),
    }


def estimate_communication_volume(
    model_config: dict[str, Any],
    strategy: PlacementStrategy,
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_dim: int = 4096,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate all-to-all communication volume for expert parallelism.

    This is relevant only when enable_expert_parallel is True.
    """
    if not strategy.enable_expert_parallel:
        return {"all2all_volume_mb": 0, "description": "No EP communication"}

    n_gpus = strategy.total_gpus()
    top_k = model_config["top_k"]

    # Each token sends its hidden state to top_k experts
    # Volume per token = hidden_dim * dtype_bytes * top_k
    # For cross-GPU: fraction of tokens going to other GPUs
    tokens_per_batch = batch_size * seq_len
    per_token_bytes = hidden_dim * dtype_bytes * top_k

    # Approximate: each GPU sends (n_gpus-1)/n_gpus fraction of its tokens
    cross_gpu_fraction = (n_gpus - 1) / n_gpus
    total_bytes = tokens_per_batch * per_token_bytes * cross_gpu_fraction

    return {
        "all2all_volume_mb": round(total_bytes / (1024**2), 2),
        "per_token_bytes": per_token_bytes,
        "cross_gpu_fraction": round(cross_gpu_fraction, 3),
    }
