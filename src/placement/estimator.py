"""Expert placement estimation framework.

Given model and GPU configuration, estimates the optimal placement
strategy and queue depth for expert-aware batching.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any

from placement.strategies import (
    PlacementStrategy,
    STRATEGIES,
    estimate_per_gpu_memory,
    estimate_communication_volume,
)


@dataclass
class PlacementRecommendation:
    """Recommendation from the placement estimator."""
    strategy_name: str
    strategy: PlacementStrategy
    memory_per_gpu_gb: float
    fits_in_memory: bool
    communication_mb: float
    estimated_queue_depth: int
    reasoning: str



def estimate_queue_depth(
    num_experts: int,
    top_k: int,
    batch_size: int,
    num_gpus: int = 1,
) -> int:
    """Estimate optimal queue depth for expert-aware batching.

    Queue depth determines how many tokens are batched per expert
    before dispatching. Higher depth improves GPU occupancy but
    increases latency.

    Heuristic: queue_depth = ceil(batch_size * top_k / num_experts_per_gpu)
    """
    experts_per_gpu = max(num_experts // num_gpus, 1)
    tokens_per_expert = (batch_size * top_k) / experts_per_gpu

    # Clamp to reasonable range
    queue_depth = max(1, min(int(tokens_per_expert + 0.5), 256))
    return queue_depth


def classify_bottleneck(
    model_config: dict[str, Any],
    gpu_memory_gb: float,
    gpu_compute_units: int = 304,  # MI300X default
) -> str:
    """Classify whether a configuration is memory-bound or compute-bound.

    Simple heuristic based on parameter-to-compute ratio.
    """
    total_params = model_config["total_params_b"] * 1e9
    active_params = model_config["active_params_b"] * 1e9

    # Memory bandwidth bound indicator: large model, few active params
    sparsity_ratio = active_params / total_params

    # MI300X: ~5.3 TB/s HBM bandwidth, ~1300 TFLOPS BF16
    # If model barely fits, memory-bound; if lots of headroom, compute-bound
    model_size_gb = total_params * 2 / (1024**3)  # bf16

    if model_size_gb > gpu_memory_gb * 0.8:
        return "memory_bound"
    elif sparsity_ratio < 0.1:
        return "memory_bound"  # Very sparse = lots of cold expert weights
    else:
        return "compute_bound"


def recommend_placement(
    model_config: dict[str, Any],
    num_gpus: int,
    gpu_memory_gb: float = 192.0,
    target_batch_size: int = 32,
) -> PlacementRecommendation:
    """Recommend optimal expert placement strategy.

    Considers:
    - Model size vs available GPU memory
    - Number of experts vs GPUs
    - Communication overhead
    - Expert-aware batching queue depth
    """
    num_experts = model_config["num_experts"]
    top_k = model_config["top_k"]

    recommendations = []

    for name, strategy in STRATEGIES.items():
        if strategy.total_gpus() > num_gpus:
            continue

        # Adjust strategy to fit available GPUs
        adjusted = PlacementStrategy(
            name=strategy.name,
            description=strategy.description,
            tensor_parallel_size=min(strategy.tensor_parallel_size, num_gpus),
            data_parallel_size=max(1, num_gpus // min(strategy.tensor_parallel_size, num_gpus)),
            enable_expert_parallel=strategy.enable_expert_parallel,
            all2all_backend=strategy.all2all_backend,
        )

        mem = estimate_per_gpu_memory(model_config, adjusted)
        comm = estimate_communication_volume(model_config, adjusted)
        fits = mem["total_per_gpu_gb"] < gpu_memory_gb * 0.85

        queue_depth = estimate_queue_depth(
            num_experts, top_k, target_batch_size, num_gpus
        )

        score = 0
        reasoning = []

        if fits:
            score += 10
            reasoning.append("fits in memory")
        else:
            reasoning.append("DOES NOT fit in memory")

        # Prefer EP when many experts and multiple GPUs
        if adjusted.enable_expert_parallel and num_experts > num_gpus:
            score += 5
            reasoning.append("EP beneficial for high expert count")

        # Penalize communication overhead
        comm_mb = comm.get("all2all_volume_mb", 0)
        if comm_mb > 100:
            score -= 2
            reasoning.append(f"high comm overhead ({comm_mb:.0f} MB)")

        recommendations.append((score, PlacementRecommendation(
            strategy_name=name,
            strategy=adjusted,
            memory_per_gpu_gb=mem["total_per_gpu_gb"],
            fits_in_memory=fits,
            communication_mb=comm_mb,
            estimated_queue_depth=queue_depth,
            reasoning="; ".join(reasoning),
        )))

    # Sort by score descending
    recommendations.sort(key=lambda x: x[0], reverse=True)

    if recommendations:
        return recommendations[0][1]
    else:
        raise ValueError("No viable placement strategy found")
