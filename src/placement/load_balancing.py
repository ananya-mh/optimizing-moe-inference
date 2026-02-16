"""Expert Parallelism load balancing analysis and optimization.

Analyzes expert activation distributions to detect imbalances and
recommends rebalancing strategies (expert replication, migration).

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ExpertLoadStats:
    """Statistics for expert load distribution."""
    expert_id: int
    gpu_id: int
    token_count: int
    activation_fraction: float  # fraction of total tokens routed here
    compute_time_ms: float = 0.0


@dataclass
class LoadBalanceReport:
    """Full load balancing analysis report."""
    num_experts: int
    num_gpus: int
    total_tokens: int
    # Per-expert stats
    expert_stats: list[ExpertLoadStats] = field(default_factory=list)
    # Per-GPU aggregates
    gpu_load: dict[int, int] = field(default_factory=dict)  # gpu -> token count
    # Imbalance metrics
    load_imbalance_ratio: float = 0.0  # max/avg load
    coefficient_of_variation: float = 0.0  # std/mean
    gini_coefficient: float = 0.0
    max_load_gpu: int = 0
    min_load_gpu: int = 0
    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_gini(values: list[float]) -> float:
    """Compute Gini coefficient for load distribution.

    0 = perfect equality, 1 = maximum inequality.
    """
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((2 * i - n + 1) * v for i, v in enumerate(sorted_vals))
    return cumsum / (n * sum(sorted_vals))


def analyze_expert_load(
    routing_decisions: list[list[int]],
    num_experts: int,
    num_gpus: int,
    expert_to_gpu: Optional[dict[int, int]] = None,
) -> LoadBalanceReport:
    """Analyze expert load distribution from routing decisions.

    Args:
        routing_decisions: List of token routing lists.
            Each inner list contains the expert IDs that token was routed to.
            E.g., [[0, 3], [1, 5], [0, 2]] means 3 tokens routed to 2 experts each.
        num_experts: Total number of experts
        num_gpus: Number of GPUs
        expert_to_gpu: Optional mapping from expert_id -> gpu_id.
            Default: round-robin assignment.

    Returns:
        LoadBalanceReport with imbalance metrics and recommendations.
    """
    if expert_to_gpu is None:
        expert_to_gpu = {e: e % num_gpus for e in range(num_experts)}

    # Count tokens per expert
    expert_counts = Counter()
    total_tokens = len(routing_decisions)
    for token_experts in routing_decisions:
        for expert_id in token_experts:
            expert_counts[expert_id] += 1

    total_activations = sum(expert_counts.values())

    # Build expert stats
    expert_stats = []
    for eid in range(num_experts):
        count = expert_counts.get(eid, 0)
        expert_stats.append(ExpertLoadStats(
            expert_id=eid,
            gpu_id=expert_to_gpu.get(eid, 0),
            token_count=count,
            activation_fraction=count / max(total_activations, 1),
        ))

    # Aggregate per GPU
    gpu_load = defaultdict(int)
    for stat in expert_stats:
        gpu_load[stat.gpu_id] += stat.token_count
    gpu_load = dict(gpu_load)

    # Compute imbalance metrics
    loads = list(gpu_load.values())
    if loads:
        avg_load = np.mean(loads)
        max_load = max(loads)
        min_load = min(loads)
        std_load = np.std(loads)

        imbalance_ratio = max_load / avg_load if avg_load > 0 else 1.0
        cv = std_load / avg_load if avg_load > 0 else 0.0
        gini = compute_gini(loads)

        max_gpu = max(gpu_load, key=gpu_load.get)
        min_gpu = min(gpu_load, key=gpu_load.get)
    else:
        imbalance_ratio = 1.0
        cv = 0.0
        gini = 0.0
        max_gpu = 0
        min_gpu = 0

    # Generate recommendations
    recommendations = []
    if imbalance_ratio > 1.2:
        recommendations.append(
            f"HIGH IMBALANCE: Max GPU load is {imbalance_ratio:.2f}x the average. "
            f"GPU {max_gpu} is overloaded."
        )
    if imbalance_ratio > 1.5:
        recommendations.append(
            "CRITICAL: Consider expert replication - replicate hot experts "
            "from overloaded GPUs to underloaded ones."
        )
    if cv > 0.3:
        recommendations.append(
            f"High variability (CV={cv:.2f}). Consider dynamic expert migration "
            "or load-aware routing."
        )

    # Check for hot/cold experts
    if expert_stats:
        activations = [s.activation_fraction for s in expert_stats]
        mean_act = np.mean(activations)
        hot_experts = [s for s in expert_stats if s.activation_fraction > 2 * mean_act]
        cold_experts = [s for s in expert_stats if s.activation_fraction < 0.2 * mean_act]

        if hot_experts:
            hot_ids = [s.expert_id for s in hot_experts[:5]]
            recommendations.append(
                f"Hot experts detected: {hot_ids}. "
                "Replicate these across GPUs or increase their capacity."
            )
        if cold_experts and len(cold_experts) > num_experts * 0.3:
            recommendations.append(
                f"{len(cold_experts)}/{num_experts} experts are cold (<20% avg activation). "
                "Consider expert pruning or consolidation."
            )

    if imbalance_ratio <= 1.1:
        recommendations.append("Load is well balanced. No action needed.")

    # Suggest optimal expert-to-GPU mapping
    if imbalance_ratio > 1.2:
        sorted_experts = sorted(expert_stats, key=lambda s: s.token_count, reverse=True)
        optimal_mapping = {}
        gpu_loads_opt = [0] * num_gpus
        for s in sorted_experts:
            target_gpu = min(range(num_gpus), key=lambda g: gpu_loads_opt[g])
            optimal_mapping[s.expert_id] = target_gpu
            gpu_loads_opt[target_gpu] += s.token_count

        opt_loads = gpu_loads_opt
        opt_ratio = max(opt_loads) / np.mean(opt_loads) if np.mean(opt_loads) > 0 else 1.0
        if opt_ratio < imbalance_ratio * 0.9:
            recommendations.append(
                f"Greedy re-mapping reduces imbalance from {imbalance_ratio:.2f}x to "
                f"{opt_ratio:.2f}x. Suggested mapping available in report."
            )

    return LoadBalanceReport(
        num_experts=num_experts,
        num_gpus=num_gpus,
        total_tokens=total_tokens,
        expert_stats=expert_stats,
        gpu_load=gpu_load,
        load_imbalance_ratio=round(imbalance_ratio, 4),
        coefficient_of_variation=round(cv, 4),
        gini_coefficient=round(gini, 4),
        max_load_gpu=max_gpu,
        min_load_gpu=min_gpu,
        recommendations=recommendations,
    )


def simulate_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    distribution: str = "uniform",
    zipf_param: float = 1.1,
    seed: int = 42,
) -> list[list[int]]:
    """Simulate expert routing decisions for load analysis.

    Args:
        num_tokens: Number of tokens to simulate
        num_experts: Number of experts
        top_k: Experts per token
        distribution: "uniform", "zipf", or "skewed"
        zipf_param: Zipf distribution parameter (higher = more skewed)
        seed: Random seed

    Returns:
        List of routing decisions (each is a list of expert IDs).
    """
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        # Each expert equally likely
        routing = []
        for _ in range(num_tokens):
            experts = rng.choice(num_experts, size=top_k, replace=False).tolist()
            routing.append(experts)

    elif distribution == "zipf":
        # Zipfian: some experts much more popular than others
        ranks = np.arange(1, num_experts + 1)
        probs = 1.0 / np.power(ranks, zipf_param)
        probs /= probs.sum()
        routing = []
        for _ in range(num_tokens):
            experts = rng.choice(num_experts, size=top_k, replace=False, p=probs).tolist()
            routing.append(experts)

    elif distribution == "skewed":
        # 80% of tokens go to 20% of experts
        hot_count = max(1, num_experts // 5)
        hot_experts = list(range(hot_count))
        cold_experts = list(range(hot_count, num_experts))
        routing = []
        for _ in range(num_tokens):
            if rng.random() < 0.8 and len(hot_experts) >= top_k:
                experts = rng.choice(hot_experts, size=min(top_k, len(hot_experts)), replace=False).tolist()
            else:
                experts = rng.choice(num_experts, size=top_k, replace=False).tolist()
            routing.append(experts)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return routing


def run_load_balance_study(
    model_config: dict[str, Any],
    num_gpus: int,
    num_tokens: int = 10000,
    distributions: list[str] = None,
) -> dict[str, LoadBalanceReport]:
    """Run load balance analysis across different routing distributions.

    Args:
        model_config: Model config from models.yaml
        num_gpus: Number of GPUs
        num_tokens: Tokens to simulate
        distributions: List of distributions to test

    Returns:
        Dict mapping distribution name to LoadBalanceReport
    """
    if distributions is None:
        distributions = ["uniform", "zipf", "skewed"]

    num_experts = model_config["num_experts"]
    top_k = model_config["top_k"]

    reports = {}
    for dist in distributions:
        routing = simulate_routing(num_tokens, num_experts, top_k, distribution=dist)
        report = analyze_expert_load(routing, num_experts, num_gpus)
        reports[dist] = report
        print(f"  {dist}: imbalance={report.load_imbalance_ratio:.3f}x, "
              f"CV={report.coefficient_of_variation:.3f}, "
              f"Gini={report.gini_coefficient:.3f}")

    return reports


def print_load_balance_summary(report: LoadBalanceReport):
    """Print a formatted load balance summary."""
    print(f"\n{'='*60}")
    print(f"EP Load Balance Report")
    print(f"{'='*60}")
    print(f"  Experts: {report.num_experts}, GPUs: {report.num_gpus}")
    print(f"  Total tokens: {report.total_tokens}")
    print(f"  Imbalance ratio: {report.load_imbalance_ratio:.3f}x (ideal: 1.0)")
    print(f"  Coefficient of variation: {report.coefficient_of_variation:.3f}")
    print(f"  Gini coefficient: {report.gini_coefficient:.3f}")
    print(f"  Max loaded GPU: {report.max_load_gpu} ({report.gpu_load.get(report.max_load_gpu, 0)} tokens)")
    print(f"  Min loaded GPU: {report.min_load_gpu} ({report.gpu_load.get(report.min_load_gpu, 0)} tokens)")
    print(f"\n  GPU Load Distribution:")
    for gpu, load in sorted(report.gpu_load.items()):
        bar_len = int(load / max(report.gpu_load.values()) * 40) if report.gpu_load else 0
        print(f"    GPU {gpu:2d}: {'#' * bar_len} ({load})")
    print(f"\n  Recommendations:")
    for rec in report.recommendations:
        print(f"    - {rec}")
    print()


if __name__ == "__main__":
    # Demo: simulate and analyze load balance for Mixtral-8x7B on 8 GPUs
    import yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
    with open(config_path) as f:
        registry = yaml.safe_load(f)

    print("=== EP Load Balance Analysis Demo ===\n")
    for model_key in ["mixtral_8x7b", "olmoe_1b_7b", "qwen_moe_a2.7b"]:
        model = registry["models"][model_key]
        print(f"--- {model_key} ({model['num_experts']} experts, top-{model['top_k']}) ---")
        reports = run_load_balance_study(model, num_gpus=8)
        for dist, report in reports.items():
            print_load_balance_summary(report)
