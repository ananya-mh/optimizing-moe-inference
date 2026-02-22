"""Controlled factorial study for MoE inference optimization.

Implements the methodology from the paper abstract: a factorial study
to isolate the contribution of each optimization strategy to overall
inference performance.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import itertools
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FactorialConfig:
    """A single factorial experiment configuration."""
    model_key: str
    num_gpus: int
    strategy: str  # tp_only, ep_only, tp_ep_hybrid, dp_ep
    expert_batching: bool  # expert-aware batching on/off
    batch_queue_depth: int
    workload: str  # short_prompt, medium_prompt, long_prompt
    concurrency: int


def generate_factorial_design(
    models: list[str],
    gpu_counts: list[int],
    strategies: list[str],
    batching_options: list[bool],
    queue_depths: list[int],
    workloads: list[str],
    concurrency_levels: list[int],
) -> list[FactorialConfig]:
    """Generate full factorial experiment design.

    Returns all combinations of the input factors.
    """
    configs = []
    for combo in itertools.product(
        models, gpu_counts, strategies, batching_options,
        queue_depths, workloads, concurrency_levels
    ):
        configs.append(FactorialConfig(
            model_key=combo[0],
            num_gpus=combo[1],
            strategy=combo[2],
            expert_batching=combo[3],
            batch_queue_depth=combo[4],
            workload=combo[5],
            concurrency=combo[6],
        ))
    return configs


def get_default_factorial_design() -> list[FactorialConfig]:
    """Get the default factorial design from the paper methodology.

    Factors:
    - Model: 3 models (small, medium, large)
    - Strategy: 4 placement strategies
    - Batching: on/off
    - Queue depth: 3 levels
    - Workload: 3 types
    """
    return generate_factorial_design(
        models=["qwen_moe_a2.7b", "mixtral_8x7b", "qwen2_57b_a14b"],
        gpu_counts=[1, 4, 8],
        strategies=["tp_only", "ep_only", "tp_ep_hybrid", "dp_ep"],
        batching_options=[True, False],
        queue_depths=[4, 16, 64],
        workloads=["short_prompt", "medium_prompt", "long_prompt"],
        concurrency_levels=[8, 32],
    )


def filter_viable_configs(
    configs: list[FactorialConfig],
    model_registry: dict[str, Any],
) -> list[FactorialConfig]:
    """Filter out configurations that are not viable.

    Removes configs where:
    - Model requires more GPUs than available
    - EP strategies on single GPU
    - TP size > num_gpus
    """
    viable = []
    for cfg in configs:
        model = model_registry.get("models", {}).get(cfg.model_key, {})
        min_gpus = model.get("min_gpus", 1)

        # Skip if not enough GPUs
        if cfg.num_gpus < min_gpus:
            continue

        # Skip EP strategies on single GPU
        if cfg.num_gpus == 1 and cfg.strategy in ("ep_only", "tp_ep_hybrid", "dp_ep"):
            continue

        # Skip batching variations when batching is off
        if not cfg.expert_batching and cfg.batch_queue_depth != 4:
            continue

        viable.append(cfg)

    return viable


def save_factorial_design(configs: list[FactorialConfig], output_path: str):
    """Save factorial design to JSON for reproducibility."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_configs": len(configs),
        "configs": [asdict(c) for c in configs],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(configs)} factorial configs to {output_path}")


if __name__ == "__main__":
    import yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
    with open(config_path) as f:
        registry = yaml.safe_load(f)

    print("=== Factorial Experiment Design ===")
    all_configs = get_default_factorial_design()
    print(f"Full factorial: {len(all_configs)} configurations")

    viable = filter_viable_configs(all_configs, registry)
    print(f"Viable configurations: {len(viable)}")

    output = Path(__file__).resolve().parent.parent.parent / "results" / "factorial_design.json"
    save_factorial_design(viable, str(output))

    # Summary
    from collections import Counter
    model_counts = Counter(c.model_key for c in viable)
    strategy_counts = Counter(c.strategy for c in viable)
    print(f"\nBy model: {dict(model_counts)}")
    print(f"By strategy: {dict(strategy_counts)}")
