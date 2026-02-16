"""Visualization for EP load balancing analysis.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from placement.load_balancing import (
    simulate_routing,
    analyze_expert_load,
    run_load_balance_study,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def plot_expert_activation_heatmap(
    routing_decisions: list[list[int]],
    num_experts: int,
    num_gpus: int,
    title: str = "Expert Activation Heatmap",
    output_path: str = None,
):
    """Plot heatmap of expert activations across GPUs."""
    from collections import Counter

    expert_counts = Counter()
    for token_experts in routing_decisions:
        for eid in token_experts:
            expert_counts[eid] += 1

    experts_per_gpu = num_experts // num_gpus
    grid = np.zeros((num_gpus, experts_per_gpu + 1))

    for eid in range(num_experts):
        gpu = eid % num_gpus
        slot = eid // num_gpus
        if slot < grid.shape[1]:
            grid[gpu][slot] = expert_counts.get(eid, 0)

    fig, ax = plt.subplots(figsize=(max(8, experts_per_gpu * 0.5), num_gpus * 0.6 + 2))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Expert slot within GPU")
    ax.set_ylabel("GPU ID")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Token count")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_gpu_load_comparison(
    reports: dict[str, "LoadBalanceReport"],
    model_name: str = "",
    output_path: str = None,
):
    """Bar chart comparing GPU load across distributions."""
    fig, axes = plt.subplots(1, len(reports), figsize=(5 * len(reports), 5), sharey=True)
    if len(reports) == 1:
        axes = [axes]

    for ax, (dist_name, report) in zip(axes, reports.items()):
        gpus = sorted(report.gpu_load.keys())
        loads = [report.gpu_load.get(g, 0) for g in gpus]
        avg = np.mean(loads)

        colors = ["#e74c3c" if l > avg * 1.2 else "#2ecc71" if l < avg * 0.8 else "#3498db"
                  for l in loads]
        ax.bar(gpus, loads, color=colors)
        ax.axhline(y=avg, color="black", linestyle="--", alpha=0.5, label=f"avg={avg:.0f}")
        ax.set_xlabel("GPU ID")
        ax.set_title(f"{dist_name}\nImbalance: {report.load_imbalance_ratio:.2f}x")
        ax.legend()

    axes[0].set_ylabel("Token count")
    fig.suptitle(f"GPU Load Distribution - {model_name}", fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    """Generate load balance analysis plots for all models."""
    import yaml

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
    with open(config_path) as f:
        registry = yaml.safe_load(f)

    print("=== Generating Load Balance Plots ===\n")

    for model_key in ["mixtral_8x7b", "olmoe_1b_7b", "qwen_moe_a2.7b", "llada_moe_7b"]:
        model = registry["models"].get(model_key)
        if not model:
            continue

        print(f"--- {model_key} ---")
        num_gpus = max(model.get("min_gpus", 1), 4)
        reports = run_load_balance_study(model, num_gpus=num_gpus)

        plot_gpu_load_comparison(
            reports,
            model_name=model_key,
            output_path=str(FIGURES_DIR / f"load_balance_{model_key}.png"),
        )

        # Heatmap for zipf distribution (most realistic)
        routing = simulate_routing(
            10000, model["num_experts"], model["top_k"], distribution="zipf"
        )
        plot_expert_activation_heatmap(
            routing,
            model["num_experts"],
            num_gpus,
            title=f"Expert Activation - {model_key} (Zipf routing)",
            output_path=str(FIGURES_DIR / f"expert_heatmap_{model_key}.png"),
        )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
