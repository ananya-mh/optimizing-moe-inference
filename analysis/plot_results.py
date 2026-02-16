"""Visualization tools for MoE benchmark results.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_all_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load all JSON result files into a DataFrame."""
    records = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not item.get("success"):
                    continue
                metrics = item.get("metrics", {})
                records.append({
                    "model": item.get("model_id", "").split("/")[-1],
                    "strategy": item.get("strategy", {}).get("name", "unknown"),
                    "workload": item.get("workload", {}).get("name", "unknown"),
                    "concurrency": item.get("concurrency", 0),
                    "throughput": metrics.get("throughput_tok_per_sec", 0),
                    "ttft_avg": metrics.get("ttft_avg_ms", 0),
                    "itl_avg": metrics.get("itl_avg_ms", 0),
                    "e2e_latency": metrics.get("e2e_latency_avg_ms", 0),
                })
    return pd.DataFrame(records)


def plot_throughput_vs_concurrency(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    """Plot throughput vs concurrency for each model and strategy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in model_df["strategy"].unique():
            strat_df = model_df[model_df["strategy"] == strategy]
            grouped = strat_df.groupby("concurrency")["throughput"].mean()
            ax.plot(grouped.index, grouped.values, marker="o", label=strategy)

        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title(f"Throughput vs Concurrency - {model}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

        path = output_dir / f"throughput_vs_concurrency_{model}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_latency_vs_concurrency(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    """Plot TTFT and ITL vs concurrency."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for strategy in model_df["strategy"].unique():
            strat_df = model_df[model_df["strategy"] == strategy]
            grouped_ttft = strat_df.groupby("concurrency")["ttft_avg"].mean()
            grouped_itl = strat_df.groupby("concurrency")["itl_avg"].mean()

            ax1.plot(grouped_ttft.index, grouped_ttft.values, marker="o", label=strategy)
            ax2.plot(grouped_itl.index, grouped_itl.values, marker="s", label=strategy)

        ax1.set_xlabel("Concurrency")
        ax1.set_ylabel("TTFT (ms)")
        ax1.set_title(f"Time to First Token - {model}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log", base=2)

        ax2.set_xlabel("Concurrency")
        ax2.set_ylabel("ITL (ms)")
        ax2.set_title(f"Inter-Token Latency - {model}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)

        path = output_dir / f"latency_vs_concurrency_{model}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    """Bar chart comparing strategies at a fixed concurrency."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use median concurrency
    median_conc = df["concurrency"].median()
    target_conc = df["concurrency"].unique()
    target_conc = target_conc[np.argmin(np.abs(target_conc - median_conc))]

    subset = df[df["concurrency"] == target_conc]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = subset.pivot_table(
        values="throughput", index="model", columns="strategy", aggfunc="mean"
    )
    pivot.plot(kind="bar", ax=ax, width=0.8)

    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title(f"Strategy Comparison (concurrency={int(target_conc)})")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    path = output_dir / f"strategy_comparison_conc{int(target_conc)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Generate all plots from available results."""
    print("=== Generating Plots ===")
    df = load_all_results()
    if df.empty:
        print("No results found. Run benchmarks first.")
        return

    print(f"Loaded {len(df)} result records")
    plot_throughput_vs_concurrency(df)
    plot_latency_vs_concurrency(df)
    plot_strategy_comparison(df)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
