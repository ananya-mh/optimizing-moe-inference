"""Generate all paper-ready figures and statistical analyses for SIEDS 2026.

Reads master_results_clean.csv and produces:
  1. Strategy comparison bars (TP vs EP vs Hybrid) per model
  2. Scaling efficiency curves
  3. Concurrency saturation analysis
  4. Diffusion vs autoregressive MoE comparison
  5. Throughput-per-GPU efficiency
  6. Latency breakdown (TTFT vs ITL) by strategy
  7. Memory-throughput Pareto frontier
  8. ANOVA factorial analysis
  9. Summary statistics table (LaTeX)

Usage:
    python analysis/paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

PROJECT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT / "results" / "tables" / "master_results_clean.csv"
OUT_DIR = PROJECT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_LABELS = {
    "baseline_no_ep": "Single-GPU",
    "tp_only": "TP-only (2 GPU)",
    "ep_only": "EP-only (2 GPU)",
    "tp_ep_hybrid": "TP+EP Hybrid (4 GPU)",
}
STRATEGY_ORDER = ["baseline_no_ep", "tp_only", "ep_only", "tp_ep_hybrid"]
STRATEGY_COLORS = {
    "baseline_no_ep": "#636363",
    "tp_only": "#3182bd",
    "ep_only": "#e6550d",
    "tp_ep_hybrid": "#31a354",
}

MODEL_SHORT = {
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "OLMoE-1B-7B-0924": "OLMoE-1B-7B",
    "Qwen1.5-MoE-A2.7B": "Qwen-MoE-A2.7B",
    "LLaDA-8B-Instruct": "LLaDA-8B (dense)",
    "LLaDA-MoE-7B-A1B-Instruct": "LLaDA-MoE-7B",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["model_short"] = df["model"].map(MODEL_SHORT).fillna(df["model"])
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    return df


def get_multi_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to multi-GPU vLLM results (exclude LLaDA single-GPU baselines)."""
    return df[df["run"] == "multi"].copy()


def get_autoregressive(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to autoregressive models with comparable workloads."""
    ar_models = ["Mixtral-8x7B-Instruct-v0.1", "OLMoE-1B-7B-0924", "Qwen1.5-MoE-A2.7B"]
    return df[df["model"].isin(ar_models)].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Strategy Comparison Bars at fixed concurrency
# ──────────────────────────────────────────────────────────────────────────────
def fig1_strategy_comparison(df: pd.DataFrame):
    """Bar chart: throughput by strategy for each model at concurrency=8, balanced workload."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)

    # Use balanced workload, concurrency=8
    sub = ar[(ar["workload"] == "balanced") & (ar["conc"] == 8)]
    if sub.empty:
        # Fall back to any workload at conc 8
        sub = ar[ar["conc"] == 8]

    pivot = sub.groupby(["model_short", "strategy"])["tok/s"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(pivot["model_short"].unique())
    strategies = [s for s in STRATEGY_ORDER if s in pivot["strategy"].unique()]
    x = np.arange(len(models))
    width = 0.8 / len(strategies)

    for i, strat in enumerate(strategies):
        vals = []
        for m in models:
            row = pivot[(pivot["model_short"] == m) & (pivot["strategy"] == strat)]
            vals.append(row["tok/s"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(
            x + i * width - (len(strategies) - 1) * width / 2,
            vals, width,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat, f"C{i}"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Placement Strategy Comparison (concurrency = 8, balanced workload)")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.savefig(OUT_DIR / "fig1_strategy_comparison.png")
    fig.savefig(OUT_DIR / "fig1_strategy_comparison.pdf")
    plt.close(fig)
    print("  [1] Strategy comparison bars")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Throughput vs Concurrency (scaling curves per strategy)
# ──────────────────────────────────────────────────────────────────────────────
def fig2_throughput_scaling(df: pd.DataFrame):
    """One subplot per model: throughput vs concurrency for each strategy."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)

    # Use balanced workload for consistency
    sub = ar[ar["workload"] == "balanced"]
    if sub.empty:
        sub = ar

    models = sorted(sub["model_short"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = sub[sub["model_short"] == model]
        for strat in STRATEGY_ORDER:
            sdf = mdf[mdf["strategy"] == strat]
            if sdf.empty:
                continue
            grouped = sdf.groupby("conc")["tok/s"].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", markersize=4,
                    label=STRATEGY_LABELS.get(strat, strat),
                    color=STRATEGY_COLORS.get(strat))

        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_title(model)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Throughput Scaling with Concurrency", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_throughput_scaling.png")
    fig.savefig(OUT_DIR / "fig2_throughput_scaling.pdf")
    plt.close(fig)
    print("  [2] Throughput scaling curves")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Latency Breakdown (TTFT + ITL) by strategy
# ──────────────────────────────────────────────────────────────────────────────
def fig3_latency_breakdown(df: pd.DataFrame):
    """Grouped bar: TTFT and ITL at concurrency=8 for each model×strategy."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[(ar["workload"] == "balanced") & (ar["conc"] == 8)]
    if sub.empty:
        sub = ar[ar["conc"] == 8]

    agg = sub.groupby(["model_short", "strategy"]).agg(
        ttft=("ttft_mean", "mean"), itl=("itl_mean", "mean")
    ).reset_index()

    models = sorted(agg["model_short"].unique())
    strategies = [s for s in STRATEGY_ORDER if s in agg["strategy"].unique()]
    x = np.arange(len(models))
    width = 0.35 / len(strategies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, strat in enumerate(strategies):
        ttfts, itls = [], []
        for m in models:
            row = agg[(agg["model_short"] == m) & (agg["strategy"] == strat)]
            ttfts.append(row["ttft"].values[0] if len(row) > 0 else 0)
            itls.append(row["itl"].values[0] if len(row) > 0 else 0)
        offset = i * width - (len(strategies) - 1) * width / 2
        ax1.bar(x + offset, ttfts, width, label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat))
        ax2.bar(x + offset, itls, width, label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat))

    for ax, title, ylabel in [(ax1, "Time to First Token", "TTFT (ms)"),
                               (ax2, "Inter-Token Latency", "ITL (ms)")]:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Latency Breakdown by Placement Strategy (conc=8, balanced)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_latency_breakdown.png")
    fig.savefig(OUT_DIR / "fig3_latency_breakdown.pdf")
    plt.close(fig)
    print("  [3] Latency breakdown")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Throughput per GPU (efficiency)
# ──────────────────────────────────────────────────────────────────────────────
def fig4_throughput_per_gpu(df: pd.DataFrame):
    """Throughput / #GPUs at concurrency=32 — shows cost efficiency."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[(ar["workload"] == "balanced") & (ar["conc"] == 32)]
    if sub.empty:
        sub = ar[ar["conc"] == 32]

    agg = sub.groupby(["model_short", "strategy", "gpus"])["tok/s"].mean().reset_index()
    agg["tok_per_gpu"] = agg["tok/s"] / agg["gpus"]

    models = sorted(agg["model_short"].unique())
    strategies = [s for s in STRATEGY_ORDER if s in agg["strategy"].unique()]
    x = np.arange(len(models))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, strat in enumerate(strategies):
        vals = []
        gpu_labels = []
        for m in models:
            row = agg[(agg["model_short"] == m) & (agg["strategy"] == strat)]
            vals.append(row["tok_per_gpu"].values[0] if len(row) > 0 else 0)
            gpu_labels.append(f"{int(row['gpus'].values[0])}G" if len(row) > 0 else "")
        offset = i * width - (len(strategies) - 1) * width / 2
        bars = ax.bar(x + offset, vals, width,
                      label=STRATEGY_LABELS.get(strat, strat),
                      color=STRATEGY_COLORS.get(strat),
                      edgecolor="white", linewidth=0.5)
        for bar, v, gl in zip(bars, vals, gpu_labels):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                        f"{v:.0f}\n({gl})", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Throughput per GPU (tok/s/GPU)")
    ax.set_title("Cost Efficiency: Throughput per GPU (conc=32, balanced)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.savefig(OUT_DIR / "fig4_throughput_per_gpu.png")
    fig.savefig(OUT_DIR / "fig4_throughput_per_gpu.pdf")
    plt.close(fig)
    print("  [4] Throughput per GPU")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Concurrency Saturation Point Analysis
# ──────────────────────────────────────────────────────────────────────────────
def fig5_saturation_analysis(df: pd.DataFrame):
    """Heatmap: throughput normalized to max, showing saturation per strategy×model."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[ar["workload"] == "balanced"]
    if sub.empty:
        sub = ar

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = sorted(sub["model_short"].unique())

    for ax, model in zip(axes, models):
        mdf = sub[sub["model_short"] == model]
        strategies = [s for s in STRATEGY_ORDER if s in mdf["strategy"].unique()]
        concs = sorted(mdf["conc"].unique())

        grid = np.zeros((len(strategies), len(concs)))
        for i, strat in enumerate(strategies):
            sdf = mdf[mdf["strategy"] == strat].groupby("conc")["tok/s"].mean()
            max_tp = sdf.max() if len(sdf) > 0 else 1
            for j, c in enumerate(concs):
                grid[i, j] = sdf.get(c, 0) / max_tp * 100 if max_tp > 0 else 0

        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_xticks(range(len(concs)))
        ax.set_xticklabels(concs, fontsize=8)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], fontsize=8)
        ax.set_xlabel("Concurrency")
        ax.set_title(model)

        # Annotate cells
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = "white" if grid[i, j] > 70 else "black"
                ax.text(j, i, f"{grid[i,j]:.0f}%", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="% of peak throughput")
    fig.suptitle("Concurrency Saturation Analysis (balanced workload)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_saturation_heatmap.png")
    fig.savefig(OUT_DIR / "fig5_saturation_heatmap.pdf")
    plt.close(fig)
    print("  [5] Saturation heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Diffusion vs Autoregressive MoE
# ──────────────────────────────────────────────────────────────────────────────
def fig6_diffusion_vs_autoregressive(df: pd.DataFrame):
    """Bar chart comparing LLaDA-8B (dense) vs LLaDA-MoE-7B vs autoregressive baselines."""
    single = df[df["run"] == "single"].copy()

    # LLaDA results: use gen_128 workload for comparable token count
    llada_dense = single[(single["model"] == "LLaDA-8B-Instruct") & (single["workload"] == "gen_128")]
    llada_moe = single[(single["model"] == "LLaDA-MoE-7B-A1B-Instruct") & (single["workload"] == "gen_128")]

    # Autoregressive: use short_prompt at conc=1 for single-GPU baseline
    ar_single = single[
        (single["model"].isin(["Mixtral-8x7B-Instruct-v0.1", "OLMoE-1B-7B-0924", "Qwen1.5-MoE-A2.7B"])) &
        (single["conc"] == 1)
    ]
    # Pick one workload per model
    ar_baselines = ar_single.groupby("model").first().reset_index()

    models = []
    throughputs = []
    colors = []
    expert_counts = []

    if len(llada_dense) > 0:
        models.append("LLaDA-8B\n(dense diffusion)")
        throughputs.append(llada_dense["tok/s"].values[0])
        colors.append("#9467bd")
        expert_counts.append("1 (dense)")

    if len(llada_moe) > 0:
        models.append("LLaDA-MoE-7B\n(MoE diffusion)")
        throughputs.append(llada_moe["tok/s"].values[0])
        colors.append("#d62728")
        expert_counts.append("64 experts")

    for _, row in ar_baselines.iterrows():
        m = MODEL_SHORT.get(row["model"], row["model"])
        models.append(f"{m}\n(autoregressive)")
        throughputs.append(row["tok/s"])
        colors.append("#2ca02c")
        expert_counts.append({
            "Mixtral-8x7B-Instruct-v0.1": "8 experts",
            "OLMoE-1B-7B-0924": "64 experts",
            "Qwen1.5-MoE-A2.7B": "60 experts",
        }.get(row["model"], "?"))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, throughputs, color=colors, edgecolor="white", linewidth=0.5)
    for bar, tp, ec in zip(bars, throughputs, expert_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{tp:.1f} tok/s\n({ec})", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Single-GPU Baseline: Diffusion vs Autoregressive MoE Models")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=max(throughputs) * 1.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_diffusion_vs_autoregressive.png")
    fig.savefig(OUT_DIR / "fig6_diffusion_vs_autoregressive.pdf")
    plt.close(fig)
    print("  [6] Diffusion vs autoregressive")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7: Workload Sensitivity (decode-heavy vs balanced vs prefill-heavy)
# ──────────────────────────────────────────────────────────────────────────────
def fig7_workload_sensitivity(df: pd.DataFrame):
    """Show how workload type affects throughput per strategy at concurrency=32."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[ar["conc"] == 32]

    workload_order = ["decode_heavy", "balanced", "prefill_heavy"]
    workload_labels = {"decode_heavy": "Decode-heavy\n(128/128)",
                       "balanced": "Balanced\n(1024/512)",
                       "prefill_heavy": "Prefill-heavy\n(512/256)"}

    models = sorted(sub["model_short"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = sub[sub["model_short"] == model]
        strategies = [s for s in STRATEGY_ORDER if s in mdf["strategy"].unique()]
        x = np.arange(len(workload_order))
        width = 0.8 / len(strategies)

        for i, strat in enumerate(strategies):
            vals = []
            for w in workload_order:
                row = mdf[(mdf["strategy"] == strat) & (mdf["workload"] == w)]
                vals.append(row["tok/s"].mean() if len(row) > 0 else 0)
            offset = i * width - (len(strategies) - 1) * width / 2
            ax.bar(x + offset, vals, width,
                   label=STRATEGY_LABELS.get(strat, strat),
                   color=STRATEGY_COLORS.get(strat))

        ax.set_xticks(x)
        ax.set_xticklabels([workload_labels.get(w, w) for w in workload_order], fontsize=8)
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_title(model)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Workload Sensitivity (conc=32)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_workload_sensitivity.png")
    fig.savefig(OUT_DIR / "fig7_workload_sensitivity.pdf")
    plt.close(fig)
    print("  [7] Workload sensitivity")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8: Memory vs Throughput Pareto
# ──────────────────────────────────────────────────────────────────────────────
def fig8_memory_throughput_pareto(df: pd.DataFrame):
    """Scatter: GPU memory usage vs throughput, colored by strategy."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[ar["conc"] == 32]  # Fixed concurrency for fair comparison

    fig, ax = plt.subplots(figsize=(9, 6))
    for strat in STRATEGY_ORDER:
        sdf = sub[sub["strategy"] == strat]
        if sdf.empty:
            continue
        ax.scatter(sdf["mem_after_mb"] / 1024, sdf["tok/s"],
                   label=STRATEGY_LABELS.get(strat, strat),
                   color=STRATEGY_COLORS.get(strat),
                   s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
        # Label each point with model name
        for _, row in sdf.iterrows():
            ax.annotate(MODEL_SHORT.get(row["model"], row["model"]),
                        (row["mem_after_mb"] / 1024, row["tok/s"]),
                        fontsize=6, alpha=0.7,
                        textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("GPU Memory per GPU (GB)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Memory–Throughput Trade-off (conc=32)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT_DIR / "fig8_memory_throughput_pareto.png")
    fig.savefig(OUT_DIR / "fig8_memory_throughput_pareto.pdf")
    plt.close(fig)
    print("  [8] Memory-throughput Pareto")


# ──────────────────────────────────────────────────────────────────────────────
# Table 1: ANOVA-style analysis
# ──────────────────────────────────────────────────────────────────────────────
def table_anova(df: pd.DataFrame):
    """Two-way ANOVA: strategy × model on throughput, for balanced workload."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[ar["workload"] == "balanced"]
    if sub.empty:
        sub = ar

    try:
        from scipy import stats

        # One-way ANOVA per factor
        print("\n  === ANOVA Results ===")

        # Factor: strategy
        groups_strat = [g["tok/s"].values for _, g in sub.groupby("strategy")]
        if len(groups_strat) >= 2:
            f_stat, p_val = stats.f_oneway(*groups_strat)
            print(f"  Strategy effect:   F={f_stat:.2f}, p={p_val:.4e} {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'}")

        # Factor: model
        groups_model = [g["tok/s"].values for _, g in sub.groupby("model")]
        if len(groups_model) >= 2:
            f_stat, p_val = stats.f_oneway(*groups_model)
            print(f"  Model effect:      F={f_stat:.2f}, p={p_val:.4e} {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'}")

        # Factor: concurrency
        groups_conc = [g["tok/s"].values for _, g in sub.groupby("conc")]
        if len(groups_conc) >= 2:
            f_stat, p_val = stats.f_oneway(*groups_conc)
            print(f"  Concurrency effect: F={f_stat:.2f}, p={p_val:.4e} {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'}")

        # Two-way: strategy × model interaction (using OLS if statsmodels available)
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            sub_clean = sub[["tok/s", "strategy", "model_short", "conc"]].dropna()
            sub_clean.columns = ["throughput", "strategy", "model", "conc"]
            model_fit = ols("throughput ~ C(strategy) * C(model) + C(conc)", data=sub_clean).fit()
            anova_table = sm.stats.anova_lm(model_fit, typ=2)
            print("\n  Two-way ANOVA (strategy × model + concurrency):")
            print(anova_table.to_string(float_format=lambda x: f"{x:.3f}"))

            # Save to file
            anova_table.to_csv(OUT_DIR / "anova_results.csv", float_format="%.4f")
            print(f"\n  Saved: {OUT_DIR / 'anova_results.csv'}")
        except ImportError:
            print("  (Install statsmodels for full two-way ANOVA: pip install statsmodels)")

    except ImportError:
        print("  (Install scipy for ANOVA: pip install scipy)")


# ──────────────────────────────────────────────────────────────────────────────
# Table 2: Summary statistics LaTeX
# ──────────────────────────────────────────────────────────────────────────────
def table_summary_latex(df: pd.DataFrame):
    """Generate LaTeX summary table for the paper."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[ar["workload"] == "balanced"]
    if sub.empty:
        sub = ar

    agg = sub.groupby(["model_short", "strategy"]).agg(
        throughput_mean=("tok/s", "mean"),
        throughput_max=("tok/s", "max"),
        ttft_mean=("ttft_mean", "mean"),
        itl_mean=("itl_mean", "mean"),
        gpus=("gpus", "first"),
        mem_gb=("mem_after_mb", lambda x: x.mean() / 1024),
    ).reset_index()

    agg["strategy_label"] = agg["strategy"].map(STRATEGY_LABELS)
    agg = agg.sort_values(["model_short", "strategy"])

    # LaTeX output
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-GPU placement strategy comparison (balanced workload, averaged across concurrency levels).}",
        r"\label{tab:strategy-comparison}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Strategy & GPUs & Tput$_{\text{avg}}$ & Tput$_{\text{max}}$ & TTFT & ITL & Mem \\",
        r" & & & (tok/s) & (tok/s) & (ms) & (ms) & (GB) \\",
        r"\midrule",
    ]

    prev_model = None
    for _, row in agg.iterrows():
        model_col = row["model_short"] if row["model_short"] != prev_model else ""
        if row["model_short"] != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = row["model_short"]
        lines.append(
            f"  {model_col} & {row['strategy_label']} & {int(row['gpus'])} & "
            f"{row['throughput_mean']:.0f} & {row['throughput_max']:.0f} & "
            f"{row['ttft_mean']:.1f} & {row['itl_mean']:.1f} & {row['mem_gb']:.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    (OUT_DIR / "table_strategy_comparison.tex").write_text(latex)
    print(f"\n  Saved: {OUT_DIR / 'table_strategy_comparison.tex'}")
    print("\n  LaTeX preview:")
    for line in lines:
        print(f"    {line}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9: Expert Count vs EP Benefit
# ──────────────────────────────────────────────────────────────────────────────
def fig9_expert_count_vs_ep_benefit(df: pd.DataFrame):
    """Show that EP benefit scales with number of experts."""
    multi = get_multi_gpu(df)
    ar = get_autoregressive(multi)
    sub = ar[(ar["workload"] == "balanced") & (ar["conc"] == 32)]
    if sub.empty:
        sub = ar[ar["conc"] == 32]

    expert_counts = {
        "Mixtral-8x7B-Instruct-v0.1": 8,
        "OLMoE-1B-7B-0924": 64,
        "Qwen1.5-MoE-A2.7B": 60,
    }

    records = []
    for model, n_experts in expert_counts.items():
        mdf = sub[sub["model"] == model]
        tp_only = mdf[mdf["strategy"] == "tp_only"]["tok/s"].mean()
        ep_only = mdf[mdf["strategy"] == "ep_only"]["tok/s"].mean()
        if tp_only > 0 and ep_only > 0:
            records.append({
                "model": MODEL_SHORT.get(model, model),
                "experts": n_experts,
                "ep_vs_tp_ratio": ep_only / tp_only,
                "ep_tput": ep_only,
                "tp_tput": tp_only,
            })

    if not records:
        print("  [9] Skipped (insufficient data)")
        return

    rdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(rdf["model"], rdf["ep_vs_tp_ratio"], color=STRATEGY_COLORS["ep_only"],
           edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="EP = TP (breakeven)")

    for i, row in rdf.iterrows():
        ax.text(i, row["ep_vs_tp_ratio"] + 0.02,
                f"{row['ep_vs_tp_ratio']:.2f}x\n({int(row['experts'])} experts)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("EP / TP Throughput Ratio")
    ax.set_title("Expert Parallelism Benefit vs Expert Count (conc=32, balanced)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=max(rdf["ep_vs_tp_ratio"]) * 1.3)

    fig.savefig(OUT_DIR / "fig9_expert_count_ep_benefit.png")
    fig.savefig(OUT_DIR / "fig9_expert_count_ep_benefit.pdf")
    plt.close(fig)
    print("  [9] Expert count vs EP benefit")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading data from {CSV_PATH}")
    df = load_data()
    print(f"Loaded {len(df)} rows: {df['model'].nunique()} models, "
          f"{df['strategy'].nunique()} strategies\n")

    print("Generating paper figures:")
    fig1_strategy_comparison(df)
    fig2_throughput_scaling(df)
    fig3_latency_breakdown(df)
    fig4_throughput_per_gpu(df)
    fig5_saturation_analysis(df)
    fig6_diffusion_vs_autoregressive(df)
    fig7_workload_sensitivity(df)
    fig8_memory_throughput_pareto(df)
    fig9_expert_count_vs_ep_benefit(df)

    print("\nGenerating tables:")
    table_anova(df)
    table_summary_latex(df)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
