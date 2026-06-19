"""Latency insights at concurrency=32: TTFT, ITL, E2E across workloads and strategies."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PROJECT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT / "results" / "tables" / "master_results_clean.csv"
OUT_DIR = PROJECT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_LABELS = {
    "tp_only": "TP-only (2 GPU)",
    "ep_only": "EP-only (2 GPU)",
    "tp_ep_hybrid": "TP+EP Hybrid (4 GPU)",
}
STRATEGY_ORDER = ["tp_only", "ep_only", "tp_ep_hybrid"]
STRATEGY_COLORS = {
    "tp_only": "#3182bd",
    "ep_only": "#e6550d",
    "tp_ep_hybrid": "#31a354",
}

MODEL_SHORT = {
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "OLMoE-1B-7B-0924": "OLMoE-1B-7B",
    "Qwen1.5-MoE-A2.7B": "Qwen-MoE-A2.7B",
}

WORKLOAD_LABELS = {
    "balanced": "Balanced",
    "decode_heavy": "Decode-heavy",
    "prefill_heavy": "Prefill-heavy",
}
WORKLOAD_ORDER = ["balanced", "decode_heavy", "prefill_heavy"]

df = pd.read_csv(CSV_PATH)
df["model_short"] = df["model"].map(MODEL_SHORT).fillna(df["model"])

ar_models = list(MODEL_SHORT.keys())
sub = df[(df["run"] == "multi") & (df["model"].isin(ar_models)) & (df["conc"] == 32)]
sub = sub[sub["strategy"].isin(STRATEGY_ORDER)]


def grouped_bars(ax, data, models, strategies, metric, fmt=".1f", fontsize=7):
    """Draw grouped bars for a single metric."""
    x = np.arange(len(models))
    width = 0.8 / len(strategies)
    for i, strat in enumerate(strategies):
        vals = []
        for m in models:
            row = data[(data["model_short"] == m) & (data["strategy"] == strat)]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        bars = ax.bar(
            x + i * width - (len(strategies) - 1) * width / 2,
            vals, width,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)


# ── Panel 1: TTFT and ITL side-by-side (mean across workloads) ──────────────
agg_mean = sub.groupby(["model_short", "strategy"]).agg(
    ttft_mean=("ttft_mean", "mean"),
    ttft_p99=("ttft_p99", "mean"),
    itl_mean=("itl_mean", "mean"),
    itl_p99=("itl_p99", "mean"),
    e2e_mean=("e2e_mean", "mean"),
).reset_index()

models = sorted(agg_mean["model_short"].unique())
strategies = [s for s in STRATEGY_ORDER if s in agg_mean["strategy"].unique()]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

grouped_bars(axes[0], agg_mean, models, strategies, "ttft_mean")
axes[0].set_ylabel("TTFT (ms)")
axes[0].set_title("Mean TTFT")

grouped_bars(axes[1], agg_mean, models, strategies, "itl_mean")
axes[1].set_ylabel("ITL (ms)")
axes[1].set_title("Mean Inter-Token Latency")

grouped_bars(axes[2], agg_mean, models, strategies, "e2e_mean")
axes[2].set_ylabel("E2E (ms)")
axes[2].set_title("Mean End-to-End Latency")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(strategies), framealpha=0.9,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Latency Breakdown at Concurrency=32 (averaged across workloads)", y=1.07, fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "conc_32_ttft.png")
fig.savefig(OUT_DIR / "conc_32_ttft.pdf")
plt.close(fig)
print("Saved conc_32_ttft.png / .pdf")


# ── Panel 2: TTFT mean vs p99 (tail latency gap) ────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

grouped_bars(axes2[0], agg_mean, models, strategies, "ttft_mean")
axes2[0].set_ylabel("TTFT (ms)")
axes2[0].set_title("TTFT Mean")

grouped_bars(axes2[1], agg_mean, models, strategies, "ttft_p99")
axes2[1].set_ylabel("TTFT p99 (ms)")
axes2[1].set_title("TTFT p99 (tail latency)")

handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=len(strategies), framealpha=0.9,
            bbox_to_anchor=(0.5, 1.02))
fig2.suptitle("TTFT Mean vs Tail Latency at Concurrency=32", y=1.07, fontsize=14)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "conc_32_ttft_tail.png")
fig2.savefig(OUT_DIR / "conc_32_ttft_tail.pdf")
plt.close(fig2)
print("Saved conc_32_ttft_tail.png / .pdf")


# ── Panel 3: TTFT by workload (how prefill/decode mix affects first-token) ──
agg_wl = sub.groupby(["model_short", "strategy", "workload"]).agg(
    ttft_mean=("ttft_mean", "mean"),
).reset_index()

workloads = [w for w in WORKLOAD_ORDER if w in agg_wl["workload"].unique()]
fig3, axes3 = plt.subplots(1, len(workloads), figsize=(6 * len(workloads), 5), sharey=True)
if len(workloads) == 1:
    axes3 = [axes3]

for j, wl in enumerate(workloads):
    wl_data = agg_wl[agg_wl["workload"] == wl]
    grouped_bars(axes3[j], wl_data, models, strategies, "ttft_mean")
    axes3[j].set_title(WORKLOAD_LABELS.get(wl, wl))
    if j == 0:
        axes3[j].set_ylabel("TTFT (ms)")

handles, labels = axes3[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="upper center", ncol=len(strategies), framealpha=0.9,
            bbox_to_anchor=(0.5, 1.02))
fig3.suptitle("TTFT by Workload at Concurrency=32", y=1.07, fontsize=14)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "conc_32_ttft_by_workload.png")
fig3.savefig(OUT_DIR / "conc_32_ttft_by_workload.pdf")
plt.close(fig3)
print("Saved conc_32_ttft_by_workload.png / .pdf")
