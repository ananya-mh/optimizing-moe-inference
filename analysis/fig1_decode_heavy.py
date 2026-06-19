"""Generate strategy comparison bar chart using decode_heavy at peak concurrency."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

df = pd.read_csv(CSV_PATH)
df["model_short"] = df["model"].map(MODEL_SHORT).fillna(df["model"])

# Filter: multi-GPU, autoregressive models, decode_heavy workload
ar_models = list(MODEL_SHORT.keys())
sub = df[(df["run"] == "multi") & (df["model"].isin(ar_models)) & (df["workload"] == "decode_heavy")]
sub = sub[sub["strategy"].isin(STRATEGY_ORDER)]

# Peak throughput per (model, strategy)
pivot = sub.groupby(["model_short", "strategy"])["tok/s"].max().reset_index()

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
        label=STRATEGY_LABELS[strat],
        color=STRATEGY_COLORS[strat],
        edgecolor="white", linewidth=0.5,
    )
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Placement Strategy Comparison (decode-heavy, peak concurrency)")
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

fig.savefig(OUT_DIR / "fig1_strategy_comparison.png")
fig.savefig(OUT_DIR / "fig1_strategy_comparison.pdf")
plt.close(fig)
print("Saved fig1_strategy_comparison.png and .pdf")
