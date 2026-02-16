"""Generate LaTeX tables for the SIEDS 2026 paper.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"


def generate_model_table() -> str:
    """Generate LaTeX table of MoE model characteristics."""
    with open(CONFIGS_DIR / "models.yaml") as f:
        config = yaml.safe_load(f)

    rows = []
    for key, m in config["models"].items():
        rows.append({
            "Model": m["hf_model_id"].split("/")[-1],
            "Total (B)": m["total_params_b"],
            "Active (B)": m["active_params_b"],
            "Experts": m["num_experts"],
            "Top-k": m["top_k"],
            "Arch": m["architecture"],
            "Min GPUs": m["min_gpus"],
        })

    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, escape=True, column_format="lrrrrll")

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{MoE models used in our evaluation, "
        "spanning 6.9B to 671B total parameters.}\n"
        "\\label{tab:models}\n"
    )
    footer = "\\end{table}"

    return header + latex + footer


def generate_results_table() -> str:
    """Generate LaTeX table from benchmark results."""
    results_files = sorted(RESULTS_DIR.glob("*.json"))
    if not results_files:
        return "% No results available yet"

    records = []
    for f in results_files:
        with open(f) as fp:
            data = json.load(fp)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not item.get("success"):
                    continue
                m = item.get("metrics", {})
                records.append({
                    "Model": item.get("model_id", "").split("/")[-1],
                    "Strategy": item.get("strategy", {}).get("name", "?"),
                    "Conc.": item.get("concurrency", 0),
                    "Throughput": f"{m.get('throughput_tok_per_sec', 0):.1f}",
                    "TTFT (ms)": f"{m.get('ttft_avg_ms', 0):.1f}",
                    "ITL (ms)": f"{m.get('itl_avg_ms', 0):.1f}",
                })

    if not records:
        return "% No successful benchmark results"

    df = pd.DataFrame(records)
    latex = df.to_latex(index=False, escape=True)

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Benchmark results across placement strategies.}\n"
        "\\label{tab:results}\n"
    )
    footer = "\\end{table}"

    return header + latex + footer


def main():
    """Generate all tables and print to stdout."""
    print("% === Model Characteristics Table ===")
    print(generate_model_table())
    print()
    print("% === Results Table ===")
    print(generate_results_table())


if __name__ == "__main__":
    main()
