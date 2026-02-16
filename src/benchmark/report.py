"""Report generation from benchmark results.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
from pathlib import Path
from typing import Any

from tabulate import tabulate
from rich.console import Console
from rich.table import Table


def load_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results_dir = Path(results_dir)
    all_results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    return all_results


def summarize_results(results: list[dict]) -> list[dict]:
    """Create summary rows from benchmark results."""
    rows = []
    for r in results:
        if not r.get("success"):
            continue
        m = r.get("metrics", {})
        rows.append({
            "model": r.get("model_id", "?").split("/")[-1],
            "strategy": r.get("strategy", {}).get("name", "?"),
            "workload": r.get("workload", {}).get("name", "?"),
            "concurrency": r.get("concurrency", "?"),
            "throughput": f"{m.get('throughput_tok_per_sec', 0):.1f}",
            "ttft_avg": f"{m.get('ttft_avg_ms', 0):.1f}",
            "itl_avg": f"{m.get('itl_avg_ms', 0):.1f}",
        })
    return rows


def print_summary_table(results: list[dict]):
    """Print a rich summary table."""
    console = Console()
    table = Table(title="MoE Benchmark Results Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Strategy", style="green")
    table.add_column("Workload")
    table.add_column("Conc.", justify="right")
    table.add_column("Throughput (tok/s)", justify="right", style="bold")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("ITL (ms)", justify="right")

    for row in summarize_results(results):
        table.add_row(
            row["model"], row["strategy"], row["workload"],
            str(row["concurrency"]), row["throughput"],
            row["ttft_avg"], row["itl_avg"],
        )

    console.print(table)


def export_csv(results: list[dict], output_path: str | Path):
    """Export results to CSV."""
    import csv
    rows = summarize_results(results)
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV exported to {output_path}")
