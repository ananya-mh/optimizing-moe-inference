#!/usr/bin/env python3
"""Train the CPU-based placement predictor from benchmark results.

Loads all collected result JSONs, derives best_strategy and best_queue_depth
labels for each (model, num_gpus, workload, concurrency) configuration, then
trains and evaluates PlacementPredictor with leave-one-model-out cross-validation.

Usage (from repo root):
    python scripts/train_predictor.py
    python scripts/train_predictor.py --results-dir results/ --output results/predictor_models/
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from placement.predictor import PlacementPredictor

# Model metadata keyed by HF model ID
MODEL_META = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "total_params_b": 46.7,
        "active_params_b": 12.9,
        "num_experts": 8,
        "top_k": 2,
    },
    "allenai/OLMoE-1B-7B-0924": {
        "total_params_b": 6.9,
        "active_params_b": 1.3,
        "num_experts": 64,
        "top_k": 8,
    },
    "Qwen/Qwen1.5-MoE-A2.7B": {
        "total_params_b": 14.3,
        "active_params_b": 2.7,
        "num_experts": 60,
        "top_k": 4,
    },
    "GSAI-ML/LLaDA-8B-Instruct": {
        "total_params_b": 8.0,
        "active_params_b": 8.0,
        "num_experts": 1,
        "top_k": 1,
    },
    "inclusionAI/LLaDA-MoE-7B-A1B-Instruct": {
        "total_params_b": 7.0,
        "active_params_b": 1.4,
        "num_experts": 8,
        "top_k": 2,
    },
}

# Normalize strategy names from result files to canonical names used in strategies.py
STRATEGY_NORM = {
    "baseline_no_ep": "colocate_single",
    "colocate_single": "colocate_single",
    "tp_only": "tp_only",
    "ep_only": "ep_only",
    "tp_ep_hybrid": "tp_ep_hybrid",
    "dp_ep": "dp_ep",
}

FEATURE_NAMES = [
    "total_params_b",
    "active_params_b",
    "num_experts",
    "top_k",
    "num_gpus",
    "gpu_memory_gb",
    "batch_size",
    "input_len",
    "output_len",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> list[dict]:
    """Load all benchmark result JSON files from results_dir."""
    entries = []
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"  Warning: no JSON files found in {results_dir}")
        return []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                entry["_source_file"] = json_file.name
            entries.extend(data)
        else:
            print(f"  Warning: unexpected format in {json_file.name} (not a list), skipping")
    print(f"Loaded {len(entries)} raw entries from {len(json_files)} files in {results_dir}")
    return entries


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_entry(entry: dict) -> dict | None:
    """Extract a flat feature+metric dict from a raw result entry.

    Returns None if the entry should be skipped (failed run, unknown model,
    missing/zero throughput, or LLaDA diffusion models whose inference
    paradigm differs fundamentally from autoregressive MoE).
    """
    if not entry.get("success", False):
        return None

    model_id = entry.get("model_id", "")
    meta = MODEL_META.get(model_id)
    if meta is None:
        return None

    # Exclude diffusion LLMs — they use a custom engine and have fundamentally
    # different performance characteristics (masked diffusion, not token generation).
    if model_id in ("GSAI-ML/LLaDA-8B-Instruct", "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"):
        return None

    metrics = entry.get("metrics", {})
    throughput = metrics.get("throughput_tok_per_sec")
    if throughput is None or throughput <= 0:
        return None

    strategy = entry.get("strategy", {})
    strategy_raw = strategy.get("name", "unknown")
    strategy_name = STRATEGY_NORM.get(strategy_raw)
    if strategy_name is None:
        print(f"  Warning: unknown strategy '{strategy_raw}' in {entry.get('_source_file','?')}, skipping")
        return None

    workload = entry.get("workload", {})
    gpus_list = entry.get("gpu_before", {}).get("gpus", [{}])
    num_gpus = len(gpus_list)
    gpu_memory_gb = gpus_list[0].get("memory_total_mb", 81920.0) / 1024.0

    return {
        # Identity (not used as features, used for grouping/labelling)
        "model_id": model_id,
        "strategy_name": strategy_name,
        "workload_name": workload.get("name", "unknown"),
        "concurrency": entry.get("concurrency", 1),
        "num_gpus": num_gpus,
        # Predictor features
        "model_total_params_b": meta["total_params_b"],
        "model_active_params_b": meta["active_params_b"],
        "num_experts": meta["num_experts"],
        "top_k": meta["top_k"],
        "gpu_memory_gb": gpu_memory_gb,
        "batch_size": entry.get("concurrency", 1),
        "input_len": workload.get("input_len", 512),
        "output_len": workload.get("output_len", 256),
        # Metrics (used only for label derivation)
        "throughput_tok_per_sec": throughput,
        "ttft_avg_ms": metrics.get("ttft_avg_ms", 0.0),
        "itl_avg_ms": metrics.get("itl_avg_ms", 0.0),
    }


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def derive_labels(parsed: list[dict]) -> list[dict]:
    """Derive best_strategy and best_queue_depth labels from parsed entries.

    Strategy label: for each (model, num_gpus, workload, concurrency) group,
    the strategy with the highest throughput is the winner.

    Queue-depth label: for each (model, num_gpus, workload) + the winning
    strategy, find the concurrency level where throughput peaks — that is
    the recommended queue depth to run at.

    Returns one training sample per unique (model, num_gpus, workload,
    concurrency) combination (deduped across strategies).
    """
    # Group by competition key → compare strategies
    strategy_groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in parsed:
        key = (p["model_id"], p["num_gpus"], p["workload_name"], p["concurrency"])
        strategy_groups[key].append(p)

    # Winner per group
    best_strategy_map: dict[tuple, str] = {}
    for key, entries in strategy_groups.items():
        best = max(entries, key=lambda e: e["throughput_tok_per_sec"])
        best_strategy_map[key] = best["strategy_name"]

    # Peak concurrency per (model, num_gpus, workload, strategy)
    # Used as the queue-depth label
    concurrency_groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in parsed:
        key = (p["model_id"], p["num_gpus"], p["workload_name"], p["strategy_name"])
        concurrency_groups[key].append(p)

    peak_concurrency_map: dict[tuple, int] = {}
    for key, entries in concurrency_groups.items():
        peak = max(entries, key=lambda e: e["throughput_tok_per_sec"])
        peak_concurrency_map[key] = peak["concurrency"]

    # Build one training sample per unique competition group
    training_data = []
    seen: set[tuple] = set()
    for p in parsed:
        comp_key = (p["model_id"], p["num_gpus"], p["workload_name"], p["concurrency"])
        if comp_key in seen:
            continue
        seen.add(comp_key)

        best_strat = best_strategy_map[comp_key]
        queue_key = (p["model_id"], p["num_gpus"], p["workload_name"], best_strat)
        best_queue = peak_concurrency_map.get(queue_key, p["concurrency"])

        sample = {
            k: v for k, v in p.items()
            if k not in ("strategy_name", "throughput_tok_per_sec", "ttft_avg_ms", "itl_avg_ms")
        }
        sample["best_strategy"] = best_strat
        sample["best_queue_depth"] = best_queue
        training_data.append(sample)

    return training_data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_loo(training_data: list[dict]) -> dict[str, dict]:
    """Leave-one-model-out cross-validation.

    For each model, trains on all other models and evaluates on the held-out
    one. Measures strategy classification accuracy.
    """
    models = sorted({d["model_id"] for d in training_data})
    results: dict[str, dict] = {}

    for test_model in models:
        train_split = [d for d in training_data if d["model_id"] != test_model]
        test_split = [d for d in training_data if d["model_id"] == test_model]

        if len(train_split) < 5 or len(test_split) == 0:
            print(f"  Skipping {test_model.split('/')[-1]}: insufficient data")
            continue

        predictor = PlacementPredictor()
        predictor.train(train_split)

        correct = 0
        for sample in test_split:
            pred = predictor.predict(sample)
            if pred["recommended_strategy"] == sample["best_strategy"]:
                correct += 1

        acc = correct / len(test_split)
        model_name = test_model.split("/")[-1]
        results[model_name] = {
            "accuracy": round(acc, 3),
            "correct": correct,
            "total": len(test_split),
            "dominant_label": Counter(d["best_strategy"] for d in test_split).most_common(1)[0],
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results/ directory (default: results/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override save directory for trained model files (default: results/predictor_models/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    print("\n=== Loading benchmark results ===")
    raw = load_results(results_dir)

    # ------------------------------------------------------------------
    print("\n=== Parsing entries ===")
    parsed = []
    skipped = 0
    for entry in raw:
        p = parse_entry(entry)
        if p:
            parsed.append(p)
        else:
            skipped += 1
    print(f"Valid: {len(parsed)}  |  Skipped: {skipped}")

    if len(parsed) < 10:
        print("Error: too few valid entries to train.")
        sys.exit(1)

    strat_dist = Counter(p["strategy_name"] for p in parsed)
    print("\nStrategy distribution in parsed data:")
    for name, count in sorted(strat_dist.items()):
        print(f"  {name:<20} {count:>4} entries")

    model_dist = Counter(p["model_id"].split("/")[-1] for p in parsed)
    print("\nModel distribution in parsed data:")
    for name, count in sorted(model_dist.items()):
        print(f"  {name:<35} {count:>4} entries")

    gpu_dist = Counter(p["num_gpus"] for p in parsed)
    print("\nGPU-count distribution:")
    for ngpu, count in sorted(gpu_dist.items()):
        print(f"  {ngpu} GPU(s): {count} entries")

    # ------------------------------------------------------------------
    print("\n=== Deriving labels ===")
    training_data = derive_labels(parsed)
    print(f"Training samples (unique configs): {len(training_data)}")

    label_dist = Counter(d["best_strategy"] for d in training_data)
    print("Label distribution:")
    for name, count in sorted(label_dist.items()):
        bar = "#" * (count * 40 // max(label_dist.values()))
        print(f"  {name:<20} {count:>4}  {bar}")

    queue_depths = [d["best_queue_depth"] for d in training_data]
    print(f"Queue depth range: {min(queue_depths)} – {max(queue_depths)}  "
          f"(median={int(np.median(queue_depths))})")

    # ------------------------------------------------------------------
    print("\n=== Leave-one-model-out cross-validation ===")
    loo = evaluate_loo(training_data)
    for model_name, res in loo.items():
        dominant_label, dominant_count = res["dominant_label"]
        baseline = dominant_count / res["total"]
        print(
            f"  {model_name:<35}  acc={res['accuracy']:.1%}  "
            f"({res['correct']}/{res['total']})  "
            f"[majority-class baseline={baseline:.1%}: '{dominant_label}']"
        )
    if loo:
        total_correct = sum(r["correct"] for r in loo.values())
        total_samples = sum(r["total"] for r in loo.values())
        print(f"\n  Overall LOO accuracy: {total_correct/total_samples:.1%}  "
              f"({total_correct}/{total_samples})")

    # ------------------------------------------------------------------
    print("\n=== Training final predictor on all data ===")
    predictor = PlacementPredictor()
    predictor.train(training_data)

    importances = predictor.strategy_clf.feature_importances_
    print("\nFeature importances (strategy classifier):")
    for feat, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = "#" * max(1, int(imp * 50))
        print(f"  {feat:<22}  {imp:.4f}  {bar}")

    queue_importances = predictor.queue_reg.feature_importances_
    print("\nFeature importances (queue-depth regressor):")
    for feat, imp in sorted(zip(FEATURE_NAMES, queue_importances), key=lambda x: -x[1]):
        bar = "#" * max(1, int(imp * 50))
        print(f"  {feat:<22}  {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    print("\n=== Saving model ===")
    predictor.save(args.output)

    # ------------------------------------------------------------------
    print("\n=== Sanity-check predictions ===")
    test_configs = [
        {
            "desc": "Mixtral-8x7B   1 GPU  (should: colocate_single)",
            "model_total_params_b": 46.7, "model_active_params_b": 12.9,
            "num_experts": 8, "top_k": 2,
            "num_gpus": 1, "gpu_memory_gb": 80.0,
            "batch_size": 32, "input_len": 128, "output_len": 128,
        },
        {
            "desc": "OLMoE-1B-7B    2 GPUs (should: ep_only — comm overhead wins at high expert count)",
            "model_total_params_b": 6.9, "model_active_params_b": 1.3,
            "num_experts": 64, "top_k": 8,
            "num_gpus": 2, "gpu_memory_gb": 80.0,
            "batch_size": 16, "input_len": 128, "output_len": 128,
        },
        {
            "desc": "Qwen-MoE-A2.7B 4 GPUs (should: tp_ep_hybrid)",
            "model_total_params_b": 14.3, "model_active_params_b": 2.7,
            "num_experts": 60, "top_k": 4,
            "num_gpus": 4, "gpu_memory_gb": 192.0,
            "batch_size": 64, "input_len": 512, "output_len": 256,
        },
        {
            "desc": "Large MoE 57B  8 GPUs (extrapolation)",
            "model_total_params_b": 57.4, "model_active_params_b": 14.0,
            "num_experts": 64, "top_k": 8,
            "num_gpus": 8, "gpu_memory_gb": 192.0,
            "batch_size": 128, "input_len": 512, "output_len": 512,
        },
    ]
    for cfg in test_configs:
        desc = cfg.pop("desc")
        pred = predictor.predict(cfg)
        print(
            f"  {desc}\n"
            f"    -> strategy={pred['recommended_strategy']:<20}  "
            f"queue_depth={pred['recommended_queue_depth']:<4}  "
            f"confidence={pred['confidence']:.2f}\n"
        )

    print("=== Done ===")


if __name__ == "__main__":
    main()
