"""Train and evaluate the CPU-based placement predictor.

Evaluation uses leave-one-model-out cross-validation:
  - Train on 2 models, predict the 3rd
  - Repeat for all 3 folds
  - Report per-fold and aggregate accuracy

This tests the real question: can the predictor generalize to an
unseen model architecture?

Usage:
    python analysis/train_predictor.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
)
from joblib import dump

PROJECT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT / "results" / "tables" / "master_results_clean.csv"
MODELS_YAML = PROJECT / "configs" / "models.yaml"
OUT_DIR = PROJECT / "results" / "predictor_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Workload → (input_len, output_len)
WORKLOAD_DIMS = {
    "decode_heavy": (128, 128),
    "balanced": (1024, 512),
    "prefill_heavy": (512, 256),
}

FEATURE_COLS = [
    "total_params_b",
    "active_params_b",
    "num_experts",
    "top_k",
    "sparsity",
    "gpus",
    "conc",
    "input_len",
    "output_len",
]


def load_model_registry() -> dict:
    """Load model metadata from YAML config."""
    with open(MODELS_YAML) as f:
        config = yaml.safe_load(f)
    registry = {}
    for key, m in config["models"].items():
        short_name = m["hf_model_id"].split("/")[-1]
        registry[short_name] = m
    return registry


def load_multi_gpu_data() -> pd.DataFrame:
    """Load CSV, filter to multi-GPU rows, enrich with model metadata."""
    df = pd.read_csv(CSV_PATH)
    df = df[df["run"] == "multi"].copy()

    ar_models = ["Mixtral-8x7B-Instruct-v0.1", "OLMoE-1B-7B-0924", "Qwen1.5-MoE-A2.7B"]
    df = df[df["model"].isin(ar_models)]

    registry = load_model_registry()

    # Enrich with model metadata
    for col, key in [("total_params_b", "total_params_b"),
                     ("active_params_b", "active_params_b"),
                     ("num_experts", "num_experts"),
                     ("top_k", "top_k")]:
        df[col] = df["model"].map(lambda m: registry.get(m, {}).get(key, 0))

    df["sparsity"] = df["active_params_b"] / df["total_params_b"]
    df["input_len"] = df["workload"].map(lambda w: WORKLOAD_DIMS.get(w, (512, 256))[0])
    df["output_len"] = df["workload"].map(lambda w: WORKLOAD_DIMS.get(w, (512, 256))[1])

    return df


def build_classification_targets(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, workload, conc), find the strategy with highest throughput."""
    best_rows = []
    for (model, workload, conc), group in df.groupby(["model", "workload", "conc"]):
        best_idx = group["tok/s"].idxmax()
        row = group.loc[best_idx].copy()
        row["best_strategy"] = row["strategy"]
        best_rows.append(row)
    return pd.DataFrame(best_rows)


# ── Leave-One-Model-Out Evaluation ──────────────────────────────────────────

def evaluate_leave_one_model_out(df: pd.DataFrame, best_df: pd.DataFrame):
    """Primary evaluation: train on 2 models, predict the 3rd."""

    logo = LeaveOneGroupOut()
    scaler = StandardScaler()
    le = LabelEncoder()

    models = sorted(df["model"].unique())
    print(f"Models: {models}")
    print(f"Total multi-GPU rows: {len(df)}")
    print(f"Classification samples (best strategy per group): {len(best_df)}")

    # ── Strategy Classification ──
    print("\n" + "=" * 60)
    print("STRATEGY CLASSIFICATION (Leave-One-Model-Out)")
    print("=" * 60)

    X_cls = best_df[FEATURE_COLS].values
    y_cls = best_df["best_strategy"].values
    model_labels = best_df["model"].values

    le.fit(y_cls)

    fold_results = []
    for train_idx, test_idx in logo.split(X_cls, y_cls, model_labels):
        held_out = model_labels[test_idx[0]]

        X_train = scaler.fit_transform(X_cls[train_idx])
        X_test = scaler.transform(X_cls[test_idx])
        y_train = le.transform(y_cls[train_idx])
        y_test = le.transform(y_cls[test_idx])

        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        correct = sum(preds == y_test)
        total = len(y_test)
        fold_results.append(acc)

        print(f"\n  Held out: {held_out}")
        print(f"  Trained on: {sorted(set(model_labels[train_idx]))}")
        print(f"  Accuracy: {correct}/{total} = {acc:.3f}")

        # Per-sample breakdown
        pred_labels = le.inverse_transform(preds)
        actual_labels = y_cls[test_idx]
        for i, idx in enumerate(test_idx):
            row = best_df.iloc[idx]
            match = "OK" if pred_labels[i] == actual_labels[i] else "MISS"
            print(f"    {match:4s} wl={row['workload']:15s} conc={int(row['conc']):>3d}  "
                  f"pred={pred_labels[i]:15s} actual={actual_labels[i]:15s}")

    mean_acc = np.mean(fold_results)
    print(f"\n  Mean accuracy across folds: {mean_acc:.3f}")

    # Majority-class baseline
    majority = pd.Series(y_cls).mode()[0]
    baseline_acc = (y_cls == majority).mean()
    print(f"  Majority-class baseline (always '{majority}'): {baseline_acc:.3f}")
    print(f"  Improvement over baseline: {mean_acc - baseline_acc:+.3f}")

    # ── Throughput Regression ──
    print("\n" + "=" * 60)
    print("THROUGHPUT REGRESSION (Leave-One-Model-Out)")
    print("=" * 60)

    X_reg = df[FEATURE_COLS].values
    y_reg = df["tok/s"].values
    model_labels_reg = df["model"].values

    fold_results_reg = []
    for train_idx, test_idx in logo.split(X_reg, y_reg, model_labels_reg):
        held_out = model_labels_reg[test_idx[0]]

        X_train = scaler.fit_transform(X_reg[train_idx])
        X_test = scaler.transform(X_reg[test_idx])

        reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        reg.fit(X_train, y_reg[train_idx])
        preds = reg.predict(X_test)

        r2 = r2_score(y_reg[test_idx], preds)
        mae = mean_absolute_error(y_reg[test_idx], preds)
        fold_results_reg.append({"model": held_out, "r2": r2, "mae": mae})

        print(f"\n  Held out: {held_out}")
        print(f"  R² = {r2:.3f}, MAE = {mae:.1f} tok/s")

        # Show a few example predictions
        sample_idx = np.linspace(0, len(test_idx) - 1, min(5, len(test_idx)), dtype=int)
        for si in sample_idx:
            actual = y_reg[test_idx[si]]
            pred = preds[si]
            row = df.iloc[test_idx[si]]
            print(f"    {row['strategy']:15s} wl={row['workload']:15s} conc={int(row['conc']):>3d}  "
                  f"actual={actual:>8.1f}  pred={pred:>8.1f}  err={abs(actual-pred):>7.1f}")

    mean_r2 = np.mean([f["r2"] for f in fold_results_reg])
    mean_mae = np.mean([f["mae"] for f in fold_results_reg])
    print(f"\n  Mean R² across folds: {mean_r2:.3f}")
    print(f"  Mean MAE across folds: {mean_mae:.1f} tok/s")

    return mean_acc, baseline_acc, mean_r2, mean_mae


def train_final_model(df: pd.DataFrame, best_df: pd.DataFrame):
    """Train on all data and save for deployment."""

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL (all data)")
    print("=" * 60)

    scaler = StandardScaler()
    le = LabelEncoder()

    # Classification
    X_cls = scaler.fit_transform(best_df[FEATURE_COLS].values)
    y_cls = le.fit_transform(best_df["best_strategy"].values)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_cls, y_cls)

    # Regression
    X_reg = scaler.transform(df[FEATURE_COLS].values)
    y_reg = df["tok/s"].values
    reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    reg.fit(X_reg, y_reg)

    # Feature importance
    print("\nFeature importance (strategy classification):")
    for name, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {name:20s} {imp:.3f} {bar}")

    print("\nFeature importance (throughput regression):")
    for name, imp in sorted(zip(FEATURE_COLS, reg.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {name:20s} {imp:.3f} {bar}")

    # Save
    dump(clf, OUT_DIR / "strategy_classifier.joblib")
    dump(reg, OUT_DIR / "throughput_regressor.joblib")
    dump(scaler, OUT_DIR / "scaler.joblib")
    dump(le, OUT_DIR / "label_encoder.joblib")
    print(f"\nModels saved to {OUT_DIR}")

    return clf, reg, scaler, le


def main():
    print("Loading multi-GPU benchmark data...")
    df = load_multi_gpu_data()
    print(f"Rows: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print(f"Workloads: {sorted(df['workload'].unique())}")

    print("\nBuilding classification targets...")
    best_df = build_classification_targets(df)
    print(f"Groups (model x workload x concurrency): {len(best_df)}")
    print(f"Best strategy distribution:\n{best_df['best_strategy'].value_counts().to_string()}")

    # Primary evaluation
    mean_acc, baseline_acc, mean_r2, mean_mae = evaluate_leave_one_model_out(df, best_df)

    # Train final model on all data
    train_final_model(df, best_df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Strategy classification (leave-one-model-out): {mean_acc:.3f}")
    print(f"Majority-class baseline:                       {baseline_acc:.3f}")
    print(f"Throughput regression (leave-one-model-out):    R²={mean_r2:.3f}, MAE={mean_mae:.1f} tok/s")


if __name__ == "__main__":
    main()
