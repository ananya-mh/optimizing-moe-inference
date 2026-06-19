"""Train and evaluate the CPU-based placement predictor with 80/20 split.

Unlike leave-one-model-out, this uses a random 80/20 split across all
rows. Note: this WILL have the same models in train and test sets,
so accuracy will be higher but less meaningful for generalization.

This is included to show the predictor works when the model is known,
i.e. "given some benchmarks for this model, predict throughput for
untested configurations of the same model."

Usage:
    python analysis/train_predictor_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
)

PROJECT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT / "results" / "tables" / "master_results_clean.csv"
MODELS_YAML = PROJECT / "configs" / "models.yaml"

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
    with open(MODELS_YAML) as f:
        config = yaml.safe_load(f)
    registry = {}
    for key, m in config["models"].items():
        short_name = m["hf_model_id"].split("/")[-1]
        registry[short_name] = m
    return registry


def load_multi_gpu_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["run"] == "multi"].copy()

    ar_models = ["Mixtral-8x7B-Instruct-v0.1", "OLMoE-1B-7B-0924", "Qwen1.5-MoE-A2.7B"]
    df = df[df["model"].isin(ar_models)]

    registry = load_model_registry()

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
    best_rows = []
    for (model, workload, conc), group in df.groupby(["model", "workload", "conc"]):
        best_idx = group["tok/s"].idxmax()
        row = group.loc[best_idx].copy()
        row["best_strategy"] = row["strategy"]
        best_rows.append(row)
    return pd.DataFrame(best_rows)


def main():
    print("Loading multi-GPU benchmark data...")
    df = load_multi_gpu_data()
    best_df = build_classification_targets(df)

    print(f"Total rows: {len(df)}")
    print(f"Classification groups: {len(best_df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Strategies: {sorted(df['strategy'].unique())}")

    scaler = StandardScaler()
    le = LabelEncoder()

    # ── Throughput Regression (80/20) ──
    print("\n" + "=" * 60)
    print("THROUGHPUT REGRESSION (80/20 split)")
    print("=" * 60)

    X_reg = df[FEATURE_COLS].values
    y_reg = df["tok/s"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_reg, y_reg, np.arange(len(df)),
        test_size=0.2, random_state=42,
    )

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    reg.fit(X_train_s, y_train)
    preds = reg.predict(X_test_s)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"R² = {r2:.3f}")
    print(f"MAE = {mae:.1f} tok/s")

    # Show test set composition
    test_models = df.iloc[idx_test]["model"].value_counts()
    print(f"\nTest set model distribution:\n{test_models.to_string()}")

    # Sample predictions
    print("\nSample predictions:")
    sample_idx = np.linspace(0, len(idx_test) - 1, min(10, len(idx_test)), dtype=int)
    for si in sample_idx:
        row = df.iloc[idx_test[si]]
        print(f"  {row['model'][:15]:15s} {row['strategy']:15s} wl={row['workload']:15s} "
              f"conc={int(row['conc']):>3d}  actual={y_test[si]:>8.1f}  pred={preds[si]:>8.1f}  "
              f"err={abs(y_test[si]-preds[si]):>7.1f}")

    # Feature importance
    print("\nFeature importance (throughput):")
    for name, imp in sorted(zip(FEATURE_COLS, reg.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {name:20s} {imp:.3f} {bar}")

    # ── Strategy Classification (80/20) ──
    print("\n" + "=" * 60)
    print("STRATEGY CLASSIFICATION (80/20 split)")
    print("=" * 60)

    X_cls = best_df[FEATURE_COLS].values
    y_cls = best_df["best_strategy"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_cls, y_cls, np.arange(len(best_df)),
        test_size=0.2, random_state=42, stratify=y_cls,
    )

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train_s, y_train_enc)
    preds = clf.predict(X_test_s)
    pred_labels = le.inverse_transform(preds)

    acc = accuracy_score(y_test_enc, preds)
    majority = pd.Series(y_cls).mode()[0]
    baseline_acc = (y_test == majority).mean()

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Majority baseline (always '{majority}'): {baseline_acc:.3f}")

    # Per-sample
    print("\nPer-sample predictions:")
    for i, idx in enumerate(idx_test):
        row = best_df.iloc[idx]
        match = "OK" if pred_labels[i] == y_test[i] else "MISS"
        print(f"  {match:4s} {row['model'][:15]:15s} wl={row['workload']:15s} "
              f"conc={int(row['conc']):>3d}  pred={pred_labels[i]:15s} actual={y_test[i]:15s}")

    # Feature importance
    print("\nFeature importance (strategy):")
    for name, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {name:20s} {imp:.3f} {bar}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY (80/20 random split)")
    print("=" * 60)
    print(f"Strategy classification: {acc:.3f} (baseline: {baseline_acc:.3f})")
    print(f"Throughput regression:   R²={r2:.3f}, MAE={mae:.1f} tok/s")
    print()
    print("NOTE: Same models appear in train and test sets.")
    print("This measures interpolation (predict unseen configs for known models),")
    print("not generalization (predict for unseen model architectures).")
    print("See train_predictor.py for leave-one-model-out evaluation.")


if __name__ == "__main__":
    main()
