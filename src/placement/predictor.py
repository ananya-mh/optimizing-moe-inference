"""CPU-based ML predictor for expert placement decisions.

Lightweight scikit-learn model that runs on idle CPU resources
to predict optimal placement strategy at runtime.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "predictor_models"


class PlacementPredictor:
    """Predicts optimal placement strategy from profiling features.

    Features:
        - model_total_params_b: Total model parameters (billions)
        - model_active_params_b: Active parameters per token
        - num_experts: Number of routed experts
        - top_k: Experts activated per token
        - num_gpus: Available GPUs
        - gpu_memory_gb: Per-GPU memory (GB)
        - batch_size: Target batch size
        - input_len: Average input sequence length
        - output_len: Average output sequence length

    Targets:
        - strategy: Best placement strategy (classification)
        - queue_depth: Optimal batching queue depth (regression)
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for the ML predictor. "
                "Install: pip install scikit-learn joblib"
            )
        self.strategy_clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.queue_reg = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def _extract_features(self, config: dict[str, Any]) -> np.ndarray:
        """Extract feature vector from configuration dict."""
        return np.array([
            config.get("model_total_params_b", 0),
            config.get("model_active_params_b", 0),
            config.get("num_experts", 0),
            config.get("top_k", 0),
            config.get("num_gpus", 1),
            config.get("gpu_memory_gb", 192),
            config.get("batch_size", 32),
            config.get("input_len", 512),
            config.get("output_len", 256),
        ]).reshape(1, -1)

    def train(self, training_data: list[dict[str, Any]]):
        """Train predictor from collected profiling/benchmark data.

        Each entry should have feature fields plus:
          - best_strategy: str (target for classifier)
          - best_queue_depth: int (target for regressor)
        """
        if len(training_data) < 5:
            print("Warning: Very small training set. Predictions may be unreliable.")

        X = np.vstack([self._extract_features(d) for d in training_data])
        y_strategy = [d["best_strategy"] for d in training_data]
        y_queue = [d["best_queue_depth"] for d in training_data]

        X_scaled = self.scaler.fit_transform(X)
        self.strategy_clf.fit(X_scaled, y_strategy)
        self.queue_reg.fit(X_scaled, y_queue)
        self._fitted = True

        print(f"Predictor trained on {len(training_data)} samples")

    def predict(self, config: dict[str, Any]) -> dict[str, Any]:
        """Predict optimal placement strategy and queue depth."""
        if not self._fitted:
            raise RuntimeError("Predictor not trained. Call train() first.")

        X = self._extract_features(config)
        X_scaled = self.scaler.transform(X)

        strategy = self.strategy_clf.predict(X_scaled)[0]
        queue_depth = max(1, int(self.queue_reg.predict(X_scaled)[0]))

        # Confidence from RF probability
        proba = self.strategy_clf.predict_proba(X_scaled)[0]
        confidence = float(max(proba))

        return {
            "recommended_strategy": strategy,
            "recommended_queue_depth": queue_depth,
            "confidence": round(confidence, 3),
        }

    def save(self, path: Optional[str] = None):
        """Save trained predictor to disk."""
        save_dir = Path(path) if path else MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        dump(self.strategy_clf, save_dir / "strategy_clf.joblib")
        dump(self.queue_reg, save_dir / "queue_reg.joblib")
        dump(self.scaler, save_dir / "scaler.joblib")
        print(f"Predictor saved to {save_dir}")

    def load(self, path: Optional[str] = None):
        """Load trained predictor from disk."""
        load_dir = Path(path) if path else MODEL_DIR
        self.strategy_clf = load(load_dir / "strategy_clf.joblib")
        self.queue_reg = load(load_dir / "queue_reg.joblib")
        self.scaler = load(load_dir / "scaler.joblib")
        self._fitted = True
        print(f"Predictor loaded from {load_dir}")
