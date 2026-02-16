"""Configuration loader and validator for MoE optimization experiments.

MIT License - Copyright (c) 2026 optimizing-moe-inference contributors
"""
import os
import yaml
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_registry() -> dict[str, Any]:
    """Load the model registry from configs/models.yaml."""
    return load_yaml(CONFIGS_DIR / "models.yaml")


def get_model_config(model_key: str) -> dict[str, Any]:
    """Get configuration for a specific model by its key name."""
    registry = get_model_registry()
    models = registry.get("models", {})
    if model_key not in models:
        available = list(models.keys())
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {available}"
        )
    return models[model_key]


def get_experiment_config(experiment_name: str) -> dict[str, Any]:
    """Load experiment configuration by name (single_gpu, multi_gpu, multi_node)."""
    path = CONFIGS_DIR / "experiments" / f"{experiment_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    return load_yaml(path)


def get_model_dir() -> str:
    """Get model weights directory from environment."""
    return os.environ.get("MODEL_DIR", "./models")


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment. Never hardcoded."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def detect_gpu_vendor() -> str:
    """Detect GPU vendor: 'amd' or 'nvidia'.

    Checks for ROCm first (rocm-smi), then CUDA (nvidia-smi).
    """
    import shutil
    if shutil.which("rocm-smi"):
        return "amd"
    elif shutil.which("nvidia-smi"):
        return "nvidia"
    else:
        raise RuntimeError(
            "No GPU tools found. Install ROCm or CUDA toolkit."
        )


def resolve_experiment_env(experiment_cfg: dict, vendor: str) -> dict[str, str]:
    """Resolve environment variables for the detected GPU vendor."""
    exp = experiment_cfg.get("experiment", experiment_cfg)
    if vendor == "amd":
        return exp.get("amd_env", {})
    else:
        return exp.get("nvidia_env", {})
