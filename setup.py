"""Setup for optimizing-moe-inference package."""
from setuptools import setup, find_packages

setup(
    name="moe-inference-opt",
    version="0.1.0",
    description="MoE inference optimization: expert placement, batching, and scaling studies",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "click>=8.1.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
        "scikit-learn>=1.5.0",
    ],
)
