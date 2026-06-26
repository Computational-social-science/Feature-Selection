# setup.py

from setuptools import setup, find_packages

setup(
    name="feature_selection_factory",
    version="1.0.0",
    description="LLM-driven Feature Selection Operator Evolution System",
    author="Feature Selection Factory Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.12.0"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=0.5.0"],
    },
    entry_points={
        "console_scripts": [
            "fs-evolve=scripts.run_evolution:main",
            "fs-benchmark=scripts.run_benchmark:main",
            "fs-compare=scripts.run_comparison:main",
        ],
    },
)