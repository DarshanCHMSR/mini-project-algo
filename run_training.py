"""
One-command entrypoint for reproducible training.

Usage:
  python run_training.py
  python run_training.py --install-missing
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "torch": "torch",
}


def _missing_modules() -> dict[str, str]:
    missing: dict[str, str] = {}
    for module_name, package_name in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module_name) is None:
            missing[module_name] = package_name
    return missing


def _install_packages(packages: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    subprocess.check_call(cmd)


def _check_dataset_exists() -> None:
    data_path = Path("dataset_V2.parquet")
    if not data_path.exists():
        raise FileNotFoundError(
            "dataset_V2.parquet not found in project root. "
            "Place the dataset file in this folder before training."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TCN training with dependency checks.")
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Automatically install missing dependencies before training.",
    )
    args = parser.parse_args()

    missing = _missing_modules()
    if missing:
        packages = sorted(set(missing.values()))
        if args.install_missing:
            print(f"Installing missing packages: {', '.join(packages)}")
            _install_packages(packages)
        else:
            print("Missing dependencies detected:")
            for module_name, package_name in missing.items():
                print(f"  - module '{module_name}' (install package '{package_name}')")
            print("\nRun one of the following:")
            print("  pip install -r requirements.txt")
            print("  python run_training.py --install-missing")
            sys.exit(1)

    _check_dataset_exists()

    # Import only after checks pass.
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
