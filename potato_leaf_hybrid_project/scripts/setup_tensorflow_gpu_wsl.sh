#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3.12 >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y python3.12 python3.12-venv python3-pip
fi

mkdir -p .venvs
python3.12 -m venv .venvs/tf-gpu-wsl
source .venvs/tf-gpu-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install 'tensorflow[and-cuda]' scikit-learn matplotlib pandas pillow streamlit pennylane
python scripts/verify_tensorflow_gpu.py
