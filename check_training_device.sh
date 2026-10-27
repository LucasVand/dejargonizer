#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: ./check_training_device.sh" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" - <<'PY'
import torch
from transformers import TrainingArguments

arguments = TrainingArguments(output_dir="artifacts/device-check", report_to="none")

print(f"PyTorch version: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Trainer-selected device: {arguments.device}")
print(f"Number of CUDA GPUs: {arguments.n_gpu}")

if str(arguments.device).startswith("mps"):
    print("Training will use the Apple GPU through MPS.")
elif str(arguments.device).startswith("cuda"):
    print("Training will use an NVIDIA GPU through CUDA.")
else:
    print("Training will use the CPU.")
PY
