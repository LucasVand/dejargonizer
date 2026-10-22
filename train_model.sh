#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: ./train_model.sh" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CONFIG_FILE="$PROJECT_DIR/configs/pilot.yaml"
INPUT_FILE="$PROJECT_DIR/data/labeled/pilot.jsonl"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: configuration file not found at $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: labeled dataset not found at $INPUT_FILE" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
echo "Starting T5-small training with $INPUT_FILE"
echo "Training configuration: $CONFIG_FILE"
echo "Checkpoint output: artifacts/t5-small-pilot"
dejargonizer-train \
  --input data/labeled/pilot.jsonl \
  --config configs/pilot.yaml
echo "Training complete. Final model saved to artifacts/t5-small-pilot"
