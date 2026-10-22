#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./simplify.sh \"technical text to simplify\"" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CHECKPOINT="$PROJECT_DIR/artifacts/t5-small-pilot"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT/config.json" ]]; then
  echo "Error: trained model not found at $CHECKPOINT" >&2
  echo "Run ./train_model.sh first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
dejargonizer "$1" --checkpoint artifacts/t5-small-pilot
