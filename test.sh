#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: ./test.sh" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CHECKPOINT="$PROJECT_DIR/artifacts/t5-small-pilot"
INPUT_FILE="$PROJECT_DIR/data/labeled/pilot.jsonl"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT/config.json" ]]; then
  echo "Error: trained model not found at $CHECKPOINT" >&2
  echo "Run ./train_model.sh first." >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: labeled dataset not found at $INPUT_FILE" >&2
  echo "Generate test pairs with ./generate_pairs.sh --test first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
echo "Evaluating saved model on accepted test pairs..."
dejargonizer-evaluate \
  --input data/labeled/pilot.jsonl \
  --checkpoint artifacts/t5-small-pilot
