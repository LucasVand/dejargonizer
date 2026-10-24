#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: ./verify_pairs.sh" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
INPUT_FILE="$PROJECT_DIR/data/labeled/pilot.jsonl"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: generated pair file not found at $INPUT_FILE" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
dejargonizer-verify-pairs --input data/labeled/pilot.jsonl
