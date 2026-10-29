#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [[ $# -eq 1 && "$1" == "--help" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Error: project virtual environment not found at $VENV_DIR" >&2
    exit 1
  fi
  cd "$PROJECT_DIR"
  "$VENV_DIR/bin/python" -m dejargonizer.combine_pairs --help
  exit 0
fi

if [[ $# -lt 3 ]]; then
  echo "Usage: ./combine_pairs.sh INPUT_A INPUT_B [INPUT_N ...] --output OUTPUT" >&2
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" -m dejargonizer.combine_pairs "$@"
