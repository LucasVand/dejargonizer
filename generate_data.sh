#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: ./generate_data.sh" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CONFIG_FILE="$PROJECT_DIR/configs/pilot.yaml"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: configuration file not found at $CONFIG_FILE" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
dejargonizer-data --config configs/pilot.yaml
