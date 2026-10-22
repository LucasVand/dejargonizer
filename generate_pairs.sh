#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--validation" && "$1" != "--test" ) ]]; then
  echo "Usage: ./generate_pairs.sh [--validation|--test]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CREDENTIALS_FILE="$PROJECT_DIR/start.sh"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: project virtual environment not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CREDENTIALS_FILE" ]]; then
  echo "Error: Cloudflare credentials file not found at $CREDENTIALS_FILE" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
set -a
source "$CREDENTIALS_FILE"
set +a

if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" || -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "Error: start.sh must set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
if [[ $# -eq 1 ]]; then
  SPLIT="${1#--}"
  echo "$SPLIT-only mode: other split records will be skipped"
  dejargonizer-label \
    --split "$SPLIT" \
    --input data/interim/pilot.jsonl \
    --output data/labeled/pilot.jsonl \
    --config configs/pilot.yaml
else
  dejargonizer-label \
    --input data/interim/pilot.jsonl \
    --output data/labeled/pilot.jsonl \
    --config configs/pilot.yaml
fi
