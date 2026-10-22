#!/usr/bin/env bash

set -euo pipefail

SPLIT=""
PROVIDER=""
OLLAMA_FLAG=0
for argument in "$@"; do
  case "$argument" in
    --validation|--test)
      if [[ -n "$SPLIT" ]]; then
        echo "Error: choose only one split flag" >&2
        exit 2
      fi
      SPLIT="${argument#--}"
      ;;
    --ollama) PROVIDER="ollama"; OLLAMA_FLAG=1 ;;
    *) echo "Usage: ./generate_pairs.sh [--validation|--test] [--ollama]" >&2; exit 2 ;;
  esac
done

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
if [[ -z "$PROVIDER" ]]; then
  PROVIDER="$(sed -n 's/^  provider: *//p' "$CONFIG_FILE" | head -1)"
fi
if [[ "$PROVIDER" == "cloudflare" ]]; then
  CREDENTIALS_FILE="$PROJECT_DIR/start.sh"
  if [[ ! -f "$CREDENTIALS_FILE" ]]; then
    echo "Error: Cloudflare credentials file not found at $CREDENTIALS_FILE" >&2
    exit 1
  fi
  set -a
  source "$CREDENTIALS_FILE"
  set +a
  if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" || -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
    echo "Error: start.sh must set Cloudflare credentials" >&2
    exit 1
  fi
fi

echo "Labeling provider: $PROVIDER"
[[ -n "$SPLIT" ]] && echo "$SPLIT-only mode: other split records will be skipped"
if [[ "$OLLAMA_FLAG" -eq 1 && -n "$SPLIT" ]]; then
  dejargonizer-label --provider ollama --split "$SPLIT" \
    --input data/interim/pilot.jsonl --output data/labeled/pilot.jsonl --config configs/pilot.yaml
elif [[ "$OLLAMA_FLAG" -eq 1 ]]; then
  dejargonizer-label --provider ollama \
    --input data/interim/pilot.jsonl --output data/labeled/pilot.jsonl --config configs/pilot.yaml
elif [[ -n "$SPLIT" ]]; then
  dejargonizer-label --split "$SPLIT" \
    --input data/interim/pilot.jsonl --output data/labeled/pilot.jsonl --config configs/pilot.yaml
else
  dejargonizer-label \
    --input data/interim/pilot.jsonl --output data/labeled/pilot.jsonl --config configs/pilot.yaml
fi
