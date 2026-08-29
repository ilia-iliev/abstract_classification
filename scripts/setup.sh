#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

uv sync
uv run python -m scripts.download_model

echo
echo "Setup complete. Start the API with:"
echo "  uv run manage.py runserver"
