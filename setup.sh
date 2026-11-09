#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
backend="${1:-cpu}"
case "$backend" in
  cpu|cu124) ;;
  *) echo 'Usage: bash setup.sh [cpu|cu124]' >&2; exit 2 ;;
esac
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
  exit 1
fi
uv python install 3.11
uv sync --frozen --extra "$backend" --group dev
uv run --no-sync pytest -q
echo 'Ready. Use: uv run --no-sync mulg --help'
