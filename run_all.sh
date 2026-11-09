#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="${1:-mosi}"
case "$dataset" in
  mosi|mosei) ;;
  *) echo 'Usage: bash run_all.sh [mosi|mosei]' >&2; exit 2 ;;
esac
# Download and prepare once; a failed download stops the script.
if [[ ! -f "data/${dataset}.h5" ]]; then
  uv run --no-sync mulg download --dataset "$dataset" --output "data/raw/${dataset}"
  uv run --no-sync mulg prepare --dataset "$dataset" --raw-dir "data/raw/${dataset}" --output "data/${dataset}.h5"
fi
uv run --no-sync mulg run --data "data/${dataset}.h5" --output "runs/${dataset}" --modalities all --seeds 13 37 73 --resume
