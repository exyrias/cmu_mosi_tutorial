#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="${1:-mosi}"
case "$dataset" in
  mosi|mosei) ;;
  *) echo 'Usage: bash run_all.sh [mosi|mosei] [kaggle|sdk] [pretrained-word-vectors.txt]' >&2; exit 2 ;;
esac
source="${2:-}"
if [[ -z "$source" ]]; then
  if [[ "$dataset" == mosi ]]; then source=kaggle; else source=sdk; fi
fi
case "$source" in
  kaggle|sdk) ;;
  *) echo 'Source must be kaggle or sdk' >&2; exit 2 ;;
esac
if [[ "$source" == kaggle && "$dataset" != mosi ]]; then
  echo 'mathurinache/cmu-mosi contains MOSI only; use sdk for MOSEI.' >&2
  exit 2
fi
prepared="data/${dataset}.h5"
run_output="runs/${dataset}"
if [[ "$source" == kaggle ]]; then
  prepared="data/${dataset}_kaggle.h5"
  run_output="runs/${dataset}_kaggle"
fi
# Download and prepare once; a failed download stops the script.
if [[ ! -f "$prepared" ]]; then
  if [[ "$source" == kaggle ]]; then
    embedding_file="${3:-}"
    if [[ ! -f "$embedding_file" ]]; then
      echo 'Supply pretrained plain-text GloVe/word2vec vectors: bash run_all.sh mosi kaggle /path/to/glove.6B.50d.txt' >&2
      exit 2
    fi
    uv run --no-sync mulg download --dataset mosi --output data/kaggle/mosi
    uv run --no-sync mulg download-labels --output data/annotations/mosi.csv
    uv run --no-sync mulg prepare-kaggle --raw-dir data/kaggle/mosi --labels-csv data/annotations/mosi.csv --embedding-file "$embedding_file" --output "$prepared"
  else
    uv run --no-sync mulg download --dataset "$dataset" --source sdk --output "data/raw/${dataset}"
    uv run --no-sync mulg prepare --dataset "$dataset" --raw-dir "data/raw/${dataset}" --output "$prepared"
  fi
fi
uv run --no-sync mulg run --data "$prepared" --output "$run_output" --modalities all --seeds 13 37 73 --resume
