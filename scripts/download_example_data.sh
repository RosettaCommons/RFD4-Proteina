#!/usr/bin/env bash
# Download the example checkpoints (and, with --with-test-dataset, the tiny training set) into a
# local dir you then export as DATA_PATH. Public download over HTTPS — no AWS CLI or credentials.
# Override the host with RFPROTEINA_DATA_URL. See the README Setup section.
#   ./scripts/download_example_data.sh <DATA_PATH> [--with-test-dataset]
set -euo pipefail

DATA_PATH="${1:-}"
if [ -z "$DATA_PATH" ] || [ "${DATA_PATH:0:1}" = "-" ]; then
    echo "Usage: $0 <DATA_PATH> [--with-test-dataset]" >&2
    exit 1
fi
BASE="${RFPROTEINA_DATA_URL:-https://YOUR_DATASET_STORE}"

# Download to <dest-dir>/<basename>, skipping if already present; a partial download lands in a
# .tmp file so an interrupted run re-fetches instead of leaving a corrupt checkpoint.
fetch() {  # fetch <url> <dest-dir>
    mkdir -p "$2"
    local out="$2/$(basename "$1")"
    if [ -f "$out" ]; then echo ">>> have $(basename "$1"), skipping"; return; fi
    curl -fL -o "$out.tmp" "$1" && mv "$out.tmp" "$out"
}

echo ">>> Downloading into DATA_PATH=$DATA_PATH"
fetch "$BASE/latent_checkpoints/ldm_checkpoints/arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt" \
      "$DATA_PATH/latent_checkpoints/ldm_checkpoints"
fetch "$BASE/latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt" \
      "$DATA_PATH/latent_checkpoints/ae_checkpoints"

if [ "${2:-}" = "--with-test-dataset" ]; then
    echo ">>> test_dataset..."
    mkdir -p "$DATA_PATH/test_dataset"
    curl -fL "$BASE/test_dataset/test_dataset.tar.gz" | tar -xzf - -C "$DATA_PATH/test_dataset"
fi

echo ">>> Done. Now: export DATA_PATH=$DATA_PATH"
