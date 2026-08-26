#!/usr/bin/env bash
# Download the released checkpoints (and, with --with-test-dataset, the tiny training set) into a
# local dir you then export as DATA_PATH. Public HTTPS, no credentials. On non-configured clusters, also
# fetches OpenFold3 by default -- see docs/installation.rst.
#   ./scripts/download_example_data.sh <DATA_PATH> [--with-test-dataset] [--skip-of3]
set -euo pipefail

DATA_PATH="${1:-}"
if [ -z "$DATA_PATH" ] || [ "${DATA_PATH:0:1}" = "-" ]; then
    echo "Usage: $0 <DATA_PATH> [--with-test-dataset] [--skip-of3]" >&2
    exit 1
fi
shift
WITH_TEST_DATASET=false
SKIP_OF3=false
for arg in "$@"; do
    case "$arg" in
        --with-test-dataset) WITH_TEST_DATASET=true ;;
        --skip-of3) SKIP_OF3=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done
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
# Denoiser name must stay in sync with `ckpt_name` in configs/inference.yaml, so a default
# `generate.py` run resolves it with no CLI override.
fetch "$BASE/latent_checkpoints/ldm_checkpoints/2026_07_22_release_280k-ema.ckpt" \
      "$DATA_PATH/latent_checkpoints/ldm_checkpoints"
fetch "$BASE/latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt" \
      "$DATA_PATH/latent_checkpoints/ae_checkpoints"

if [ "$WITH_TEST_DATASET" = true ]; then
    echo ">>> test_dataset..."
    mkdir -p "$DATA_PATH/test_dataset"
    curl -fL "$BASE/test_dataset/test_dataset.tar.gz" | tar -xzf - -C "$DATA_PATH/test_dataset"
fi

# configured clusters already have AF3 configured (see docs/installation.rst); everyone else defaults to
# OpenFold3.
case "${CLUSTER:-}" in
    digs|ord|gb200|coreweave) SKIP_OF3=true ;;
esac
if [ "$SKIP_OF3" = true ]; then
    echo ">>> Skipping OpenFold3 download (configured cluster or --skip-of3)."
elif ! command -v apptainer &>/dev/null; then
    echo ">>> apptainer not found — skipping OpenFold3 download (it can't run without it)."
    echo "    Install apptainer, then re-run this script to fetch it."
else
    echo ">>> Downloading OpenFold3 (free, Apache-licensed refolding oracle, ~5 GB)..."
    fetch "$BASE/af3/apptainer_images/af3-open-36fad16.sif" \
          "$DATA_PATH/af3/apptainer_images"
    fetch "$BASE/af3/af3_weights/of3/of3_ported_weights.bin.zst" \
          "$DATA_PATH/af3/af3_weights/of3"
    fetch "$BASE/af3/af3_weights/of3/of3_ported_weights.provenance.json" \
          "$DATA_PATH/af3/af3_weights/of3"
    echo ">>> OpenFold3 ready. For direct generate.py/CLI use (the tutorial notebook finds these"
    echo "    automatically), export:"
    echo "    export AF3_IMAGE=\"$DATA_PATH/af3/apptainer_images/af3-open-36fad16.sif\""
    echo "    export AF3_MODEL_DIR=\"$DATA_PATH/af3/af3_weights/of3\""
    echo "    export AF3_JAX_CACHE_DIR=\"$DATA_PATH/af3/jax_cache\""
    echo "    ...and pass metrics/refolding_oracles=of3 as a Hydra override."
fi

echo ">>> Done. Now: export DATA_PATH=$DATA_PATH"
