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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Opportunistic fast path: S3 multipart GET via boto3 (scripts/s3_fetch.py); falls back to curl
# if boto3 isn't installed. No aws-cli needed -- boto3 is a normal Python dep (`.[dev]`).
_BASE_HOST="${BASE#https://}"
S3_BUCKET="${_BASE_HOST%%.*}"
S3_ENDPOINT="https://${_BASE_HOST#*.}"
HAVE_BOTO3=false
if command -v python3 &>/dev/null && python3 -c "import boto3" 2>/dev/null; then
    HAVE_BOTO3=true
fi

# Download to <dest-dir>/<basename>, skipping if already present; a partial download lands in a
# .tmp file so an interrupted run re-fetches instead of leaving a corrupt checkpoint.
fetch() {  # fetch <url> <dest-dir>
    mkdir -p "$2"
    local out="$2/$(basename "$1")"
    if [ -f "$out" ]; then echo ">>> have $(basename "$1"), skipping"; return; fi
    if [ "$HAVE_BOTO3" = true ] && \
            python3 "$SCRIPT_DIR/s3_fetch.py" "$S3_BUCKET" "${1#"$BASE"/}" "$out.tmp" "$S3_ENDPOINT"; then
        mv "$out.tmp" "$out"
    else
        curl -fL -o "$out.tmp" "$1" && mv "$out.tmp" "$out"
    fi
}

echo ">>> Downloading into DATA_PATH=$DATA_PATH"
# Read the release checkpoint names from the inference defaults.
INFERENCE_CONFIG="$SCRIPT_DIR/../configs/inference.yaml"
DENOISER_CKPT="$(sed -n 's/^ckpt_name:[[:space:]]*//p' "$INFERENCE_CONFIG")"
AE_CKPT="$(sed -n 's/^autoencoder_ckpt_path:.*\///p' "$INFERENCE_CONFIG")"
fetch "$BASE/flow_checkpoints/$DENOISER_CKPT" \
      "$DATA_PATH/flow_checkpoints"
fetch "$BASE/ae_checkpoints/$AE_CKPT" \
      "$DATA_PATH/ae_checkpoints"

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
