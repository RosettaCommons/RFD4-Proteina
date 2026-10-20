#!/usr/bin/env bash
# Download the configured released model and optional example/refolding assets.
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_CONFIG="$SCRIPT_DIR/../configs/inference.yaml"
REPO="$(sed -n 's/^hf_ckpt_path: *\([^ #]*\).*/\1/p' "$INFERENCE_CONFIG")"
REVISION="$(sed -n 's/^hf_ckpt_revision: *\([^ #]*\).*/\1/p' "$INFERENCE_CONFIG")"
ASSET_REPO="$REPO"
BASE="${RFPROTEINA_DATA_URL:-https://YOUR_DATASET_STORE}"

fetch() {
    hf download "$ASSET_REPO" "$1" --revision "$REVISION" --local-dir "$DATA_PATH"
}

fetch_native() {
    local relative_path="$1"
    local output="$DATA_PATH/$relative_path"
    if [ -f "$output" ]; then
        echo ">>> have $relative_path, skipping"
        return
    fi
    mkdir -p "$(dirname "$output")"
    curl --fail --location --retry 3 --output "$output.partial" "$BASE/$relative_path"
    mv "$output.partial" "$output"
}

# TODO: Once the unified release is on HF, revisit the native fallback after updating configs/inference.yaml.
case "$REPO" in
    ""|null|"~")
        ASSET_REPO="nvidia/RFD4-Proteina"
        DENOISER_CKPT="$(sed -n 's/^ckpt_name:[[:space:]]*\([^ #]*\).*/\1/p' "$INFERENCE_CONFIG")"
        AE_CKPT="$(sed -n 's/^autoencoder_ckpt_path:.*\/\([^ #]*\).*/\1/p' "$INFERENCE_CONFIG")"
        echo ">>> Downloading released checkpoints into DATA_PATH=$DATA_PATH..."
        fetch_native "flow_checkpoints/$DENOISER_CKPT"
        fetch_native "ae_checkpoints/$AE_CKPT"
        ;;
    *)
        echo ">>> Caching the released model (HF_HOME controls the cache location)..."
        hf download "$REPO" config.json model.safetensors --revision "$REVISION"
        ;;
esac

if [ "$WITH_TEST_DATASET" = true ]; then
    echo ">>> test_dataset..."
    mkdir -p "$DATA_PATH/test_dataset"
    fetch "test_dataset/test_dataset.tar.gz"
    tar -xzf "$DATA_PATH/test_dataset/test_dataset.tar.gz" -C "$DATA_PATH/test_dataset"
fi

# A runtime profile may already provide a compatible AF3 image.
if [ -n "${AF3_IMAGE:-}" ]; then
    SKIP_OF3=true
fi
if [ "$SKIP_OF3" = true ]; then
    echo ">>> Skipping OpenFold3 download (oracle image configured or --skip-of3)."
elif ! command -v apptainer &>/dev/null; then
    echo ">>> apptainer not found — skipping OpenFold3 download (it can't run without it)."
    echo "    Install apptainer, then re-run this script to fetch it."
else
    echo ">>> Downloading OpenFold3 (free, Apache-licensed refolding oracle, ~5 GB)..."
    RUNTIME_NAME="af3-open-8d36011.sif"
    RUNTIME_SHA256="100679cc5bb19d0232050b64953bdb1378ec056614bada4ffd124d18a0eca32c"
    RUNTIME_PATH="$DATA_PATH/openfold3/runtime/$RUNTIME_NAME"
    RUNTIME_URL="https://YOUR_DATASET_STORE/af3/apptainer_images/$RUNTIME_NAME"
    mkdir -p "$(dirname "$RUNTIME_PATH")"
    if ! printf '%s  %s\n' "$RUNTIME_SHA256" "$RUNTIME_PATH" | sha256sum --check --status 2>/dev/null; then
        curl --fail --location --retry 3 --output "$RUNTIME_PATH.partial" "$RUNTIME_URL"
        printf '%s  %s\n' "$RUNTIME_SHA256" "$RUNTIME_PATH.partial" | sha256sum --check
        mv "$RUNTIME_PATH.partial" "$RUNTIME_PATH"
    fi
    fetch "openfold3/weights/of3_ported_weights.bin.zst"
    fetch "openfold3/weights/of3_ported_weights.provenance.json"
    echo ">>> OpenFold3 ready. For direct generate.py/CLI use (the tutorial notebook finds these"
    echo "    automatically), export:"
    echo "    export AF3_IMAGE=\"$RUNTIME_PATH\""
    echo "    export AF3_MODEL_DIR=\"$DATA_PATH/openfold3/weights\""
    echo "    export AF3_JAX_CACHE_DIR=\"$DATA_PATH/openfold3/jax_cache\""
    echo "    ...and pass metrics/refolding_oracles=of3 as a Hydra override."
fi

echo ">>> Done. Now: export DATA_PATH=$DATA_PATH"
