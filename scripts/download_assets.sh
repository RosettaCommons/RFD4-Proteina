#!/usr/bin/env bash
# Download the configured released model and optional example/refolding assets.
set -euo pipefail

DATA_PATH="${1:-}"
if [ -z "$DATA_PATH" ] || [ "${DATA_PATH:0:1}" = "-" ]; then
    echo "Usage: $0 <DATA_PATH> [--with-test-dataset] [--skip-of3] [--skip-benchmarks] [--cuda 12|13]" >&2
    exit 1
fi
shift
WITH_TEST_DATASET=false
SKIP_OF3=false
SKIP_BENCHMARKS=false
CUDA_MAJOR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-test-dataset) WITH_TEST_DATASET=true ;;
        --skip-of3) SKIP_OF3=true ;;
        --skip-benchmarks) SKIP_BENCHMARKS=true ;;
        --cuda)
            if [[ $# -lt 2 || ( "$2" != 12 && "$2" != 13 ) ]]; then
                echo "--cuda must be 12 or 13" >&2; exit 2
            fi
            CUDA_MAJOR="$2"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/oracle_artifacts.sh"
INFERENCE_CONFIG="$SCRIPT_DIR/../configs/inference.yaml"
resolve_hf_release "$INFERENCE_CONFIG"
REPO="$HF_RELEASE_REPO"
REVISION="$HF_RELEASE_REVISION"

fetch() {
    hf download "$REPO" "$@" ${REVISION:+--revision "$REVISION"} --local-dir "$DATA_PATH"
}

echo ">>> Caching the released model from Hugging Face ($REPO; HF_HOME controls the cache location)..."
hf download "$REPO" config.json model.safetensors ${REVISION:+--revision "$REVISION"}

if [ "$SKIP_BENCHMARKS" = false ]; then
    BENCHMARKS_REPO="${RFD4_BENCHMARKS_REPO:-nvidia/RFD4-Proteina}"
    echo ">>> Downloading the validation benchmarks from Hugging Face ($BENCHMARKS_REPO, about 63 MB) into data/benchmarks/..."
    for benchmark in ae_pdb_temporal_holdout_2026_10_04 nanobody; do
        hf download "$BENCHMARKS_REPO" --repo-type dataset ${RFD4_BENCHMARKS_REVISION:+--revision "$RFD4_BENCHMARKS_REVISION"} \
            --include "benchmarks/$benchmark/*" --local-dir "$REPO_ROOT/data"
    done
    echo ">>> Caching the CDR torsion reference data for the antibody metric (Hugging Face cache)..."
    hf download "$BENCHMARKS_REPO" --repo-type dataset ${RFD4_BENCHMARKS_REVISION:+--revision "$RFD4_BENCHMARKS_REVISION"} \
        --include "metrics/cdr_torsion_data/*"
fi

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
    select_of3_runtime "${CUDA_MAJOR:-$(detect_oracle_cuda)}"
    RUNTIME_PATH="$DATA_PATH/$OF3_RUNTIME_FILE"
    download_oracle_runtime "$OF3_RUNTIME_FILE" "$OF3_RUNTIME_SHA256" "$DATA_PATH"
    hf download "$ORACLE_ARTIFACT_REPO" "${OF3_WEIGHTS_FILES[@]}" --revision "$ORACLE_ARTIFACT_REVISION" --local-dir "$DATA_PATH"
    mkdir -p "$DATA_PATH/openfold3/jax_cache"
    echo ">>> OpenFold3 ready. For direct generate.py/CLI use (the tutorial notebook finds these"
    echo "    automatically), export:"
    echo "    export AF3_IMAGE=\"$RUNTIME_PATH\""
    echo "    export AF3_MODEL_DIR=\"$DATA_PATH/openfold3/weights\""
    echo "    export AF3_JAX_CACHE_DIR=\"$DATA_PATH/openfold3/jax_cache\""
    echo "    ...and pass metrics/refolding_oracles=of3 as a Hydra override."
fi

echo ">>> Done. Now: export DATA_PATH=$DATA_PATH"
