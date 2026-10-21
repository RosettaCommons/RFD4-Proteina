#!/usr/bin/env bash
# Published refolding-oracle artifacts shared by both setup entry points.
ORACLE_ARTIFACT_REPO="nvidia/RFD4-Proteina"
ORACLE_ARTIFACT_REVISION="598acab93d5d0aeacc4362f6e88dc20e7db85ef6"

OF3_WEIGHTS_FILES=("openfold3/weights/of3_ported_weights.bin.zst" "openfold3/weights/of3_ported_weights.provenance.json")
AF2_PARAMS_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

select_of3_runtime() {
    case "$1" in
        12) OF3_RUNTIME_SHA256="c2b17e5b9262614844c71ce1ddfa1a03ce3534b367b1e919ea41866027087c71" ;;
        13) OF3_RUNTIME_SHA256="7bbd35478ed5e95131efa7756d03a2e754f3508e603ad08bb8f8db0bd89484a1" ;;
        *) echo "Unsupported CUDA major: $1" >&2; return 2 ;;
    esac
    OF3_RUNTIME_FILE="openfold3/runtime/of3_af3_refolding_env-cuda$1.sif"
}

select_af2_runtime() {
    case "$1" in
        12) AF2_RUNTIME_SHA256="62b6531b87bafbbfb584ae2c341b9bab4591a733ffbc39abd2d6dd4b7ac214c6" ;;
        13) AF2_RUNTIME_SHA256="b5160d179a0f11608e176f8883c8e9c644ab5c6a6b92bf05c2ca5b7a5ecb8d3f" ;;
        *) echo "Unsupported CUDA major: $1" >&2; return 2 ;;
    esac
    AF2_RUNTIME_FILE="alphafold2/runtime/alphafold2-pallas-cuda$1-amd64.sif"
}

detect_oracle_cuda() {
    local major=""
    if command -v nvidia-smi &>/dev/null; then
        major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -n 1)"
    fi
    case "$major" in 12|13) echo "$major" ;; *) echo 13 ;; esac
}

download_oracle_runtime() {
    local file="$1" checksum="$2" data_path="$3" dry_run="${4:-false}"
    local target="$data_path/$file"
    if [[ "$dry_run" == true ]]; then
        echo "    [dry run] hf download $ORACLE_ARTIFACT_REPO $file --revision $ORACLE_ARTIFACT_REVISION -> $data_path (sha256 $checksum)"
    elif printf '%s  %s\n' "$checksum" "$target" | sha256sum --check --status 2>/dev/null; then
        echo ">>> have ${file##*/}, skipping"
    else
        mkdir -p "$(dirname "$target")"
        hf download "$ORACLE_ARTIFACT_REPO" "$file" --revision "$ORACLE_ARTIFACT_REVISION" --local-dir "$data_path"
        printf '%s  %s\n' "$checksum" "$target" | sha256sum --check
    fi
}

# Model and autoencoder revisions remain independent of refolding-runtime updates.
resolve_hf_release() {
    local config="$1"
    HF_RELEASE_REPO="$(sed -n 's/^hf_ckpt_path: *\([^ #]*\).*/\1/p' "$config")"
    HF_RELEASE_REVISION="$(sed -n 's/^hf_ckpt_revision: *\([^ #]*\).*/\1/p' "$config")"
    case "$HF_RELEASE_REPO" in ""|null|"~") HF_RELEASE_REPO="nvidia/RFD4-Proteina" ;; esac
    case "$HF_RELEASE_REVISION" in null|"~") HF_RELEASE_REVISION="" ;; esac
}
