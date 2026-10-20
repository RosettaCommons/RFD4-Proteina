#!/usr/bin/env bash
# Published refolding-oracle artifacts, shared by download_assets.sh and setup_search_envs.sh.
# Source this file; it only defines variables.

# Everything except the AlphaFold 2 parameters is served from the Hugging Face model repository named by
# configs/inference.yaml (hf_ckpt_path at hf_ckpt_revision), under the paths below.

# OpenFold3 runtime image (Apptainer). Only an amd64 build is published.
OF3_RUNTIME_NAME="af3-open-8d36011.sif"
OF3_RUNTIME_SHA256="100679cc5bb19d0232050b64953bdb1378ec056614bada4ffd124d18a0eca32c"
OF3_RUNTIME_FILE="openfold3/runtime/$OF3_RUNTIME_NAME"
OF3_RUNTIME_ARCHES="amd64"

# OpenFold3 weights converted to an AF3 model directory (Apache-2.0, about 1.4 GB).
OF3_WEIGHTS_FILES=("openfold3/weights/of3_ported_weights.bin.zst" "openfold3/weights/of3_ported_weights.provenance.json")

# AlphaFold 2 parameters published by DeepMind (CC BY 4.0, about 5.3 GB). The oracle uses
# model_1_ptm and model_1_multimer_v3.
AF2_PARAMS_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

# Set HF_RELEASE_REPO and HF_RELEASE_REVISION from configs/inference.yaml (the repository defaults to
# nvidia/RFD4-Proteina when the config selects a native checkpoint).
resolve_hf_release() {
    local config="$1"
    HF_RELEASE_REPO="$(sed -n 's/^hf_ckpt_path: *\([^ #]*\).*/\1/p' "$config")"
    HF_RELEASE_REVISION="$(sed -n 's/^hf_ckpt_revision: *\([^ #]*\).*/\1/p' "$config")"
    case "$HF_RELEASE_REPO" in ""|null|"~") HF_RELEASE_REPO="nvidia/RFD4-Proteina" ;; esac
    case "$HF_RELEASE_REVISION" in null|"~") HF_RELEASE_REVISION="" ;; esac
}
