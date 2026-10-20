#!/usr/bin/env bash
# Optional refolding-oracle environments for reward-guided search and evaluation.
#
# Search scores designs with AlphaFold 2 (protein only) or AlphaFold 3 / OpenFold3 refolding. Those
# runtimes live outside the Python environment, so setup asks whether you want them, and for which
# CPU architecture and CUDA major. `make setup` runs this after the base install.
#
#   bash scripts/setup_search_envs.sh                      # ask every question
#   bash scripts/setup_search_envs.sh --af3 --of3-weights --arch amd64 --cuda 13
#
# Flags (each skips its question): --af2 | --no-af2, --af2-params | --no-af2-params,
# --af3 | --no-af3, --of3-weights | --no-of3-weights, --arch amd64|arm64, --cuda 12|13.
# --dry-run prints what would happen. Without a terminal the script asks nothing and installs nothing,
# unless RFD4_SETUP_PROMPT=1 forces reading answers from stdin.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$repo_root/scripts/oracle_artifacts.sh"
resolve_hf_release "$repo_root/configs/inference.yaml"

af2="" af2_params="" af3="" of3_weights="" arch="" cuda="" dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --af2) af2=yes ;;
        --no-af2) af2=no ;;
        --af2-params) af2_params=yes ;;
        --no-af2-params) af2_params=no ;;
        --af3) af3=yes ;;
        --no-af3) af3=no ;;
        --of3-weights) of3_weights=yes ;;
        --no-of3-weights) of3_weights=no ;;
        --arch) arch="${2:-}"; shift ;;
        --cuda) cuda="${2:-}"; shift ;;
        --dry-run) dry_run=true ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
case "$arch" in ""|amd64|arm64) ;; *) echo "--arch must be amd64 or arm64" >&2; exit 2 ;; esac
case "$cuda" in ""|12|13) ;; *) echo "--cuda must be 12 or 13" >&2; exit 2 ;; esac

interactive=false
if [[ -t 0 || "${RFD4_SETUP_PROMPT:-}" == 1 ]]; then interactive=true; fi

# ask <variable> <question> <default yes|no>: leaves an already set variable alone.
ask_yes_no() {
    local var="$1" question="$2" default="$3" hint reply
    if [[ -n "${!var}" ]]; then return; fi
    if [[ "$interactive" != true ]]; then printf -v "$var" '%s' no; return; fi
    if [[ "$default" == yes ]]; then hint="[Y/n]"; else hint="[y/N]"; fi
    read -r -p "    $question $hint " reply || reply=""
    reply="$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]')"
    case "$reply" in
        y|yes) printf -v "$var" '%s' yes ;;
        n|no) printf -v "$var" '%s' no ;;
        *) printf -v "$var" '%s' "$default" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo amd64 ;;
        aarch64|arm64) echo arm64 ;;
        *) echo amd64 ;;
    esac
}

detect_cuda() {
    local major=""
    if command -v nvidia-smi &>/dev/null; then
        major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -n 1)"
    fi
    if [[ "$major" == 12 || "$major" == 13 ]]; then echo "$major"; else echo 13; fi
}

# ask_choice <variable> <question> <detected> <option1> <option2>
ask_choice() {
    local var="$1" question="$2" detected="$3" first="$4" second="$5" reply default=1
    if [[ -n "${!var}" ]]; then return; fi
    if [[ "$detected" == "$second" ]]; then default=2; fi
    if [[ "$interactive" != true ]]; then printf -v "$var" '%s' "$detected"; return; fi
    echo "    $question"
    echo "      1) $first"
    echo "      2) $second"
    read -r -p "    Choice [$default, detected $detected]: " reply || reply=""
    case "${reply:-$default}" in
        1) printf -v "$var" '%s' "$first" ;;
        2) printf -v "$var" '%s' "$second" ;;
        *) printf -v "$var" '%s' "$detected" ;;
    esac
}

echo ""
echo ">>> Search and refolding environments (optional)"
echo "    Reward-guided search and the refolding metrics score designs with AlphaFold 2 (protein only)"
echo "    or AlphaFold 3 / OpenFold3. They run in their own containers, separate from .venv."
if [[ "$interactive" != true && -z "$af2$af3" ]]; then
    echo "    No terminal attached, so nothing was requested. Re-run with flags, for example:"
    echo "      bash scripts/setup_search_envs.sh --af3 --of3-weights --arch amd64 --cuda 13"
    exit 0
fi

ask_yes_no af2 "Set up an AlphaFold 2 environment for search?" no
ask_yes_no af3 "Set up an AlphaFold 3 / OpenFold3 environment for search?" no
if [[ "$af3" == yes ]]; then
    ask_yes_no of3_weights "Download the OpenFold3 weights (Apache-2.0, about 1.4 GB)?" yes
fi
if [[ "$af2" == yes ]]; then
    ask_yes_no af2_params "Download the AlphaFold 2 parameters (CC BY 4.0, about 5.3 GB)?" yes
fi
if [[ "$af2" != yes && "$af3" != yes ]]; then
    echo ">>> No search environments requested."
    exit 0
fi

ask_choice arch "Which CPU architecture will run them?" "$(detect_arch)" amd64 arm64
ask_choice cuda "Which CUDA major version does your driver support?" "$(detect_cuda)" 12 13

# DATA_PATH and any existing settings come from .env.local, like every other entry point.
source "$repo_root/scripts/load_local_env.sh"
DATA_PATH="${DATA_PATH:-$repo_root/data_path}"
env_file="$repo_root/.env.local"
if [[ ! -f "$env_file" && -f "$repo_root/.env" ]]; then env_file="$repo_root/.env"; fi
echo ">>> Using arch=$arch cuda=$cuda DATA_PATH=$DATA_PATH"

run() {
    if [[ "$dry_run" == true ]]; then echo "    [dry run] $*"; else "$@"; fi
}

block=()
notes=()
overrides=()

# ── AlphaFold 3 / OpenFold3 ────────────────────────────────────────────────
if [[ "$af3" == yes ]]; then
    of3_dir="$DATA_PATH/openfold3"
    run mkdir -p "$of3_dir/runtime" "$of3_dir/weights" "$of3_dir/jax_cache"
    runtime_path="$of3_dir/runtime/$OF3_RUNTIME_NAME"
    if [[ " $OF3_RUNTIME_ARCHES " == *" $arch "* ]]; then
        if ! command -v apptainer &>/dev/null; then
            notes+=("apptainer is not installed, so the OpenFold3 image was not downloaded. Install it from https://github.com/apptainer/apptainer/releases and re-run this script.")
        elif [[ "$dry_run" == true ]]; then
            echo "    [dry run] hf download $HF_RELEASE_REPO $OF3_RUNTIME_FILE -> $DATA_PATH (sha256 $OF3_RUNTIME_SHA256)"
        elif printf '%s  %s\n' "$OF3_RUNTIME_SHA256" "$runtime_path" | sha256sum --check --status 2>/dev/null; then
            echo ">>> have $OF3_RUNTIME_NAME, skipping"
        else
            echo ">>> Downloading the AlphaFold 3 / OpenFold3 runtime image (about 4 GB)..."
            if ! command -v hf &>/dev/null; then
                notes+=("The hf command is not on PATH, so the OpenFold3 image was not downloaded. Activate .venv (source .venv/bin/activate) and re-run this script.")
            else
                hf download "$HF_RELEASE_REPO" "$OF3_RUNTIME_FILE" ${HF_RELEASE_REVISION:+--revision "$HF_RELEASE_REVISION"} --local-dir "$DATA_PATH"
                printf '%s  %s\n' "$OF3_RUNTIME_SHA256" "$runtime_path" | sha256sum --check
            fi
        fi
        block+=("AF3_IMAGE=\${DATA_PATH}/openfold3/runtime/$OF3_RUNTIME_NAME")
    else
        notes+=("No prebuilt AlphaFold 3 / OpenFold3 image is published for $arch. Build one from the runtime source (see docs/how-to/evaluate-with-metrics.md), then set AF3_IMAGE to the resulting .sif in $env_file.")
    fi
    block+=("AF3_JAX_CACHE_DIR=\${DATA_PATH}/openfold3/jax_cache")
    if [[ "$of3_weights" == yes ]]; then
        hf_repo="$HF_RELEASE_REPO"
        hf_revision="$HF_RELEASE_REVISION"
        if [[ "$dry_run" == true ]]; then
            echo "    [dry run] hf download $hf_repo ${OF3_WEIGHTS_FILES[*]} --local-dir $DATA_PATH"
        elif ! command -v hf &>/dev/null; then
            notes+=("The hf command is not on PATH, so the OpenFold3 weights were not downloaded. Activate .venv (source .venv/bin/activate) and re-run this script.")
        else
            echo ">>> Downloading the OpenFold3 weights..."
            hf download "$hf_repo" "${OF3_WEIGHTS_FILES[@]}" ${hf_revision:+--revision "$hf_revision"} --local-dir "$DATA_PATH"
        fi
        block+=("AF3_MODEL_DIR=\${DATA_PATH}/openfold3/weights")
        overrides+=("metrics/refolding_oracles=of3")
    else
        notes+=("AlphaFold 3's own weights are gated by Google DeepMind. Set AF3_MODEL_DIR in $env_file to your copy, or re-run with --of3-weights to use the free OpenFold3 weights.")
        overrides+=("metrics/refolding_oracles=af3")
    fi
fi

# ── AlphaFold 2 ────────────────────────────────────────────────────────────
if [[ "$af2" == yes ]]; then
    af2_dir="$DATA_PATH/af2"
    run mkdir -p "$af2_dir/model_dir/params" "$af2_dir/jax_cache" "$af2_dir/runtime"
    image_name="alphafold2-pallas-cuda${cuda}-${arch}"
    if [[ "$af2_params" == yes ]]; then
        params_dir="$af2_dir/model_dir/params"
        if [[ "$dry_run" == true ]]; then
            echo "    [dry run] download $AF2_PARAMS_URL -> $params_dir"
        elif [[ -f "$params_dir/params_model_1_ptm.npz" && -f "$params_dir/params_model_1_multimer_v3.npz" ]]; then
            echo ">>> have the AlphaFold 2 parameters, skipping"
        else
            echo ">>> Downloading the AlphaFold 2 parameters (about 5.3 GB)..."
            curl --fail --location --retry 3 --output "$af2_dir/alphafold_params.tar.partial" "$AF2_PARAMS_URL"
            tar -xf "$af2_dir/alphafold_params.tar.partial" -C "$params_dir"
            rm -f "$af2_dir/alphafold_params.tar.partial"
        fi
    else
        notes+=("Put the AlphaFold 2 parameters under $af2_dir/model_dir/params (model_1_ptm and model_1_multimer_v3), or re-run with --af2-params.")
    fi
    if command -v docker &>/dev/null; then
        backend=docker
        image_ref="$image_name:latest"
    elif command -v apptainer &>/dev/null; then
        backend=apptainer
        image_ref="\${DATA_PATH}/af2/runtime/$image_name.sif"
    else
        backend=""
        image_ref="\${DATA_PATH}/af2/runtime/$image_name.sif"
        notes+=("Neither docker nor apptainer is installed, so the AlphaFold 2 runtime cannot run yet. Install one of them.")
    fi
    if [[ -n "$backend" ]]; then block+=("AF2_RUNTIME_BACKEND=$backend"); fi
    block+=("AF2_IMAGE=$image_ref"
            "AF2_MODEL_DIR=\${DATA_PATH}/af2/model_dir"
            "AF2_JAX_CACHE_DIR=\${DATA_PATH}/af2/jax_cache"
            "AF2_RESIDENT_MODELS=model_1_ptm,model_1_multimer_v3")
    notes+=("No prebuilt AlphaFold 2 image is published. Build $image_name for linux/$arch with jax[cuda$cuda] in the pip install, then point AF2_IMAGE at it. Pallas kernels need a GPU with compute capability 8.0 or newer.")
    overrides+=("metrics/refolding_oracles=af2")
fi

# ── Record the settings in .env.local ──────────────────────────────────────
marker_start="# >>> rfd4 search environments >>>"
marker_end="# <<< rfd4 search environments <<<"
if [[ "$dry_run" == true ]]; then
    echo "    [dry run] would write to $env_file:"
    printf '      %s\n' "${block[@]}"
else
    touch "$env_file"
    tmp="$(mktemp)"
    awk -v start="$marker_start" -v end="$marker_end" \
        '$0 == start { skip = 1 } !skip { print } $0 == end { skip = 0 }' "$env_file" > "$tmp"
    {
        cat "$tmp"
        printf '\n%s\n' "$marker_start"
        printf '# Written by scripts/setup_search_envs.sh (arch=%s, cuda=%s). Re-run it to change these.\n' "$arch" "$cuda"
        printf '%s\n' "${block[@]}"
        printf '%s\n' "$marker_end"
    } > "$env_file"
    rm -f "$tmp"
    echo ">>> Recorded the search environment settings in $env_file"
fi

echo ""
if [[ ${#notes[@]} -gt 0 ]]; then
    echo ">>> Still to do:"
    printf '    - %s\n' "${notes[@]}"
fi
if [[ ${#overrides[@]} -gt 0 ]]; then
    echo ">>> Select an oracle with a Hydra override, for example:"
    printf '      %s\n' "${overrides[@]}"
fi
