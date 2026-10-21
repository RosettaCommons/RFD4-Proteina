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
ask_choice cuda "Which CUDA major version does your driver support?" "$(detect_oracle_cuda)" 12 13

# DATA_PATH and any existing settings come from .env.local, like every other entry point.
source "$repo_root/scripts/load_local_env.sh"
DATA_PATH="${DATA_PATH:-$repo_root/data_path}"
if [[ "$DATA_PATH" != /* ]]; then DATA_PATH="$PWD/$DATA_PATH"; fi
env_file="$repo_root/.env.local"
if [[ ! -f "$env_file" && -f "$repo_root/.env" ]]; then env_file="$repo_root/.env"; fi
echo ">>> Using arch=$arch cuda=$cuda DATA_PATH=$DATA_PATH"

run() {
    if [[ "$dry_run" == true ]]; then echo "    [dry run] $*"; else "$@"; fi
}

# Double quotes preserve spaces for both Bash and Python's dotenv loader.
quote_env_path() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//\$/\\\$}"
    value="${value//\`/\\\`}"
    printf '"%s"' "$value"
}

block=()
notes=()
overrides=()

# ── AlphaFold 3 / OpenFold3 ────────────────────────────────────────────────
if [[ "$af3" == yes ]]; then
    of3_dir="$DATA_PATH/openfold3"
    run mkdir -p "$of3_dir/runtime" "$of3_dir/weights" "$of3_dir/jax_cache"
    select_of3_runtime "$cuda"
    if [[ "$arch" == amd64 ]]; then
        if ! command -v apptainer &>/dev/null; then
            notes+=("apptainer is not installed, so the OpenFold3 image was not downloaded. Install it from https://github.com/apptainer/apptainer/releases and re-run this script.")
        else
            download_oracle_runtime "$OF3_RUNTIME_FILE" "$OF3_RUNTIME_SHA256" "$DATA_PATH" "$dry_run"
            block+=("AF3_IMAGE=$(quote_env_path "$DATA_PATH/$OF3_RUNTIME_FILE")")
        fi
    else
        notes+=("No prebuilt AlphaFold 3 / OpenFold3 image is published for $arch. Build one from the runtime source (see docs/how-to/evaluate-with-metrics.md), then set AF3_IMAGE to the resulting .sif in $env_file.")
    fi
    block+=("AF3_JAX_CACHE_DIR=$(quote_env_path "$of3_dir/jax_cache")")
    if [[ "$of3_weights" == yes ]]; then
        run hf download "$ORACLE_ARTIFACT_REPO" "${OF3_WEIGHTS_FILES[@]}" --revision "$ORACLE_ARTIFACT_REVISION" --local-dir "$DATA_PATH"
        block+=("AF3_MODEL_DIR=$(quote_env_path "$of3_dir/weights")")
        overrides+=("metrics/refolding_oracles=of3")
    else
        notes+=("AlphaFold 3's own weights are gated by Google DeepMind. Set AF3_MODEL_DIR in $env_file to your copy, or re-run with --of3-weights to use the free OpenFold3 weights.")
        overrides+=("metrics/refolding_oracles=af3")
    fi
fi

# ── AlphaFold 2 ────────────────────────────────────────────────────────────
if [[ "$af2" == yes ]]; then
    af2_dir="$DATA_PATH/af2"
    run mkdir -p "$af2_dir/model_dir/params" "$af2_dir/jax_cache"
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
    if [[ "$arch" == amd64 ]] && command -v apptainer &>/dev/null; then
        select_af2_runtime "$cuda"
        download_oracle_runtime "$AF2_RUNTIME_FILE" "$AF2_RUNTIME_SHA256" "$DATA_PATH" "$dry_run"
        block+=("AF2_RUNTIME_BACKEND=apptainer" "AF2_IMAGE=$(quote_env_path "$DATA_PATH/$AF2_RUNTIME_FILE")")
    else
        notes+=("The prebuilt AlphaFold 2 images require amd64 and Apptainer. Install Apptainer and re-run setup, or configure a runtime following docs/installation.rst. Pallas requires GPU compute capability 8.0 or newer.")
    fi
    block+=("AF2_MODEL_DIR=$(quote_env_path "$af2_dir/model_dir")"
            "AF2_JAX_CACHE_DIR=$(quote_env_path "$af2_dir/jax_cache")"
            "AF2_RESIDENT_MODELS=model_1_ptm,model_1_multimer_v3")
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
    tmp="$(mktemp "$env_file.XXXXXX")"
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
