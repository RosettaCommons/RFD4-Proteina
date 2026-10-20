#!/usr/bin/env bash
# Install the native Python environment, or retain the existing IPD setup.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
if [[ $# == 0 && -f "$repo_root/.ipd/setup.sh" ]]; then
    exec bash "$repo_root/.ipd/setup.sh"
fi
if [[ "${1:-}" == --venv && $# == 1 ]]; then
    exec bash "$repo_root/scripts/setup_venv.sh"
fi
download_weights=true
for arg in "$@"; do
    case "$arg" in
        --native) ;;
        --skip-weights) download_weights=false ;;
        -h|--help)
            echo "Usage: bash setup.sh [--native] [--skip-weights] | --venv"
            echo "Install a native .venv and the configured released model."
            echo "Use --skip-weights for an environment-only install."
            echo "Use --native to select native setup in an IPD checkout; --venv adds development tools."
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
if [[ ! -e .env.local && ! -e .env ]]; then
    (umask 077; set -C; cat .env.local.example > .env.local)
    echo "Created .env.local; edit it for your data paths and optional tools."
else
    echo "Keeping existing .env.local/.env settings."
fi
bash "$repo_root/scripts/setup_venv.sh" --runtime
if [[ "$download_weights" == true ]]; then
    source "$repo_root/scripts/load_local_env.sh"
    if [[ -z "${DATA_PATH:-}" ]]; then
        printf '\nDATA_PATH=${PROJECT_ROOT}/data_path\n' >> "$rfproteina_env_file"
        export DATA_PATH="$PROJECT_ROOT/data_path"
    fi
    PATH="$repo_root/.venv/bin:$PATH" bash "$repo_root/scripts/download_assets.sh" "$DATA_PATH" --skip-of3
    echo 'Configured released model is ready.'
fi
printf 'Next: source %q/.venv/bin/activate\n' "$repo_root"
if [[ "$download_weights" == true ]]; then
    echo 'Run: rfd4-generate inputs=data/benchmarks/monomer subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke'
else
    echo 'Weights skipped; download them before generation. See docs/installation.rst.'
fi
