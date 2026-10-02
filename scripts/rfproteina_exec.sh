#!/usr/bin/env bash
# Preserve private execution when available; otherwise use the public runtime.
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 inference|training <script.py | -m module | -c code> [arguments...]" >&2
    exit 2
fi
mode="$1"
shift
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$mode" in
    inference) ipd_wrapper="$repo_root/.ipd/shebang/latent_inference_exec.sh" ;;
    training) ipd_wrapper="$repo_root/.ipd/shebang/latent_exec.sh" ;;
    *) echo "Unknown execution mode: $mode" >&2; exit 2 ;;
esac
if [[ "${CLUSTER:-}" != local && -x "$ipd_wrapper" ]]; then
    exec "$ipd_wrapper" "$@"
fi
export CLUSTER="${CLUSTER:-local}"
source "$repo_root/scripts/load_local_env.sh" || exit 1
source "$repo_root/scripts/common.sh"
source_venv "$repo_root" >/dev/null || exit 1
setup_pythonpath "$repo_root" >/dev/null
python_command=(python)
if [[ -n "${DEBUG_PORT:-}" ]]; then
    python_command+=(-m debugpy --listen "$DEBUG_PORT" --wait-for-client)
fi
exec "${python_command[@]}" "$@"
