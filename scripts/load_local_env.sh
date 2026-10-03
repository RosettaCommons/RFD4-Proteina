#!/usr/bin/env bash
# Source a public runtime profile while preserving already exported shell settings.
rfproteina_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$rfproteina_repo_root}"
rfproteina_env_file="$rfproteina_repo_root/.env.${CLUSTER:-local}"
if [[ "${CLUSTER:-local}" == local && ! -f "$rfproteina_env_file" ]]; then
    rfproteina_env_file="$rfproteina_repo_root/.env"
fi
if [[ "${CLUSTER:-local}" != local && ! -f "$rfproteina_env_file" ]]; then
    echo "Runtime profile unavailable: configure $rfproteina_env_file or select CLUSTER=local." >&2
    return 1
fi
if [[ -f "$rfproteina_env_file" ]]; then
    rfproteina_exported_env="$(export -p | sed 's/^declare -x /export /')"
    rfproteina_shell_options="$-"
    set -a
    source "$rfproteina_env_file"
    if [[ "$rfproteina_shell_options" != *a* ]]; then set +a; fi
    eval "$rfproteina_exported_env"
    unset rfproteina_exported_env rfproteina_shell_options
fi
