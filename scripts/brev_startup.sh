#!/bin/bash
set -Eeuo pipefail
trap 'printf "Startup failed at line %s.\n" "$LINENO" >&2' ERR

fail() { printf '%s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'Requires Linux x86_64.'
command -v nvidia-smi >/dev/null || fail 'An NVIDIA GPU driver is required.'
driver_info="$(nvidia-smi)"
cuda_version="$(sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' <<< "$driver_info" | head -n 1)"
[[ "$cuda_version" =~ ^[0-9]+\.[0-9]+$ ]] || fail 'Cannot determine driver CUDA support.'
cuda_major="${cuda_version%%.*}"
cuda_minor="${cuda_version#*.}"
if (( cuda_major >= 13 )); then
    cuda_major=13
elif (( cuda_major == 12 && cuda_minor >= 6 )); then
    cuda_major=12
else
    fail 'Requires a driver supporting CUDA 12.6 or newer.'
fi

admin=()
if (( EUID != 0 )); then
    sudo -n true || fail 'Passwordless sudo is required to install system packages.'
    admin=(sudo -n)
fi
command -v apt-get >/dev/null || fail 'Requires an Ubuntu/Debian host with apt-get.'
for source_file in /etc/apt/sources.list.d/apptainer-ubuntu-ppa-*.{list,sources}; do
    if [[ -f "$source_file" ]] && grep -Eq 'ppa\.launchpad(content)?\.net/apptainer/ppa' "$source_file"; then
        "${admin[@]}" mv -- "$source_file" "$source_file.disabled"
    fi
done
apt_run() {
    "${admin[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Languages=none -o Acquire::Retries=5 "$@"
}
apt_run update
apt_run install -y --no-install-recommends \
    ca-certificates curl git build-essential libxrender1 libx11-6 libxext6

if ! command -v apptainer >/dev/null; then
    package_dir="$(mktemp -d)"
    chmod 755 "$package_dir"
    package="$package_dir/apptainer_1.5.4_amd64.deb"
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
        https://github.com/apptainer/apptainer/releases/download/v1.5.4/apptainer_1.5.4_amd64.deb \
        -o "$package"
    printf '%s  %s\n' 281cf9bd68f16b6f7c34a3a94d77668bc327b9c09db0412ce7890162734a97f7 "$package" | sha256sum -c -
    chmod 644 "$package"
    apt_run install -y "$package"
    rm -f -- "$package"
    rmdir -- "$package_dir"
fi
apptainer --version

REPO_DIR="${REPO_DIR:-$HOME/RFD4-Proteina}"
repo_url='https://github.com/RosettaCommons/RFD4-Proteina.git'
if [[ ! -e "$REPO_DIR" ]]; then
    git clone --depth 1 "$repo_url" "$REPO_DIR"
else
    [[ -d "$REPO_DIR/.git" ]] || fail "$REPO_DIR already exists and is not a Git checkout."
    origin="$(git -C "$REPO_DIR" remote get-url origin)"
    [[ "${origin%.git}" == "${repo_url%.git}" ]] || fail "Use a fresh directory for the public repository: $REPO_DIR"
fi
cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
printf 'Using repository commit %s (CUDA %s).\n' "$(git rev-parse HEAD)" "$cuda_major"

export CLUSTER=local PROJECT_ROOT="$REPO_DIR"
export PATH="$REPO_DIR/.venv/bin:$HOME/.local/bin:$PATH"
if [[ ! -e .env.local && ! -e .env ]]; then
    (umask 077; set -C; cat .env.local.example > .env.local)
fi
source scripts/load_local_env.sh
export DATA_PATH="${DATA_PATH:-$REPO_DIR/data_path}"
mkdir -p "$DATA_PATH"

if [[ -x .venv/bin/python ]]; then
    installed_cuda="$(.venv/bin/python -c 'import importlib.metadata as m; print(m.version("torch").split("+cu")[-1][:2])' 2>/dev/null || true)"
    if [[ "$installed_cuda" == 12 || "$installed_cuda" == 13 ]]; then
        [[ "$installed_cuda" == "$cuda_major" ]] || fail 'Existing .venv uses a different CUDA profile; use a fresh REPO_DIR.'
    fi
fi

if [[ "$cuda_major" == 13 ]]; then
    bash setup.sh --native --skip-search-envs
else
    if ! command -v uv >/dev/null; then
        curl -fsSL --retry 3 https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
    fi
    if [[ ! -d .venv ]]; then
        uv python install '>=3.12,<=3.12.10'
        uv venv --python '>=3.12,<=3.12.10'
    fi
    uv pip install --python .venv/bin/python 'uv==0.13.4'
    .venv/bin/uv pip install --python .venv/bin/python --no-sources --torch-backend cu126 \
        --only-binary tmol,tmol-ops-cu12 -e '.[all-cuda12]'
    bash scripts/download_assets.sh "$DATA_PATH" --skip-of3
fi

source .venv/bin/activate
uv pip install --python .venv/bin/python 'ipykernel>=6.31.0,<7'
uv pip check --python .venv/bin/python
python -m ipykernel install --user --name rfd4-proteina --display-name 'RFD4-Proteina (.venv)'
bash scripts/setup_search_envs.sh --no-af2 --af3 --of3-weights --arch amd64 --cuda "$cuda_major"
bash skills/rfd4-proteina-setup/scripts/preflight.sh
printf 'Ready. Activate with: source %q/.venv/bin/activate\n' "$REPO_DIR"
