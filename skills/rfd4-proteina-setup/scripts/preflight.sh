#!/usr/bin/env bash
# Preflight for RFProteína: verify the environment + checkpoints are ready before
# launching a generation/evaluation run. Prints a PASS/FAIL report; exits nonzero
# on any FAIL. Safe to run repeatedly.
#
#   bash skills/rfd4-proteina-setup/scripts/preflight.sh
#
# Auto-detects the repo root from this script's location, so it can be run from
# anywhere.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

fail=0
pass()  { printf '  PASS  %s\n' "$1"; }
warn()  { printf '  warn  %s\n' "$1"; }
bad()   { printf '  FAIL  %s\n' "$1"; fail=1; }

echo "RFProteína preflight (repo root: $REPO_ROOT)"
echo "-------------------------------------------------------------"

# 1. Python 3.12 + venv
if command -v python >/dev/null 2>&1; then
    pyver="$(python -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    if [ "$pyver" = "3.12" ]; then pass "python $pyver"; else warn "python is $pyver (expected 3.12 — is the venv active?)"; fi
else
    bad "python not found on PATH (activate the venv: source .venv/bin/activate)"
fi
if [ -n "${VIRTUAL_ENV:-}" ]; then pass "virtualenv active: $VIRTUAL_ENV"; else warn "no VIRTUAL_ENV set (run: source .venv/bin/activate)"; fi

# 2. atomworks submodule importable
if [ -f "lib/atomworks/pyproject.toml" ]; then pass "atomworks submodule present"; else bad "atomworks submodule missing (git submodule update --init lib/atomworks)"; fi
if python -c "import atomworks" >/dev/null 2>&1; then pass "atomworks importable"; else bad "cannot import atomworks (re-run setup.sh)"; fi

# 3. rfproteina + conditions registry importable
if python -c "import rfproteina, rfproteina.conditions" >/dev/null 2>&1; then pass "rfproteina + conditions importable"; else bad "cannot import rfproteina (activate venv / re-run setup.sh)"; fi

# 4. GPU
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1)"
    pass "GPU visible: ${gpu:-unknown}"
else
    bad "no GPU visible (nvidia-smi failed) — generation requires a CUDA GPU"
fi

# 5. Cluster env file
CLUSTER="${CLUSTER:-}"
if [ -z "$CLUSTER" ]; then
    warn "CLUSTER not set (export CLUSTER=local for a non-IPD machine)"
else
    if [ -f ".ipd/.env.$CLUSTER" ]; then pass "cluster env file: .ipd/.env.$CLUSTER"; else bad "missing .ipd/.env.$CLUSTER (create it; for local: printf 'DATA_PATH=%s\\n' \"\$DATA_PATH\" > .ipd/.env.local)"; fi
fi

# 6. DATA_PATH + checkpoints
DATA_PATH="${DATA_PATH:-}"
if [ -z "$DATA_PATH" ]; then
    bad "DATA_PATH not set (export DATA_PATH=... ; ./scripts/download_example_data.sh \"\$DATA_PATH\")"
else
    pass "DATA_PATH=$DATA_PATH"
    ldm_dir="$DATA_PATH/latent_checkpoints/ldm_checkpoints"
    ae_ckpt="$DATA_PATH/latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt"
    if compgen -G "$ldm_dir/*.ckpt" >/dev/null; then
        pass "denoiser checkpoint(s) found in $ldm_dir"
        printf '        (%s)\n' "$(cd "$ldm_dir" && ls -1 *.ckpt | tr '\n' ' ')"
    else
        bad "no denoiser .ckpt in $ldm_dir (./scripts/download_example_data.sh \"\$DATA_PATH\")"
    fi
    if [ -f "$ae_ckpt" ]; then pass "autoencoder checkpoint found"; else bad "missing autoencoder checkpoint: $ae_ckpt"; fi
fi

echo "-------------------------------------------------------------"
if [ "$fail" -eq 0 ]; then
    echo "PREFLIGHT PASSED"
    exit 0
else
    echo "PREFLIGHT FAILED — resolve the FAIL items above before running."
    exit 1
fi
