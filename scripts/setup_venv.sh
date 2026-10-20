#!/usr/bin/env bash
# Install the native model runtime or development tools.

set -e

runtime=false
case "${1:-}" in
    --runtime) runtime=true ;;
    "") ;;
    *) echo "Usage: bash scripts/setup_venv.sh [--runtime]" >&2; exit 2 ;;
esac
if [[ "$runtime" == true && ( "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ) ]]; then
    echo "The native CUDA 13 dependency profile currently supports Linux x86_64 only." >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ── 1. Install uv ──────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
    # Add uv to PATH for the remainder of this script
    source "$HOME/.local/bin/env"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# ── 3. Create virtualenv ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo ">>> Creating .venv with Python (>=3.12,<=3.12.10)..."
    uv python install '>=3.12,<=3.12.10'
    uv venv --python '>=3.12,<=3.12.10'
else
    echo ">>> .venv already exists — skipping creation."
fi

source .venv/bin/activate


# ── 5. Custom metrics scaffolding ──────────────────────────────────────────
CUSTOM_METRICS_CONFIG="configs/metrics/custom_metrics/all_custom_metrics.yaml"
if [ ! -f "$CUSTOM_METRICS_CONFIG" ]; then
    mkdir -p "$(dirname "$CUSTOM_METRICS_CONFIG")"
    cat > "$CUSTOM_METRICS_CONFIG" << 'EOF'
# @package metrics
#
# Git-ignored: register your own custom metrics here without touching tracked
# files. See docs/how-to/advanced/custom-metrics.md for the full guide.
#
# inline_metrics:
#   my_metric:
#     _target_: rfproteina.metrics.custom_metrics.my_module.MyMetric
#
# refolding_metrics:
#   my_refolding_metric:
#     _target_: rfproteina.metrics.custom_metrics.my_module.MyRefoldingMetric
EOF
    echo ">>> Created custom metrics config at $CUSTOM_METRICS_CONFIG."
else
    echo ">>> $CUSTOM_METRICS_CONFIG already exists — skipping."
fi

# Create the (git-ignored) Python package dir so it's ready for custom metric modules.
CUSTOM_METRICS_PKG="src/rfproteina/metrics/custom_metrics"
mkdir -p "$CUSTOM_METRICS_PKG"

# Same for custom sampling transforms. See docs/how-to/run-the-model.md#sampling-transforms.
CUSTOM_SAMPLING_TRANSFORMS_PKG="src/rfproteina/transforms/sampling"
mkdir -p "$CUSTOM_SAMPLING_TRANSFORMS_PKG"

# ── 6. Install dependencies ────────────────────────────────────────────────
echo ">>> Installing dependencies..."
build_torch_constraint=$(python -c "import tomllib; print(next(d for d in tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies'] if d.startswith('torch==')))")
if [[ "$runtime" == true ]]; then
    uv pip install --build-constraint <(printf '%s\n' "$build_torch_constraint") --torch-backend cu132 -e ".[metrics-cuda13]"
    uv pip check
    echo ">>> Native CUDA 13 environment ready: source $repo_root/.venv/bin/activate"
    exit 0
fi
uv pip install --build-constraint <(printf '%s\n' "$build_torch_constraint") --torch-backend cu132 -e ".[dev,metrics-cuda13]"
uv pip check

# ── Register Jupyter kernel (non-fatal) ──────────────────────────────────────
echo ">>> Registering Jupyter kernel..."
if python -m ipykernel install --user --name rfd4-proteina --display-name "RFD4-Proteina (.venv)"; then
    echo ">>> Registered kernel \"RFD4-Proteina (.venv)\" — pick it from Jupyter's kernel picker."
else
    echo ">>> WARNING: could not register the Jupyter kernel automatically. Register it manually:"
    echo "    python -m ipykernel install --user --name rfd4-proteina --display-name \"RFD4-Proteina (.venv)\""
fi

echo ">>> Development environment ready: source $repo_root/.venv/bin/activate"
