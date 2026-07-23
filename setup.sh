#!/usr/bin/env bash
# =============================================================================
# setup.sh — First-time setup for rfproteina on a new Linux machine.
#
# Can be run from any directory — it auto-detects the repo root:
#   bash /path/to/rfproteina/setup.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── 1. Install uv ──────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the remainder of this script
    source "$HOME/.local/bin/env"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# ── 2. Atomworks submodule ─────────────────────────────────────────────────
if [ ! -f "lib/atomworks/pyproject.toml" ]; then
    echo ">>> Initialising atomworks submodule..."
    git submodule update --init lib/atomworks
else
    echo ">>> atomworks submodule already present."
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
# files. See docs/guides/custom_metrics.md for the full guide.
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

# ── 6. Install dependencies ────────────────────────────────────────────────
echo ">>> Installing dependencies..."
# AtomWorks is intentionally omitted from RFProteina's pyproject and installed
# explicitly from the local submodule.
uv pip install -e "lib/atomworks[dev,ml,openbabel,docs]" -e ".[dev]"

# ── 7. Shell rc file ───────────────────────────────────────────────────────
# Determine which rc file to edit based on the user's login shell.
if [[ "$(basename "$SHELL")" == "zsh" ]]; then
    RC_FILE="$HOME/.zshrc"
else
    RC_FILE="$HOME/.bashrc"
fi

MARKER="# >>> rfproteina cluster settings >>>"
if grep -q "$MARKER" "$RC_FILE" 2>/dev/null; then
    echo ">>> Cluster settings already present in $RC_FILE — skipping."
else
    echo ""
    echo ">>> Which cluster will you submit jobs to?"
    echo "    1) digs       (CONTAINER_TYPE=apptainer)   — IPD / DIGS"
    echo "    2) ord        (CONTAINER_TYPE=docker)       — OCI / ORD"
    echo "    3) gb200      (CONTAINER_TYPE=docker)       — GB200"
    echo "    4) coreweave  (CONTAINER_TYPE=apptainer)    — CoreWeave"
    echo "    5) other      (I will set these manually)"
    read -rp "    Choice [1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1) CLUSTER_VAL=digs;      CT_VAL=apptainer ;;
        2) CLUSTER_VAL=ord;       CT_VAL=docker ;;
        3) CLUSTER_VAL=gb200;     CT_VAL=docker ;;
        4) CLUSTER_VAL=coreweave; CT_VAL=apptainer ;;
        *) CLUSTER_VAL="";        CT_VAL="" ;;
    esac

    if [ -n "$CLUSTER_VAL" ]; then
        cat >> "$RC_FILE" << EOF

$MARKER
# Cluster identity for rfproteina SLURM launch scripts.
# CLUSTER selects which scripts/.user_info_<cluster> config is loaded when
# submitting jobs. Update this if you switch to a different cluster.
export CLUSTER=$CLUSTER_VAL
# CONTAINER_TYPE is 'apptainer' (DIGS/CoreWeave) or 'docker' (OCI/ORD/GB200).
export CONTAINER_TYPE=$CT_VAL
# <<< rfproteina cluster settings <<<
EOF
        echo ">>> Added cluster settings to $RC_FILE."
    else
        echo ">>> Skipping rc edits — set CLUSTER and CONTAINER_TYPE manually in $RC_FILE."
    fi
fi

# ── 8. User info file ──────────────────────────────────────────────────────
if [ -n "$CLUSTER_VAL" ]; then
    USER_INFO_DEST="scripts/.user_info_${CLUSTER_VAL}"
    USER_INFO_EXAMPLE="scripts/.user_info_${CLUSTER_VAL}_example"
    if [ ! -f "$USER_INFO_DEST" ] && [ -f "$USER_INFO_EXAMPLE" ]; then
        cp "$USER_INFO_EXAMPLE" "$USER_INFO_DEST"
        echo ">>> Created $USER_INFO_DEST from example template."
        echo "    For most inference workflows no changes are needed."
        echo "    Edit it to customise default slurm settings."
    elif [ -f "$USER_INFO_DEST" ]; then
        echo ">>> $USER_INFO_DEST already exists — skipping."
    else
        echo ">>> No example file found for $CLUSTER_VAL — create $USER_INFO_DEST manually."
    fi
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " Next steps:"
echo "   1. Reload your shell:  source $RC_FILE"
echo "   2. Activate the venv:  source $PROJECT_ROOT/.venv/bin/activate"
echo "   3. Follow the quickstart:"
echo "      https://rosettacommons.github.io/RFD4-Proteina/latest/tutorial/quickstart.html"
echo ""
