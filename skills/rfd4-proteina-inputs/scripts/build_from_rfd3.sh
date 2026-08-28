#!/usr/bin/env bash
# Convert an existing RFD3 JSON/YAML example with the Python compatibility builder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 2 ]; then
    echo "usage: build_from_rfd3.sh <benchmark.json> <key> [--output out.cif | --list_metal_candidates | --dump ...]" >&2
    exit 2
fi

export ALLOW_BIOTITE_CCD="${ALLOW_BIOTITE_CCD:-True}"
RFPROTEINA_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$RFPROTEINA_PYTHON" ]; then
    RFPROTEINA_PYTHON="python"
fi
exec "$RFPROTEINA_PYTHON" skills/rfd4-proteina-inputs/scripts/build_from_rfd3.py "$@"
