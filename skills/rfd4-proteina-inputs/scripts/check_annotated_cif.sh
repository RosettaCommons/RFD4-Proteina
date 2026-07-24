#!/usr/bin/env bash
# Thin wrapper over scripts/json2cif/check_annotated_cif.py — validates an annotated
# CIF (invariant pipeline + gotcha scan) with the CCD env var set. Never repairs.
# Exits 1 on any error (with --strict, also on warnings); 2 if the file can't load.
#
#   bash skills/rfd4-proteina-inputs/scripts/check_annotated_cif.sh out.cif
#   bash skills/rfd4-proteina-inputs/scripts/check_annotated_cif.sh out.cif --strict
#
# Flags after the path pass through to check_annotated_cif.py (fire):
# --strict, --skip_pipeline, --quiet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
    echo "usage: check_annotated_cif.sh <file.cif> [--strict] [--quiet] [--skip_pipeline]" >&2
    exit 2
fi

export ALLOW_BIOTITE_CCD="${ALLOW_BIOTITE_CCD:-True}"
exec python scripts/json2cif/check_annotated_cif.py "$@"
