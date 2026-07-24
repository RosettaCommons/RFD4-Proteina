#!/usr/bin/env bash
# Thin wrapper over scripts/json2cif/build_cif.py — builds an annotated CIF from an
# RFD3-style benchmark JSON/YAML example, with the CCD env var set. Runs from the
# repo root regardless of where it is invoked.
#
#   bash skills/rfd4-proteina-inputs/scripts/build_cif.sh <benchmark.json> <key> --output out.cif
#   bash skills/rfd4-proteina-inputs/scripts/build_cif.sh <benchmark.json> <key> --list_metal_candidates
#   bash skills/rfd4-proteina-inputs/scripts/build_cif.sh <benchmark.json> <key> --output out.cif \
#        --coordination_bond 'C/ZN/296/ZN,A/CYS/365/SG'
#
# All flags after <benchmark> <key> are passed straight through to build_cif.py
# (fire): --output, --dump, --benchmark_root, --fix_res_id_gap, --coordination_bond,
# --list_metal_candidates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 2 ]; then
    echo "usage: build_cif.sh <benchmark.json> <key> [--output out.cif | --list_metal_candidates | --dump ...]" >&2
    exit 2
fi

export ALLOW_BIOTITE_CCD="${ALLOW_BIOTITE_CCD:-True}"
exec python scripts/json2cif/build_cif.py "$@"
