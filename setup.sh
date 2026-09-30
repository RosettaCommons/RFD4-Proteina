#!/usr/bin/env bash
# Install the public package and its pinned AtomWorks Git dependency.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if [ ! -d .venv ]; then
    uv venv --python 3.12.10 .venv
fi
uv pip install --python .venv/bin/python -e '.[dev]'
echo "Installed. Activate with: source .venv/bin/activate"
echo 'Download the release weights with: bash scripts/download_example_data.sh "$DATA_PATH" --skip-of3'
