"""Every symbol an entrypoint imports from the package must actually exist.

The entrypoints (``train.py``, ``generate.py``, ...) are only ever executed as
subprocesses by the GPU lane, so nothing in the CPU suite imports them -- a stale import
inside one is invisible until a GPU allocation runs it, minutes in.

This resolves each entrypoint's ``from rfproteina.* import ...`` statements
against the real modules, on CPU, in about a second.

It exists because dissolving ``foundry`` moved the old ``utils/logging.py`` aside to
``utils/config_html.py``, leaving ``train.py`` importing ``save_config_html`` from a
module that no longer defines it -- and only the GPU lane caught it.
"""

from __future__ import annotations

import ast
import importlib

import pytest

from tests.e2e.helpers.gpu_e2e import ROOT

ENTRYPOINTS = [
    "src/rfproteina/train.py",
    "src/rfproteina/generate.py",
    "src/rfproteina/evaluate.py",
    "src/rfproteina/partial_autoencoder/train.py",
]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_entrypoint_imports_resolve(entrypoint: str):
    path = ROOT / entrypoint
    assert path.is_file(), f"entrypoint missing: {entrypoint}"

    missing: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or not node.module.startswith("rfproteina"):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            if alias.name != "*" and not hasattr(module, alias.name):
                missing.append(f"{node.module}.{alias.name}")

    assert not missing, f"{entrypoint} imports names that do not exist: {missing}"
