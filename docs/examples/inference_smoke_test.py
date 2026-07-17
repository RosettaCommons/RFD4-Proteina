"""
Inference smoke test
====================

This example assembles the command used to generate one unconditional protein
design from the bundled monomer benchmark. It is intentionally lightweight so
the documentation can build without a GPU or downloaded model checkpoints.
"""

###########################################################################
# Prepare the environment
# -----------------------
#
# Install RFD4-Proteína and download the released checkpoints as described in
# the :doc:`../tutorial/quickstart`. The generation process reads checkpoints
# and shared assets from ``DATA_PATH``.

import shlex
from pathlib import Path

repo_root = Path.cwd()
input_specification = repo_root / "data" / "benchmarks" / "monomer.json"
output_directory = repo_root / "runs" / "smoke"

###########################################################################
# Build the generation command
# ----------------------------
#
# RFD4-Proteína uses Hydra's ``key=value`` override syntax. The same command
# can be pasted into a shell from the repository root.

command = [
    "python",
    "src/rfproteina/generate.py",
    f"inputs={input_specification}",
    "num_replicates=1",
    f"out_dir={output_directory}",
]

print(shlex.join(command))

###########################################################################
# Run and inspect
# ---------------
#
# Run the printed command in an activated RFD4-Proteína environment. The output
# directory will contain the generated CIF and its metadata. Continue with the
# :doc:`../guides/inference` for task-specific inputs, batching, evaluation,
# and reward-guided search.
