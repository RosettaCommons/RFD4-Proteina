#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Run GPU E2E regression tests inside Apptainer with squashfs mounts.

Executed via launch_run_gpu_regression_tests.sh (or directly by srun on DIGS).
The latent_exec.sh shebang sets up the Apptainer container and squashfs mounts
before Python runs this file.

Options:
    --tests NAMES   Comma-separated subset to run (default: all four tests).
                    Choices: sota_training,ae_training,inference,beam_search
"""

import argparse
import os
import sys
from pathlib import Path

# CWD must be projects/rfproteina/ for pytest paths to resolve.
PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)

# Must be set before CUDA is initialized (cannot be passed as a Hydra override).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ["RUN_GPU_E2E_TESTS"] = "1"

import pytest  # noqa: E402  (import after chdir / env setup)

ALL_TESTS = {
    "inference": "tests/e2e/test_gpu_regression.py::test_inference_regression",
    "beam_search": "tests/e2e/test_gpu_regression.py::test_beam_search_regression",
    "sota_training": "tests/e2e/test_gpu_regression.py::test_sota_training_regression",
    "ae_training": "tests/e2e/test_gpu_regression.py::test_ae_training_regression",
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--tests",
    default=",".join(ALL_TESTS),
    help="Comma-separated list of tests to run.",
)
args = parser.parse_args()

selected = [t.strip() for t in args.tests.split(",")]
unknown = [t for t in selected if t not in ALL_TESTS]
if unknown:
    parser.error(f"Unknown test(s): {unknown}. Valid: {list(ALL_TESTS)}")

sys.exit(pytest.main([ALL_TESTS[t] for t in selected] + ["-v", "-s"]))
