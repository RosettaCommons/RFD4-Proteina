"""Run an explicitly selected public test suite using configured SLURM resources."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Validate resource arguments and submit the selected suite synchronously."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cpu", "gpu"))
    args = parser.parse_args()
    resources = json.loads(os.environ["CI_SRUN_ARGS"])
    if not isinstance(resources, list) or not resources or not all(isinstance(x, str) for x in resources):
        raise ValueError("CI_SRUN_ARGS must be a nonempty JSON array of srun arguments.")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("CLUSTER", "CONTAINER_TYPE", "PYTHONPATH", "PYROSETTA_CONTAINER"):
        env.pop(key, None)
    env["DATA_PATH"] = str(root / ".ci-data")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        ["bash", "scripts/download_example_data.sh", env["DATA_PATH"], "--with-test-dataset", "--skip-of3"],
        cwd=root,
        env=env,
        check=True,
    )
    selection = "not benchmark and not very_slow and not gpu and not regression and not digs"
    if args.suite == "gpu":
        selection = "gpu and not very_slow and not digs"
        env["RUN_GPU_E2E_TESTS"] = "1"
    subprocess.run(
        ["srun", *resources, sys.executable, "-m", "pytest", "tests", "-m", selection, "-q"],
        cwd=root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
