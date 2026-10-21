#!/usr/bin/env bash
# CPU lane pytest marker filter (collect and run must match).
# `not digs` keeps host-specific sanity checks (maintainer clusters only) out of this lane.
CPU_PYTEST_MARKER='not benchmark and not very_slow and not gpu and not regression and not digs'

set -euo pipefail  # Exit on error

echo "Running from $PWD"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Cluster env: SIF and mirror paths come from the runner-local profile ($RFD4_CI_PROFILE).
CI_CLUSTER="${CI_CLUSTER:-coreweave}"
# The cluster profile lives on the runner host, never in the repository.
PROFILE="${RFD4_CI_PROFILE:?RFD4_CI_PROFILE must point to the runner-local cluster profile (see .github/ci/RUNNERS.md)}"
set -a
source "$PROFILE"
set +a
# The Python settings loader reads the root .env.<cluster> (gitignored); expose the runner-local profile
# there for this run only.
cp "$PROFILE" ".env.${CI_CLUSTER}"
chmod 600 ".env.${CI_CLUSTER}"
trap 'rm -f ".env.${CI_CLUSTER}"' EXIT

APPTAINER_PATH="${LATENT_APPTAINER_SIF:?LATENT_APPTAINER_SIF must be set in $RFD4_CI_PROFILE}"
TEST_PATH=${TEST_PATH:-"./tests"}
XDIST_WORKERS=${XDIST_WORKERS:-"auto"}

# RFProteina from the checkout; AtomWorks comes from the container image.
export CLUSTER="$CI_CLUSTER"
export PYTHONPATH="/workspace/src"

# Get max processes from environment variable, default to 24 if not set
N_CPU=${N_CPU:-24}
echo "Running tests for $TEST_PATH with max $N_CPU CPUs (SIF=$APPTAINER_PATH)"

# Bind the data roots named by $APPTAINER_BIND_PATHS, as the gpu lane does.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
build_apptainer_bind_args

APPTAINER_ENV=(
  # PEFT imports NGC's Transformer Engine; bind driver libraries but keep tests on CPU.
  --env "CUDA_VISIBLE_DEVICES="
  --env "PYTHONPATH=$PYTHONPATH"
  --env "CLUSTER=$CLUSTER"
  --env "PDB_MIRROR_PATH=${PDB_MIRROR_PATH:-}"
  --env "CCD_MIRROR_PATH=${CCD_MIRROR_PATH:-}"
)

# Ensure we can collect all tests (i.e. imports succeed)
echo "Testing imports by trying to collect all tests"
apptainer exec --nv "${BIND_ARGS[@]}" --pwd /workspace "${APPTAINER_ENV[@]}" "$APPTAINER_PATH" \
  pytest -m "$CPU_PYTEST_MARKER" --collect-only "$TEST_PATH"

# Run the tests. GPU/regression tests run in the separate `gpu` job; the cheap
# `cluster` launcher syntax checks (no GPU) stay in this lane.
apptainer exec --nv "${BIND_ARGS[@]}" --pwd /workspace "${APPTAINER_ENV[@]}" "$APPTAINER_PATH" \
  pytest -m "$CPU_PYTEST_MARKER" \
  -n="$XDIST_WORKERS" --maxprocesses="$N_CPU" --dist=worksteal \
  --cov=rfproteina --cov-report=xml:/workspace/coverage.cpu.xml "$TEST_PATH"
