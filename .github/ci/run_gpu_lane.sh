#!/usr/bin/bash
# =============================================================================
# run_gpu_tests.sh - GPU test lane (self-hosted runner).
# =============================================================================
# Runs the GPU-marked + regression tests inside the latent-dev apptainer.
# Individual training/inference subprocesses (train.py, generate.py, ...) are
# invoked directly by the test suite inside the SIF. CONTAINER_TYPE=docker below
# tells their shebang wrappers to use the current environment instead of trying
# to launch a nested Apptainer, while the outer invocation supplies all binds.
#
# IMPORTANT: this script must itself run inside the `srun` allocation (the
# workflow invokes it as `srun ... bash run_gpu_tests.sh`), exactly like the CPU
# lane's run_tests.sh.
#
# Honors (from the GitHub Actions env / cluster env file):
#   RFD4_CI_PROFILE the runner-local cluster profile to source (SIF, DATA_PATH, mirrors, AF*/OF3 paths)
#   CI_CLUSTER      cluster name exported to the tests (CLUSTER)
#   SOTA_CKPT_PATH  abs path to a .ckpt -> un-skips MCSA validation + search
#   AF3_*           from the cluster env file; host-side daemons + passed in
# =============================================================================
set -euo pipefail

echo "Running on $(hostname) from $PWD"

# CI retry-wrapper sentinel: presence means the srun allocation started and the
# tests are running here, so a non-zero exit is a real failure -- not a TaskProlog
# dirty-GPU rejection (which fails before this point). See lint_and_test.yaml.
[ -n "${GPU_ALLOC_SENTINEL:-}" ] && : > "${GPU_ALLOC_SENTINEL}" || true

# Cluster env: SIF path, DATA_PATH, mirrors, squash mounts, NCCL flags, CLUSTER, etc.
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
ENV_PROFILE_COPY=".env.${CI_CLUSTER}"

SIF="${LATENT_APPTAINER_SIF:?LATENT_APPTAINER_SIF must be set in $RFD4_CI_PROFILE}"
: "${APPTAINERENV_AWS_ACCESS_KEY_ID:?export APPTAINERENV_AWS_ACCESS_KEY_ID in the runner-local profile}"
: "${APPTAINERENV_AWS_SECRET_ACCESS_KEY:?export APPTAINERENV_AWS_SECRET_ACCESS_KEY in the runner-local profile}"
export APPTAINERENV_AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL must be set in $RFD4_CI_PROFILE}"

# Cluster data roots apptainer does not auto-bind, then the squashfs training datasets.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
build_apptainer_bind_args
append_squashfs_binds

# Unique rendezvous port per run for DDP.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="$((20000 + RANDOM % 20000))"

# Host-side AF3 daemons for the training smokes that use AF3 prevalidation.
# AF3 runs in its own apptainer image; daemons must be spawned outside latent-dev.
# AF3_* paths come from the cluster env file (sourced above); pass them through to
# pytest so has_af3 / path checks work inside latent-dev.
# One AF3 daemon per xdist worker to enable parallelism
GPU_PYTEST_WORKERS="${GPU_PYTEST_WORKERS:-6}"
if [ "$GPU_PYTEST_WORKERS" -gt 0 ]; then
  AF3_DAEMON_COUNT="$GPU_PYTEST_WORKERS"
else
  AF3_DAEMON_COUNT=1
fi

AF3_DAEMONS_PID=""
AF2_DAEMONS_PID=""
OF3_DAEMONS_PID=""
AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT=""
AF2_LAUNCHER_SOCKET_TEMPLATE_EXPORT=""
OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT=""
REQUIRE_AF2=0

cleanup_daemons() {
  rm -f "${ENV_PROFILE_COPY:-}"
  for pid in "${AF3_DAEMONS_PID:-}" "${AF2_DAEMONS_PID:-}" "${OF3_DAEMONS_PID:-}"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_daemons EXIT

# A cluster env that declares AF3 must deliver it
if [[ -n "${AF3_IMAGE:-}" && ! -r "${AF3_IMAGE}" ]]; then
  echo "FATAL: AF3_IMAGE=${AF3_IMAGE} is not readable on $(hostname)." >&2
  echo "Stage the image on this cluster, or clear AF3_IMAGE in $RFD4_CI_PROFILE" >&2
  echo "to declare that this cluster has no AF3." >&2
  exit 1
fi
if [[ -n "${AF3_IMAGE:-}" && -n "${AF3_JAX_CACHE_DIR:-}" && -n "${AF3_MODEL_DIR:-}" ]]; then
  AF3_LOG_DIR="${TMPDIR:-/tmp}/af3_gpu_e2e_${SLURM_JOB_ID:-$$}"
  mkdir -p "$AF3_LOG_DIR"
  if AF3_EVAL="$(/usr/bin/python3 src/rfproteina/af3/af3_daemon.py launch \
      --num-gpus "$AF3_DAEMON_COUNT" --log-dir "$AF3_LOG_DIR" --background --skip-if-no-env \
      2>>"$AF3_LOG_DIR/launch.log")"; then
    eval "$AF3_EVAL"
    AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT="${AF3_LAUNCHER_SOCKET_TEMPLATE:-}"
    echo "AF3 daemons ready (pid=${AF3_DAEMONS_PID:-unknown}, template=${AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT})"
  else
    echo "FATAL: AF3 daemon launch failed. See ${AF3_LOG_DIR}/launch.log" >&2
    tail -30 "${AF3_LOG_DIR}/launch.log" >&2 || true
    exit 1
  fi
  # Require AF3 for our standard CI (to avoid silent skips)
  REQUIRE_AF3=1
else
  echo "WARNING: AF3 env incomplete; AF3 tests will skip. Set AF3_IMAGE/AF3_MODEL_DIR/AF3_JAX_CACHE_DIR" \
    "in $RFD4_CI_PROFILE to require them."
  REQUIRE_AF3=0
fi

# AF2 uses the same host-launcher pattern as AF3: fail rather than silently
# skip if the configured oracle artifacts are incomplete or unreadable.
if [[ -n "${AF2_IMAGE:-}${AF2_JAX_CACHE_DIR:-}${AF2_MODEL_DIR:-}" ]]; then
  if [[ ! -r "${AF2_IMAGE:-}" || ! -d "${AF2_MODEL_DIR:-}" || ! -r "${AF2_MODEL_DIR:-}" \
    || ! -d "${AF2_JAX_CACHE_DIR:-}" || ! -r "${AF2_JAX_CACHE_DIR:-}" ]]; then
    echo "FATAL: configured AF2 image, model directory, or JAX cache is missing or unreadable." >&2
    exit 1
  fi
  if [[ ! -w "$AF2_JAX_CACHE_DIR" ]]; then
    AF2_WRITABLE_JAX_CACHE_DIR="${TMPDIR:-/tmp}/af2_jax_cache_${SLURM_JOB_ID:-$$}"
    mkdir -p "$AF2_WRITABLE_JAX_CACHE_DIR"
    # Shared caches can contain entries written without world-read permission by
    # another runner. Reuse everything readable and compile the rest locally.
    if ! cp -a "$AF2_JAX_CACHE_DIR/." "$AF2_WRITABLE_JAX_CACHE_DIR/" 2>/dev/null; then
      echo "WARNING: some shared AF2 JAX cache entries are unreadable; treating them as cache misses." >&2
    fi
    chmod -R u+rwX "$AF2_WRITABLE_JAX_CACHE_DIR"
    export AF2_JAX_CACHE_DIR="$AF2_WRITABLE_JAX_CACHE_DIR"
  fi
  AF2_LOG_DIR="${TMPDIR:-/tmp}/af2_gpu_e2e_${SLURM_JOB_ID:-$$}"
  mkdir -p "$AF2_LOG_DIR"
  if ! AF2_SMOKE_OUTPUT="$(
    apptainer exec --env PYTHONPATH=/app/alphafold \
      "$AF2_IMAGE" /opt/conda/bin/python -c "import alphafold" 2>&1
  )"; then
    echo "FATAL: AF2_IMAGE cannot be executed by Apptainer:" >&2
    echo "$AF2_SMOKE_OUTPUT" >&2
    apptainer sif list "$AF2_IMAGE" >&2 || true
    exit 1
  fi
  if AF2_EVAL="$(/usr/bin/python3 src/rfproteina/af2/af2_daemon.py launch \
      --num-gpus "$AF3_DAEMON_COUNT" --log-dir "$AF2_LOG_DIR" --background --skip-if-no-env \
      2>>"$AF2_LOG_DIR/launch.log")"; then
    eval "$AF2_EVAL"
    AF2_LAUNCHER_SOCKET_TEMPLATE_EXPORT="${AF2_LAUNCHER_SOCKET_TEMPLATE:-}"
    echo "AF2 daemons ready (pid=${AF2_DAEMONS_PID:-unknown}, template=${AF2_LAUNCHER_SOCKET_TEMPLATE_EXPORT})"
    REQUIRE_AF2=1
  else
    echo "FATAL: AF2 daemon launch failed. See ${AF2_LOG_DIR}/launch.log" >&2
    tail -30 "${AF2_LOG_DIR}/launch.log" >&2 || true
    exit 1
  fi
else
  echo "INFO: AF2 is not configured; AF2 GPU oracle tests will skip."
fi

# A second daemon set on converted OF3 weights. `launch` reports under the AF3_ names,
# so stash ours and put AF3's back.
# Safe to co-schedule: the worker sets XLA_PYTHON_CLIENT_PREALLOCATE=false.
if [[ -n "${AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT}" && -d "${OF3_MODEL_DIR:-}" ]]; then
  AF3_DAEMONS_PID_SAVED="$AF3_DAEMONS_PID"
  OF3_LOG_DIR="${AF3_LOG_DIR}/of3"
  mkdir -p "$OF3_LOG_DIR"
  if OF3_EVAL="$(AF3_MODEL_DIR="$OF3_MODEL_DIR" AF3_SOCKET_DIR="/tmp/of3_${SLURM_JOB_ID:-$$}" \
      /usr/bin/python3 src/rfproteina/af3/af3_daemon.py launch \
      --num-gpus "$AF3_DAEMON_COUNT" --log-dir "$OF3_LOG_DIR" --background --skip-if-no-env \
      2>>"$OF3_LOG_DIR/launch.log")"; then
    eval "$OF3_EVAL"
    OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT="${AF3_LAUNCHER_SOCKET_TEMPLATE:-}"
    OF3_DAEMONS_PID="${AF3_DAEMONS_PID:-}"
    AF3_DAEMONS_PID="$AF3_DAEMONS_PID_SAVED"
    AF3_LAUNCHER_SOCKET_TEMPLATE="$AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT"
    # An empty OF3_EVAL would leave AF3's template in place and the OF3 arm would pass on
    # AF3 weights.
    # Compare against the saved copy: the eval above just overwrote the unsuffixed variable.
    if [[ -z "$OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT" \
          || "$OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT" == "$AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT" ]]; then
      echo "FATAL: OF3 daemon launch produced no distinct socket template" \
        "(got '${OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT}'). See ${OF3_LOG_DIR}/launch.log" >&2
      tail -30 "${OF3_LOG_DIR}/launch.log" >&2 || true
      exit 1
    fi
    echo "OF3 daemons ready (pid=${OF3_DAEMONS_PID}, template=${OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT})"
  else
    echo "FATAL: OF3 daemon launch failed. See ${OF3_LOG_DIR}/launch.log" >&2
    tail -30 "${OF3_LOG_DIR}/launch.log" >&2 || true
    exit 1
  fi
elif [[ -n "${OF3_MODEL_DIR:-}" ]]; then
  # Missing weights must not take the lane down, nor pass quietly: leave the template empty
  # and the OF3 arm fails on its own under REQUIRE_AF3.
  echo "WARNING: OF3_MODEL_DIR=${OF3_MODEL_DIR} is not a directory on $(hostname);" \
    "the OF3 benchmark arm will fail. Stage the weights on this cluster." >&2
fi

apptainer exec --nv \
  "${BIND_ARGS[@]}" \
  --env "CLUSTER=${CLUSTER:-}" \
  --env "CONTAINER_TYPE=docker" \
  --env "MASTER_ADDR=$MASTER_ADDR" \
  --env "MASTER_PORT=$MASTER_PORT" \
  --env "SOTA_CKPT_PATH=${SOTA_CKPT_PATH:-}" \
  --env "AF3_LAUNCHER_SOCKET_TEMPLATE=${AF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT}" \
  --env "AF2_LAUNCHER_SOCKET_TEMPLATE=${AF2_LAUNCHER_SOCKET_TEMPLATE_EXPORT}" \
  --env "OF3_LAUNCHER_SOCKET_TEMPLATE=${OF3_LAUNCHER_SOCKET_TEMPLATE_EXPORT}" \
  --env "OF3_MODEL_DIR=${OF3_MODEL_DIR:-}" \
  --env "GPU_PYTEST_WORKERS=${GPU_PYTEST_WORKERS}" \
  --env "REQUIRE_AF3=${REQUIRE_AF3}" \
  --env "REQUIRE_AF2=${REQUIRE_AF2}" \
  --env "AF3_IMAGE=${AF3_IMAGE:-}" \
  --env "AF3_JAX_CACHE_DIR=${AF3_JAX_CACHE_DIR:-}" \
  --env "AF3_MODEL_DIR=${AF3_MODEL_DIR:-}" \
  --env "DATA_PATH=${DATA_PATH:-}" \
  --env "LOCAL_DRIVE_MOUNT=${LOCAL_DRIVE_MOUNT:-/tmp}" \
  --env "CACHE_DIR=${CACHE_DIR:-}" \
  --env "PDB_MIRROR_PATH=${PDB_MIRROR_PATH:-}" \
  --env "CCD_MIRROR_PATH=${CCD_MIRROR_PATH:-}" \
  "$SIF" bash <<'GPU_TEST'
set -euo pipefail
cd /workspace

# RFProteina from the checkout; AtomWorks comes from the container image.
export PYTHONPATH=/workspace/src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# GPU e2e gate (helpers skip unless this is set).
export RUN_GPU_E2E_TESTS=1
# One GPU per xdist worker (conftest pins CUDA_VISIBLE_DEVICES per worker), so each
# test still sees a single GPU -- as it did when the lane ran serially on one.
export GPU_COUNT=1
export ALLOW_SINGLE_GPU_E2E=1

# The inference_backend=bioir e2e tests skip on an image without a BioIR that can run the workflow.
if ! python -c "from rfproteina.nn.bioir_full import require_bioir; require_bioir()" >/dev/null 2>&1; then
  echo "WARNING: this image has no BioIR that can run the whole workflow; the inference_backend=bioir" \
    "e2e tests will skip. Rebuild the SIF with the current BioIR wheel." >&2
fi

# Val CCD benchmark CIFs are gitignored; extract from the tracked tarball when missing.
# Use [[ -d ... ]] first: find on a missing dir exits 1 and kills the script under pipefail.
CCD_TAR="data/benchmarks/structure_prediction/ccd.tar.gz"
CCD_DIR="data/benchmarks/structure_prediction/ccd"
if [[ -f "$CCD_TAR" ]]; then
  n_ccd=0
  if [[ -d "$CCD_DIR" ]]; then
    n_ccd=$(find "$CCD_DIR" -maxdepth 1 -name '*.cif' | wc -l)
  fi
  if [[ "$n_ccd" -eq 0 ]]; then
    echo "Extracting CCD benchmark CIFs from ${CCD_TAR} ..."
    mkdir -p "$(dirname "$CCD_DIR")"
    tar -xzf "$CCD_TAR" -C "$(dirname "$CCD_DIR")"
  fi
fi

# The AE validation benchmark lives on Hugging Face, not in git (`make setup` fetches it). Fetch it when missing;
# the AE training regression test skips when the download is unavailable.
AE_BENCHMARK="data/benchmarks/ae_pdb_temporal_holdout_2026_10_04"
if [[ ! -d "$AE_BENCHMARK" ]] || [[ -z "$(ls -A "$AE_BENCHMARK" 2>/dev/null)" ]]; then
  echo "Fetching ${AE_BENCHMARK} from Hugging Face ..."
  hf download "${RFD4_BENCHMARKS_REPO:-nvidia/RFD4-Proteina}" --repo-type dataset \
    ${RFD4_BENCHMARKS_REVISION:+--revision "$RFD4_BENCHMARKS_REVISION"} \
    --include "benchmarks/ae_pdb_temporal_holdout_2026_10_04/*" --local-dir data \
    || echo "WARNING: could not download the AE validation benchmark; the AE training regression test will skip." >&2
fi

MARKER="gpu or regression"

echo "=== GPU tests: collect-only sanity ==="
pytest -m "${MARKER}" --collect-only ./tests

echo "=== GPU tests: run ==="
# --maxfail, not -x: under xdist -x tears down every worker on the first failure, leaving
# partial junit. Set GPU_E2E_STREAM_OUTPUT=1 and add -s for live subprocess logs.
PYTEST_EXTRA_ARGS="${GPU_PYTEST_EXTRA_ARGS:---maxfail=4 --tb=long}"

# One worker per allocated GPU; 0 runs serially. Each unmarked test is its own loadgroup, so the
# longest-first collection order distributes the heavy tests across workers instead of batching them.
GPU_PYTEST_WORKERS="${GPU_PYTEST_WORKERS:-6}"
XDIST_ARGS=()
[ "$GPU_PYTEST_WORKERS" -gt 0 ] && XDIST_ARGS=(-n "$GPU_PYTEST_WORKERS" --dist loadgroup)

pytest -m "${MARKER}" -v --durations=25 \
  "${XDIST_ARGS[@]}" \
  ${PYTEST_EXTRA_ARGS} \
  --cov \
  --cov-config=/workspace/.github/ci/coverage.gpu.toml \
  --cov-report= \
  --junitxml=/workspace/testing.gpu.junit.xml \
  ./tests

# coverage.py's subprocess patch records each Python process separately. Reporting
# combines those shards before producing the coverage XML.
coverage xml \
  --rcfile=/workspace/.github/ci/coverage.gpu.toml \
  -o /workspace/coverage.gpu.xml

# Guard subprocess coverage explicitly: a green GPU lane with zero hits in the
# real CLI entrypoints would make the combined coverage report misleading.
python - <<'COVERAGE_CHECK'
from pathlib import Path
from xml.etree import ElementTree

report = Path("/workspace/coverage.gpu.xml")
expected = (
    "src/rfproteina/train.py",
    "src/rfproteina/partial_autoencoder/train.py",
    "src/rfproteina/generate.py",
)

root = ElementTree.parse(report).getroot()
hits_by_file: dict[str, int] = {}
for class_node in root.findall(".//class"):
    filename = class_node.attrib.get("filename", "").replace("\\", "/")
    hits_by_file[filename] = hits_by_file.get(filename, 0) + sum(
        int(line.attrib.get("hits", "0")) for line in class_node.findall("./lines/line")
    )

missing = []
for repository_path in expected:
    coverage_names = {
        repository_path,
        repository_path.removeprefix("src/"),
        repository_path.removeprefix("src/rfproteina/"),
    }
    hits = sum(hits for filename, hits in hits_by_file.items() if filename in coverage_names)
    if hits == 0:
        missing.append(repository_path)
if missing:
    discovered = ", ".join(sorted(hits_by_file))
    raise SystemExit(
        f"GPU subprocess coverage is missing for {missing}. "
        f"Files present in the report: {discovered}"
    )
print("GPU subprocess coverage verified for:", ", ".join(expected))
COVERAGE_CHECK
GPU_TEST
