# syntax=docker/dockerfile:1
# RFD4-Proteina application image: the public build.
#
# One file covers every supported accelerator; pick one with a build argument:
#   docker build --build-arg RFD4_ACCELERATOR=cu13 -t rfd4-proteina:cu13 .   # default
#   docker build --build-arg RFD4_ACCELERATOR=cu12 -t rfd4-proteina:cu12 .
#   docker build --build-arg RFD4_ACCELERATOR=cpu  -t rfd4-proteina:cpu  .
#
# RFD4_ACCELERATOR selects the extra and the Torch wheel index, nothing else:
#   cu13 -> .[all-cuda13], Torch cu130, tmol-ops-cu13
#   cu12 -> .[all-cuda12], Torch cu126, tmol-ops-cu12
#   cpu  -> no metrics extra,  Torch cpu,   no CUDA operators
#
# Everything installs from public PyPI and the public PyTorch index. `--no-sources`
# ignores [tool.uv.sources] in pyproject.toml, so no Git or SSH dependency is reachable
# from this build and it needs no submodule and no internal credentials. TMol and its
# operator packages come from published wheels only, never a source build.
#
# CUDA runtime libraries come from the PyTorch wheels, so no CUDA base image is needed;
# the host only needs an NVIDIA driver and the NVIDIA container toolkit.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ARG RFD4_ACCELERATOR=cu13
# Minimum uv: 0.13.4 is the first release with the emSign CA fix (python.org announcement, Oct 20 2026).
ARG UV_MIN_VERSION=0.13.4

ENV DATA_PATH=/data \
    HF_HOME=/opt/rfd4/cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONNOUSERSITE=1 \
    TMOL_DISABLE_WHEEL_FETCH=1 \
    TMOL_JIT_FALLBACK=0 \
    TMOL_USE_JIT=0

# build-essential: torch.compile (Inductor/Triton) compiles C++ and CUDA glue at run time.
# The lib* packages are runtime dependencies of OpenBabel.
# ca-certificates: makes sure the system CA bundle is the newest the distro offers.
#
# Bootstrap trust (eMudhra/emSign CA incident, Oct 2026): the pip shipped in the base image bundles a
# certifi that still trusts the compromised CA, and pip always trusts its bundled certifi in addition
# to the system store. The base image is pinned by digest, so its pip does not change on rebuild.
# Install the fixed pip from a hash-pinned URL so this download is verified by checksum rather than
# by the old trust store. Remove this step once the base image ships pip >= 26.2.2.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates build-essential libxrender1 libx11-6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" && \
    python -m pip install --no-cache-dir "uv>=${UV_MIN_VERSION}"

WORKDIR /opt/rfd4

# Dependencies first, from pyproject.toml alone, so source edits do not invalidate this layer.
# TMol and its operators must come from published wheels (never a source build).
# The certifi constraint is a floor so no resolution can install a copy that trusts the compromised CA;
# it only applies if certifi is pulled in and does not add the package.
COPY pyproject.toml ./
RUN printf 'certifi>=2026.10.19\n' > /tmp/rfd4-constraints.txt && \
    case "${RFD4_ACCELERATOR}" in \
        cu13) set -- --extra all-cuda13 --torch-backend cu130 ;; \
        cu12) set -- --extra all-cuda12 --torch-backend cu126 ;; \
        cpu) set -- --torch-backend cpu ;; \
        *) echo "RFD4_ACCELERATOR must be one of: cu13, cu12, cpu" >&2; exit 2 ;; \
    esac && \
    uv pip install --system --no-cache --no-sources -c /tmp/rfd4-constraints.txt \
        --only-binary tmol,tmol-ops-cu12,tmol-ops-cu13 "$@" -r pyproject.toml && \
    rm -f /tmp/rfd4-constraints.txt

# The application itself.
COPY . .
RUN uv pip install --system --no-cache --no-sources --no-deps . && \
    python -m pip check && \
    uv --version && \
    python -m pip --version && \
    (python -m pip list 2>/dev/null | grep -i '^certifi ' || true)

# BioIR (inference_backend=bioir) comes with the all-cuda13 extra; cu12 and cpu stay on the native runtime.
RUN if [ "${RFD4_ACCELERATOR}" = "cu13" ]; then \
        python -c "from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock, PairUpdateBlock, with_cuda_graph"; \
    fi

# The cpu profile has no CUDA extra; install the physics metrics it needs directly.
RUN if [ "${RFD4_ACCELERATOR}" = "cpu" ]; then \
        printf 'certifi>=2026.10.19\n' > /tmp/rfd4-constraints.txt && \
        uv pip install --system --no-cache -c /tmp/rfd4-constraints.txt --only-binary tmol "tmol>=0.2.0,<0.3" posebusters && \
        rm -f /tmp/rfd4-constraints.txt; \
    fi

RUN mkdir -p /data /opt/rfd4/cache/huggingface /workspace
WORKDIR /workspace

# Smoke test: imports, TMol operators for the selected profile, Inductor, and the CLI.
RUN TORCHINDUCTOR_CACHE_DIR=/tmp/rfd4-build-compile RFD4_ACCELERATOR="${RFD4_ACCELERATOR}" python -c \
        'import os; from importlib.metadata import distributions, version; import torch, atomworks, rfproteina; from openbabel import pybel; from tmol.score.ljlk.potentials._compiled import ljlk_pose_scores; accelerator = os.environ["RFD4_ACCELERATOR"]; assert torch.__version__.endswith("+" + {"cu13": "cu130", "cu12": "cu126", "cpu": "cpu"}[accelerator]); assert pybel.readstring("smi", "CCO").OBMol.NumAtoms() == 3; assert callable(ljlk_pose_scores); ops = {d.metadata["Name"].lower().replace("_", "-") for d in distributions() if d.metadata["Name"].lower().replace("_", "-").startswith("tmol-ops-")}; assert ops == (set() if accelerator == "cpu" else {"tmol-ops-" + accelerator}), ops; x = torch.ones(8); torch.testing.assert_close(torch.compile(lambda t: t.sin() + 1, backend="inductor", fullgraph=True)(x), x.sin() + 1)' && \
    rm -rf /tmp/rfd4-build-compile && \
    rfd4-generate --help

CMD ["rfd4-generate", "--help"]
