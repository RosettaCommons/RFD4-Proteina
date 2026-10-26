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
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS model

ARG RFD4_ACCELERATOR=cu13
ARG UV_VERSION=0.12.13

ENV DATA_PATH=/data \
    HF_HOME=/opt/rfd4/cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONNOUSERSITE=1 \
    TMOL_DISABLE_WHEEL_FETCH=1 \
    TMOL_JIT_FALLBACK=0 \
    TMOL_USE_JIT=0

# build-essential: torch.compile (Inductor/Triton) compiles C++ and CUDA glue at run time.
# The lib* packages are runtime dependencies of OpenBabel.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libxrender1 libx11-6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /opt/rfd4

# Dependencies first, from pyproject.toml alone, so source edits do not invalidate this layer.
# TMol and its operators must come from published wheels (never a source build).
COPY pyproject.toml ./
RUN case "${RFD4_ACCELERATOR}" in \
        cu13) set -- --extra all-cuda13 --torch-backend cu130 ;; \
        cu12) set -- --extra all-cuda12 --torch-backend cu126 ;; \
        cpu) set -- --torch-backend cpu ;; \
        *) echo "RFD4_ACCELERATOR must be one of: cu13, cu12, cpu" >&2; exit 2 ;; \
    esac && \
    uv pip install --system --no-cache --no-sources \
        --only-binary tmol,tmol-ops-cu12,tmol-ops-cu13 "$@" -r pyproject.toml

# The application itself.
COPY . .
RUN uv pip install --system --no-cache --no-sources --no-deps . && \
    python -m pip check

# BioIR (inference_backend=bioir) comes with the all-cuda13 extra; cu12 and cpu stay on the native runtime.
RUN if [ "${RFD4_ACCELERATOR}" = "cu13" ]; then \
        python -c "from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock, PairUpdateBlock, with_cuda_graph"; \
    fi

# The cpu profile has no CUDA extra; install the physics metrics it needs directly.
RUN if [ "${RFD4_ACCELERATOR}" = "cpu" ]; then \
        uv pip install --system --no-cache --only-binary tmol "tmol>=0.2.0,<0.3" posebusters; \
    fi

RUN mkdir -p /data /opt/rfd4/cache/huggingface /workspace
WORKDIR /workspace

# Smoke test: imports, TMol operators for the selected profile, Inductor, and the CLI.
RUN TORCHINDUCTOR_CACHE_DIR=/tmp/rfd4-build-compile RFD4_ACCELERATOR="${RFD4_ACCELERATOR}" python -c \
        'import os; from importlib.metadata import distributions, version; import torch, atomworks, rfproteina; from openbabel import pybel; from tmol.score.ljlk.potentials._compiled import ljlk_pose_scores; accelerator = os.environ["RFD4_ACCELERATOR"]; assert torch.__version__.endswith("+" + {"cu13": "cu130", "cu12": "cu126", "cpu": "cpu"}[accelerator]); assert pybel.readstring("smi", "CCO").OBMol.NumAtoms() == 3; assert callable(ljlk_pose_scores); ops = {d.metadata["Name"].lower().replace("_", "-") for d in distributions() if d.metadata["Name"].lower().replace("_", "-").startswith("tmol-ops-")}; assert ops == (set() if accelerator == "cpu" else {"tmol-ops-" + accelerator}), ops; x = torch.ones(8); torch.testing.assert_close(torch.compile(lambda t: t.sin() + 1, backend="inductor", fullgraph=True)(x), x.sin() + 1)' && \
    rm -rf /tmp/rfd4-build-compile && \
    rfd4-generate --help

CMD ["rfd4-generate", "--help"]

FROM model AS api
RUN case "${RFD4_ACCELERATOR}" in \
        cu13) set -- --extra all-cuda13 --torch-backend cu130 ;; \
        cu12) set -- --extra all-cuda12 --torch-backend cu126 ;; \
        cpu) set -- --torch-backend cpu ;; \
    esac && \
    uv pip install --system --no-cache --no-sources \
        --only-binary tmol,tmol-ops-cu12,tmol-ops-cu13 "$@" --extra api -r /opt/rfd4/pyproject.toml && \
    python -m pip check
ENV PROJECT_ROOT=/opt/rfd4 DATA_PATH=/workspace/models CLUSTER=local \
    PYTHONPATH=/opt/rfd4/deploy/api HOME=/workspace OMP_NUM_THREADS=4 \
    HF_HOME=/workspace/cache/huggingface PYTHONUNBUFFERED=1 \
    TORCH_COMPILE_CACHE_BLOB=/workspace/cache/torch/compile \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    PATH=/usr/local/nvidia/bin:${PATH}
RUN python -c "from app.api import app; assert app.openapi()['info']['title'] == 'RFD4-Proteína'"
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM model AS runtime
