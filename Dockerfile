# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS runtime

ARG UV_VERSION=0.12.13
ENV DEBIAN_FRONTEND=noninteractive \
    PROJECT_ROOT=/opt/rfd4 DATA_PATH=/workspace/models CLUSTER=local \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential curl ca-certificates \
    libxrender1 libx11-6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/nvidia/bin:${PATH}
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /opt/rfd4
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    python -c 'import tomllib; print(next(d for d in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"] if d.startswith("torch==")))' > /tmp/build-constraints.txt && \
    uv pip install --no-sources --build-constraint /tmp/build-constraints.txt \
      --torch-backend cu132 '.[metrics-cuda13]' && \
    uv pip check && rm /tmp/build-constraints.txt
RUN python -c 'import torch, atomworks, rfd4_proteina; assert torch.version.cuda == "13.2"' && \
    rfd4-generate --help
RUN mkdir -p /workspace/models
WORKDIR /workspace
CMD ["rfd4-generate", "--help"]

FROM runtime AS api
RUN uv pip install --no-cache -r /opt/rfd4/deploy/api/requirements.txt && uv pip check
ENV PYTHONPATH=/opt/rfd4/deploy/api HOME=/workspace OMP_NUM_THREADS=4 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
