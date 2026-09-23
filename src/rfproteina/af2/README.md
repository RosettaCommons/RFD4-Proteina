# AF2 runtime

RFD4-Proteína's AF2 oracle performs database-free protein refolding from inline
mmCIF. A resident worker keeps `model_1_ptm` and `model_1_multimer_v3` loaded;
the oracle chooses between them from chain count unless `model_name` is set.

Configure the runtime before launching:

```bash
export AF2_RUNTIME_BACKEND=pyxis
export AF2_IMAGE=/path/to/alphafold2-pallas-cuda12-amd64.sqsh
export AF2_MODEL_DIR=/path/to/shared/storage
export AF2_JAX_CACHE_DIR=/path/to/af2/jax_cache
export AF2_RESIDENT_MODELS=model_1_ptm,model_1_multimer_v3
export AF2_USE_PALLAS=true
export AF2_COMPILE_MODE=tuned
```

On ORD, `.ipd/.env.ord` sets `AF2_RUNTIME_BACKEND=pyxis` and points
`AF2_IMAGE` at the canonical squash under `${DATA_PATH}/squash_images/`.
When `AF2_RUNTIME_BACKEND` is unset, `CONTAINER_TYPE=docker` selects Pyxis;
`CONTAINER_TYPE=apptainer` selects Apptainer. Set
`AF2_RUNTIME_BACKEND=docker` explicitly for a local Docker daemon (workstation /
CI host). That path needs an OCI reference (for example
`alphafold2-pallas-cuda12-amd64:latest`), not a `.sif` / `.sqsh` path; the
builder overrides the image `ENTRYPOINT` so the warm worker runs under
`/opt/conda/bin/python`.

Use `configs/metrics/refolding_oracles/af2.yaml` in a metric configuration.
The inference and evaluation launchers pre-start one warm daemon per GPU when
the AF2 environment is configured. To score every design with multiple weight
sets, configure one uniquely named `AF2Oracle` per `model_name`; they share the
same resident worker and return separate metric namespaces and artifacts.

For an ad-hoc daemon:

```bash
python src/rfproteina/af2/af2_daemon.py launch \
  --num-gpus 1 --log-dir /tmp/af2-logs --background
```

Manifest entries accepted by `run_alphafold2_batch.py` contain `name`,
`output_dir`, inline `cif`, and optional `template`, `seed`, `model_name`,
`initial_guess`, and `pct_seq_mask`.

- `initial_guess` (bool, default `false`): when true, seed ColabFold recycle
  `prev_pos` from the design CIF via `all_atom_positions` (orthogonal to AF2
  templates). Requires the `alphafold-colabfold` model overlay in the image.
- `pct_seq_mask` (float in `[0, 1]`, default `0.15`): AF2 BERT MSA
  `replace_fraction`. Set to `0` to disable inference masking stochasticity.

Smoke-check the image has ColabFold initial-guess support:

```python
import inspect
from alphafold.model import model
assert "all_atom_positions" in inspect.getsource(model.RunModel.predict)
```

## Pallas container artifacts

The image is built manually from the public
[`alphafold-colabfold`](https://pypi.org/project/alphafold-colabfold/)
package. Image-build scripts and Dockerfiles are intentionally not stored in
this repository. Start from the same DeepMind AlphaFold CUDA 12 base used by
the deployment, then create a temporary Dockerfile outside the worktree:

```bash
cat >/tmp/af2-pallas.Dockerfile <<'DOCKERFILE'
ARG AF2_BASE_IMAGE
FROM ${AF2_BASE_IMAGE}
RUN python -m pip install --no-cache-dir --force-reinstall \
      "alphafold-colabfold==2.3.18" \
      "jax[cuda12]==0.10.2" \
 && rm -rf /app/alphafold/alphafold/model \
 && cp -a /opt/conda/lib/python3.11/site-packages/alphafold/model \
      /app/alphafold/alphafold/model
RUN python - <<'PY'
import inspect
from alphafold.data import mmcif_parsing
from alphafold.model import config, model, tri_flash, tri_mul

cfg = config.model_config("model_1_ptm")
assert mmcif_parsing.__file__.startswith("/app/alphafold/")
assert "jit_kwargs" in inspect.signature(model.RunModel).parameters
assert "all_atom_positions" in inspect.getsource(model.RunModel.predict)
assert hasattr(tri_flash, "pallas_attention")
assert hasattr(tri_mul, "gated_dual_proj")
cfg.model.global_config.use_pallas = True
cfg.model.global_config.bfloat16 = True
PY
DOCKERFILE

cbuild build \
  --dockerfile=/tmp/af2-pallas.Dockerfile \
  --context=. \
  --build-arg "AF2_BASE_IMAGE=PUBLIC_OR_PINNED_DEEPMIND_AF2_CUDA12_BASE" \
  --cpu 16 --mem 64Gi --platform linux/amd64 \
  --name=registry.example.com/genair/protein-foundation-models/alphafold2-pallas-cuda12-amd64 \
  --cache-repo=registry.example.com/genair/protein-foundation-models/alphafold2-pallas-cache \
  --tag-latest
```

For the direct Docker backend on a workstation, pull the OCI image (or load a
saved archive) and point `AF2_IMAGE` at the local tag:

```bash
docker pull registry.example.com/genair/protein-foundation-models/alphafold2-pallas-cuda12-amd64:latest
docker tag registry.example.com/genair/protein-foundation-models/alphafold2-pallas-cuda12-amd64:latest \
  alphafold2-pallas-cuda12-amd64:latest
export AF2_RUNTIME_BACKEND=docker
export AF2_IMAGE=alphafold2-pallas-cuda12-amd64:latest
```

Model params still come from the CoreWeave bucket (`s3://YOUR_DATASET_STORE/af2/models/params/`).

Export the Pyxis/Enroot squash on ORD:

```bash
OCI=registry.example.com/genair/protein-foundation-models/alphafold2-pallas-cuda12-amd64:latest
SQSH=/path/to/shared/storage
cbuild export "${OCI}" "${SQSH}"
```

Build the Apptainer artifact manually on a host with Apptainer:

```bash
OCI=registry.example.com/genair/protein-foundation-models/alphafold2-pallas-cuda12-amd64:latest
SIF=/path/to/shared/storage
apptainer build --force "${SIF}" "docker://${OCI}"
```

On DIGS, `.ipd/.env.digs` selects the Apptainer backend and the SIF at
`${DATA_PATH}/af2/apptainer_images/alphafold2-pallas-cuda12-amd64.sif`.
Do the live SIF smoke test there; ORD does not provide Apptainer.

Store the artifacts using the same layout as AF3:

```text
Local SIF (CoreWeave): ${DATA_PATH}/af2/alphafold2-pallas-cuda12-amd64.sif
Local SIF (DIGS):      ${DATA_PATH}/af2/apptainer_images/alphafold2-pallas-cuda12-amd64.sif
S3 SIF:                s3://YOUR_DATASET_STORE/af2/alphafold2-pallas-cuda12-amd64.sif
                       s3://YOUR_DATASET_STORE/af2/apptainer_images/alphafold2-pallas-cuda12-amd64.sif
Local squash:          ${DATA_PATH}/squash_images/alphafold2-pallas-cuda12-amd64.sqsh
S3 squash:             s3://YOUR_DATASET_STORE/squash_images/alphafold2-pallas-cuda12-amd64.sqsh
Model params:          ${AF2_MODEL_DIR}/params/params_model_*.npz
S3 params:             s3://YOUR_DATASET_STORE/af2/models/params/
```

On CoreWeave, `.ipd/.env.coreweave` points `AF2_IMAGE` at
`/path/to/data` and `AF2_MODEL_DIR` at
`/path/to/data` (so params live under `/path/to/data`).

On Apptainer/CoreWeave, the warm worker sets
`LD_LIBRARY_PATH=/.singularity.d/libs` so JAX can load the driver libraries
that `--nv` injects (this image does not set that path itself).

The Apptainer SIF must mark its squashfs payload as the primary system
partition (`Partition Type: *System`). If `apptainer exec` reports
`no root filesystem found`, repair the descriptor without rebuilding:

```bash
apptainer sif setprim 1 "${SIF}"
# or, with the standalone siftool:
sif setprim 1 "${SIF}"
```

Pull / push CoreWeave objects with s5cmd against `https://cwobject.com`:

```bash
s5cmd --endpoint-url https://cwobject.com cp \
  's3://YOUR_DATASET_STORE/af2/alphafold2-pallas-cuda12-amd64.sif' \
  /path/to/data
s5cmd --endpoint-url https://cwobject.com cp \
  /path/to/data \
  's3://YOUR_DATASET_STORE/af2/alphafold2-pallas-cuda12-amd64.sif'
s5cmd --endpoint-url https://cwobject.com cp \
  /path/to/data \
  's3://YOUR_DATASET_STORE/af2/apptainer_images/alphafold2-pallas-cuda12-amd64.sif'
sha256sum "${SIF}" "${SQSH}"
```

Pallas is enabled by default and requires bf16 plus an NVIDIA GPU with compute
capability 8.0 or newer (Ampere, Ada, Hopper, or Blackwell). Set
`AF2_USE_PALLAS=false` for an A/B comparison or unsupported hardware.
`AF2_COMPILE_MODE` accepts `fast`, `tuned` (default), or `full`.

Do **not** bake genetic databases into the image; the oracle path is
single-sequence only and mounts `AF2_MODEL_DIR` (params) plus
`AF2_JAX_CACHE_DIR` at runtime. Point `AF2_IMAGE` at the artifact appropriate
for the selected runtime backend.
Use a fresh `AF2_JAX_CACHE_DIR` after switching from the DeepMind image because
compiled executables are not compatible across the JAX/model implementation
change.

The worker retains the fixed sequence-length buckets and stable-sorts each
manifest by `(model_name, bucket)`. This groups equal compile shapes without
turning inference into multi-sequence batching.
