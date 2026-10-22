# Generation on AWS HealthOmics

Run the existing `rfd4-generate` CLI as a Nextflow DSL2 private workflow. One
annotated CIF and a matching model bundle produce generated structures, effective
Hydra configurations, per-shard provenance, and a JSON/CSV summary. Replicates are
split using the CLI's existing `job_id` / `gen_njobs` semantics.

This example covers ordinary generation. Metrics and refolding are disabled
explicitly; the summary reports generated files and execution time, not
designability or biological success. The separate AF2/AF3/OpenFold3 runtimes and
reward-guided search need additional orchestration.

## Inputs and settings

Use one annotated `.cif`, `.cif.gz` or `.cif.zst` input. Prepare YAML/JSON condition
specifications as annotated CIFs before submitting them to this workflow; see the
[input preparation guide](../../docs/how-to/input_prep/index.md).

The model bundle is a directory containing `config.json` and
`model.safetensors`. Both files come from the same exported bundle and include the
matching flow model and autoencoder. For example:

```bash
hf download nvidia/RFD4-Proteina config.json model.safetensors \
    --revision a06e2509c7033b010fc43614e9fd840303588502 \
    --local-dir runs/healthomics/model
```

`settings.json` contains explicit `seed`, `num_replicates`, `batch_size`, and
optional Hydra `overrides`. The supplied settings use one replicate and skip
compilation for a first smoke test. For larger runs, remove
`compile_lengths=null` to retain the model's normal length buckets. Set the
recycle strategy for your task: for example `medium/front_loaded` for PPI and
`medium/midpeak` for small-molecule or MCSA tasks. See the
[task schedule table](../../docs/how-to/run-the-model.md#recycling-schedules).

The workflow accepts sampling, precision, compression, and native network
overrides. Input paths, output paths, shard counts, seeds, metrics, and search
cannot be changed through the override list. Unsupported keys fail explicitly.

Choose `num_shards <= num_replicates`. Each shard receives the original base seed;
the CLI adds its zero-based shard ID exactly once. Changing the number of shards
changes the random streams and batching, so retain it when reproducing a run.
Settings, input files, model files, helper script and shard IDs are task inputs,
allowing Nextflow/HealthOmics task caching to distinguish different experiments.
Use an immutable container digest and a new seed when requesting new samples.

## Build with Finch

Finch can build the Linux amd64 image on macOS for execution on HealthOmics.
Use the repository's existing Dockerfile and select the CUDA profile explicitly:

```bash
finch build --platform linux/amd64 --output type=image --build-arg RFD4_ACCELERATOR=cu12 \
    -t rfd4-proteina:cuda12 .
finch build --platform linux/amd64 --output type=image -f examples/healthomics/Containerfile \
    --build-arg RFD4_IMAGE=rfd4-proteina:cuda12 \
    -t rfd4-proteina:healthomics .
finch run --rm --platform linux/amd64 rfd4-proteina:healthomics \
    rfd4-generate --help
```

The CLI check validates the container environment. Actual model generation still
requires an NVIDIA GPU, so use HealthOmics for the GPU check when building on a
Mac. Ensure the Finch VM has enough free disk space for PyTorch and CUDA wheels.
The small second image adds `procps`: Nextflow task metrics need `ps` before your
Python command starts. A successful `rfd4-generate --help` alone does not check
this workflow requirement.

After creating your ECR repository, authenticate through the AWS CLI's configured
credentials, tag the image, and push it:

```bash
aws ecr get-login-password --region us-east-1 |
    finch login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
finch tag rfd4-proteina:healthomics \
    YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/YOUR_REPOSITORY:validation
finch push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/YOUR_REPOSITORY:validation
```

Use the resulting ECR digest as `container_image` when starting a workflow.

## Optional: run locally on a supported NVIDIA host

Use Java 17 or later, Nextflow **25.10.0**, and a Linux amd64 CUDA image built from
the repository's existing Dockerfile. The optional Nextflow `docker` profile is
for Linux NVIDIA hosts with Docker available. Validate the chosen CUDA profile
against the host driver; the example uses CUDA 12.

```bash
docker build --platform linux/amd64 --build-arg RFD4_ACCELERATOR=cu12 \
    -t rfd4-proteina:cuda12 .
docker build --platform linux/amd64 -f examples/healthomics/Containerfile \
    --build-arg RFD4_IMAGE=rfd4-proteina:cuda12 \
    -t rfd4-proteina:healthomics .

nextflow run examples/healthomics -profile docker \
    --input_cif "$PWD/data/benchmarks/monomer/monomer-short.cif.zst" \
    --model_bundle "$PWD/runs/healthomics/model" \
    --settings "$PWD/examples/healthomics/settings.json" \
    --container_image rfd4-proteina:healthomics \
    --num_shards 1 \
    --outdir "$PWD/runs/healthomics/results"
```

Generation requests one GPU, eight CPUs and 32 GB of host memory per task. The
HealthOmics accelerator is `nvidia-l40s`, a GPU qualified by RFD4-Proteina.
`generation_accelerator` can explicitly select `nvidia-l4` (24 GB) for a separate
compatibility and memory check. The workflow never switches GPUs automatically.
L4 runs must be qualified for the intended input lengths and batch sizes before
scaling; successful container loading does not establish model generation support.
`generation_cpus` adjusts the CPU request without changing the accelerator;
for example, `--generation_cpus 4` can select `omics.g6e.xlarge` instead of
`omics.g6e.2xlarge`. Measure CPU utilization and end-to-end runtime before choosing
the smaller request for your workload.
The L40S default is a compatibility baseline, not a measured cost optimum.
Compare successfully generated structures per billed compute hour with the same
input, checkpoint and sampling settings. Include model startup and failed
attempts, and record provisioning time separately from model processing time.
With this workflow's 32 GB host-memory request, selecting L4 requires at least
`omics.g6.2xlarge` even when `generation_cpus` is four.
Start with batch size one and verify the longest intended input before scaling.
The default task timeout is one hour; resource selectors can be adjusted in a
Nextflow configuration without changing model sampling settings.
Inductor uses a cache inside each task's work directory instead of the shared
HealthOmics `/tmp`; an explicitly configured `TORCHINDUCTOR_CACHE_DIR` takes
precedence. Compilation caches are not published with generated structures.

## Run on HealthOmics

Use a HealthOmics Region that supports the selected accelerator. The commands
below use `us-east-1`, where the short L4 example was validated. Supply your own
S3 bucket, ECR repository and HealthOmics run role. The role needs access to the
input/model objects, output prefix, task images and HealthOmics CloudWatch logs.
See the AWS documentation for [private workflows](https://docs.aws.amazon.com/omics/latest/dev/private-workflows.html),
[container permissions](https://docs.aws.amazon.com/omics/latest/dev/permissions-ecr.html)
and [task accelerators](https://docs.aws.amazon.com/omics/latest/dev/task-accelerators.html).

Push the tested image to ECR and record its digest. Upload the two model files,
the annotated CIF and settings to S3 in the workflow's Region. Staging the model
as a workflow input avoids downloading weights from the Hub inside every GPU
task; no HF token or internet access is needed during generation.

Package this directory from its root:

```bash
cd examples/healthomics
zip -r /tmp/rfd4-healthomics.zip \
    main.nf nextflow.config conf bin parameter-template.json README.md \
    -x '*/__pycache__/*'

aws omics create-workflow \
    --region us-east-1 \
    --name rfd4-generation \
    --engine NEXTFLOW \
    --accelerators GPU \
    --definition-zip fileb:///tmp/rfd4-healthomics.zip \
    --main main.nf \
    --parameter-template file://parameter-template.json
```

Wait for the workflow to become `ACTIVE`. Create a run parameter file using
your own S3 URIs and ECR digest. These parameters explicitly select L4 for the
validated 100-residue smoke example; the workflow default remains L40S:

```json
{
  "input_cif": "s3://YOUR_BUCKET/inputs/monomer-short.cif.zst",
  "model_bundle": "s3://YOUR_BUCKET/models/rfd4/",
  "settings": "s3://YOUR_BUCKET/inputs/settings.json",
  "container_image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/YOUR_REPOSITORY@sha256:YOUR_DIGEST",
  "num_shards": "1",
  "generation_cpus": "4",
  "generation_accelerator": "nvidia-l4"
}
```

Start the run with the HealthOmics profile. It sets the required publication
paths and exports Nextflow execution reports. The pinned engine version matches
the locally tested workflow.

```bash
aws omics start-run \
    --region us-east-1 \
    --workflow-id YOUR_WORKFLOW_ID \
    --role-arn YOUR_HEALTHOMICS_RUN_ROLE_ARN \
    --parameters file://run-parameters.json \
    --output-uri s3://YOUR_BUCKET/outputs/first-run/ \
    --storage-type DYNAMIC \
    --engine-settings '{"engineVersion":"25.10.0","profile":"healthomics"}' \
    --log-level ALL
```

Creating a private workflow does not register a managed Ready2Run workflow.

## Outputs and validation

`shards/shard-N/` contains the native `inference_samples/` tree,
`effective_config.yaml`, Hydra output, and `provenance.json`. Compression is
preserved. Provenance records the input and model SHA-256 hashes, requested
settings, command arguments, effective seed, container reference, installed
package version, output hashes, and elapsed time. It never dumps the environment.
Before loading the model, a separate preflight process checks CUDA availability
and records the actual GPU, GPU memory, PyTorch version and CUDA build.

`summary/summary.json` and `summary/shards.csv` combine complete shard inventories.
The wrapper fails if generation exits unsuccessfully, produces empty files, or
produces fewer/more designs than requested. It also checks the effective seed and
output names recorded in the model's native inventory CSV. Summarization rejects
missing shards and inconsistent settings, and rechecks the transferred generated
files against their sizes and SHA-256 hashes.

Nextflow publishes shard directories after successful task completion. After
input validation, a failed shard writes its provenance, including any partial
output inventory, to task stderr with the prefix `RFD4_SHARD_FAILURE`.
HealthOmics retains this record in
CloudWatch task logs even when the failed task's files are not published to S3.
Check run/task status and logs for failures; a failed run must not be counted as
a successful design campaign.
The helper's elapsed time starts after input/model hashing and includes CUDA
preflight, CLI startup and configuration composition. It is not a GPU kernel
benchmark. Full task timing and resource allocation are also available in
the Nextflow reports and HealthOmics metrics.

The reference validation on 2026-10-22 used `us-east-1`, Nextflow 25.10.0,
RFD4-Proteina 0.0.3, PyTorch 2.14.1+cu126, and one NVIDIA L4 per generation
task (`generation_cpus=4`, 32 GB host memory). The CUDA 12 image was built with
Finch 1.17.0 on Apple Silicon using Rosetta emulation. The reviewed workflow
completed three replicates across two shards (2+1), using batch size two and
base seed 61. Native inventories confirmed seeds 61, 61 and 62; all output hashes,
summary and report exports were checked. All three 100-residue outputs passed
annotated-CIF pipeline, finite-coordinate and residue/chain-count checks.
A separate CPU-only test failed the CUDA preflight as expected and retained its
failure manifest in CloudWatch without publishing a successful summary.
Local replay tests also rejected a missing CIF, same-size file corruption and
an incorrect native seed.

The generated CIFs retain warnings for unspecified formal charges and missing
`_struct_conn`; no oracle refolding was performed. This validates the short
monomer example, not larger inputs, larger batches or other GPU types.

Run the standalone tests from the repository root without installing the model:

```bash
python3 tests/test_healthomics_workflow.py -v
```

When Nextflow 25.10.0 is on `PATH`, the same test runs the actual two-shard
workflow in stub mode using clearly marked fixtures. No Docker, GPU, model
weights or AWS account is needed for this test. Stub reports have status `STUB`
and zero generated designs; they are never reported as successful model runs.
