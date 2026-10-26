# Run RFD4-Proteina on AWS HealthOmics

This community deployment example uses the upstream RFD4-Proteina model and CLI.
Use this guide to build the RFD4 image with Finch, stage a matching model bundle
and one annotated CIF in S3, and submit generation to managed GPUs on AWS
HealthOmics. You provision the AWS resources once; HealthOmics schedules the
compute and publishes the results for each run.

The first run generates one 100-residue monomer on an NVIDIA L4. The workflow
also supports splitting replicates across GPU tasks, records seeds/configuration
and file hashes, and rejects incomplete outputs. It uses the existing
`rfd4-generate` CLI without changing the model's sampling code.

This example covers generation. Refolding, biological designability and
reward-guided search need separate evaluation/orchestration. AWS usage incurs
charges, including stored images/model files after compute has finished.

## Licenses and attribution

The example follows the repository's [BSD 3-Clause license](../../LICENSE).
The pinned generation model and weights have their own
[BSD 3-Clause LICENSE](https://huggingface.co/nvidia/RFD4-Proteina/blob/a06e2509c7033b010fc43614e9fd840303588502/LICENSE).
Retain the copyright notices, license conditions and disclaimer when
redistributing covered source or model assets. The license does not grant
permission to imply endorsement by the copyright holder or contributors.
The steps below download and stage the model's LICENSE alongside its weights.

Third-party components retain their own terms. The CUDA 12 build includes
[Open Babel under GPLv2](https://github.com/openbabel/openbabel/blob/openbabel-3-1-1/COPYING)
and [cuEquivariance binary kernels under NVIDIA's math-library license](https://pypi.org/project/cuequivariance-ops-cu12/0.13.0/).
Other installed packages and system tools also carry separate notices.
Before distributing a built image, review its exact installed versions and
licensing compatibility, retain applicable notices, and satisfy any
corresponding-source or redistribution requirements. The repository's BSD
license does not cover every component inside the image.

This setup downloads the generation bundle and its LICENSE. Optional
AlphaFold/OpenFold refolding assets and training datasets have separate terms;
review the [upstream model card](https://huggingface.co/nvidia/RFD4-Proteina)
before adding them to this example.

## What you will set up

The supplied [bootstrap template](bootstrap.yaml) creates a dedicated private,
versioned S3 bucket, an immutable-tag ECR repository, a scoped HealthOmics run
role, and a validation run group. The group allows up to two GPUs, 16 vCPUs and
one concurrent run, with a default 60-minute run limit. Creating the stack does
not start GPU compute.

The run role can read `inputs/` and `models/`, read/write `outputs/`, pull images
from this repository, and write HealthOmics logs. The bucket requires encrypted
transport. The bucket, its policy and the image repository are retained if the
stack is removed. The template does not change account-wide ECR/Inspector
scanning settings or create a VPC/NAT gateway.

The tested generation path uses `RESTRICTED` networking: runtime access is
limited to same-Region S3/ECR, with no public internet access. PrivateLink can
provide a private **client-to-API** path as well. For the distinction, optional
VPC configuration and validation checks, see [Private networking](networking.md).

## 1. Prepare your workstation and AWS access

Use a macOS or Linux shell and a checkout containing `examples/healthomics`.
Run the following steps in the **same shell**, starting at the repository root.

| Requirement | Purpose | Check |
| --- | --- | --- |
| Recent AWS CLI v2 with configured credentials | Provision resources and submit/read runs | `aws --version` |
| Finch, with a running VM on macOS | Build and push a Linux amd64 image | `finch --version`; `finch vm status` |
| Hugging Face CLI | Download the pinned model bundle | `hf --help` |
| Python 3.12 | Write parameter JSON and run standalone tests | `python3 --version` |
| `zip` | Package the Nextflow definition | `zip -v` |

Install tools from the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html),
[Finch](https://runfinch.com/docs/) and
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) guides.
This AWS route does not require a local NVIDIA GPU or local Nextflow/Java.
Local workflow tests use Nextflow 25.10.0 and Java 21.
The AWS CLI must support the HealthOmics `start-run --networking-mode` option.

On macOS, initialize a Finch VM only for a new installation (`finch vm init`),
or start the existing VM (`finch vm start`). The validated Apple Silicon build
used Rosetta emulation. Ensure the VM has ample free disk space: PyTorch/CUDA
build layers need substantially more space than the final compressed image.
The fresh Linux validation build used an 80 GiB disk; this is a reference size,
not a minimum requirement. A shared 50 GiB Finch data disk that was already full
could not complete the build. Check available space before downloading layers.
On Linux, follow Finch's permission setup; commands may require `sudo finch`.

Your deployment identity needs permission to create the stack's S3/ECR/IAM/
HealthOmics resources, manage private workflows, upload the assets, and read the
results/logs. Publishing images also needs ECR push permissions; the run role
only pulls images. Starting a run needs `iam:PassRole` for the created run role;
scope it to that role and `iam:PassedToService=omics.amazonaws.com`. The run role
is not your deployment identity. Use your organization's approved deployment
role and adapt the template to required permission boundaries or encryption
policies before deployment.

```bash
# Stop on errors; keep these variables for the remaining steps.
set -euo pipefail
export AWS_REGION=us-east-1
export RFD4_ROOT="$PWD"
export RFD4_STACK=rfd4-healthomics-demo
export RFD4_TAG="validation-$(date -u +%Y%m%dT%H%M%SZ)"
export RFD4_WORK_DIR="$RFD4_ROOT/runs/healthomics-$RFD4_TAG"
test -f "$RFD4_ROOT/pyproject.toml"
test -f "$RFD4_ROOT/examples/healthomics/main.nf"
mkdir -p "$RFD4_WORK_DIR"
aws sts get-caller-identity --region "$AWS_REGION"
```

Confirm that the displayed account/role is the intended one. Use your configured
AWS profile or exported temporary credentials; do not put credentials in the
settings JSON, template or repository.

## 2. Create the dedicated AWS resources

Review `bootstrap.yaml`, then deploy it in the selected Region. Use a new stack
name rather than reusing an unrelated stack. The IAM capability acknowledges
creation of the scoped run role; it does not grant your caller extra permissions.

```bash
aws cloudformation deploy \
    --region "$AWS_REGION" \
    --stack-name "$RFD4_STACK" \
    --template-file "$RFD4_ROOT/examples/healthomics/bootstrap.yaml" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides MaxRunDurationMinutes=60

stack_output() {
    aws cloudformation describe-stacks \
        --region "$AWS_REGION" --stack-name "$RFD4_STACK" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" \
        --output text
}
RFD4_BUCKET="$(stack_output BucketName)"
RFD4_ECR_REPOSITORY="$(stack_output RepositoryName)"
RFD4_ECR_URI="$(stack_output RepositoryUri)"
RFD4_RUN_ROLE_ARN="$(stack_output RunRoleArn)"
RFD4_RUN_GROUP_ID="$(stack_output RunGroupId)"
export RFD4_BUCKET RFD4_ECR_REPOSITORY RFD4_ECR_URI
export RFD4_RUN_ROLE_ARN RFD4_RUN_GROUP_ID
for value in "$RFD4_BUCKET" "$RFD4_ECR_REPOSITORY" "$RFD4_ECR_URI" \
             "$RFD4_RUN_ROLE_ARN" "$RFD4_RUN_GROUP_ID"; do
    test -n "$value"
    test "$value" != None
done
```

Stop if deployment fails or any output is empty/`None`. Diagnose the stack error
before continuing. If you use pre-existing resources instead, export these same
variables and apply equivalent S3 access, ECR repository permissions and run-role
trust/permissions; see [HealthOmics service roles](https://docs.aws.amazon.com/omics/latest/dev/permissions-service.html)
and [ECR permissions](https://docs.aws.amazon.com/omics/latest/dev/permissions-ecr.html).

## 3. Download the model and prepare the first input

Download both model files and their LICENSE from the same pinned bundle. It
contains the matching flow model and autoencoder. The model is staged as input
data rather than included in the image, so GPU tasks do not download weights
from Hugging Face.

```bash
hf download nvidia/RFD4-Proteina config.json model.safetensors LICENSE \
    --revision a06e2509c7033b010fc43614e9fd840303588502 \
    --local-dir "$RFD4_WORK_DIR/model"
cp "$RFD4_ROOT/examples/healthomics/settings.json" "$RFD4_WORK_DIR/settings.json"
test -s "$RFD4_WORK_DIR/model/config.json"
test -s "$RFD4_WORK_DIR/model/model.safetensors"
test -s "$RFD4_WORK_DIR/model/LICENSE"
```

The supplied settings use seed 42, one replicate, batch size one, the explicit
medium/front-loaded recycling schedule and no length-bucket compilation. Keep
these settings for the first smoke test. The input is the repository's
`data/benchmarks/monomer/monomer-short.cif.zst` example.

## 4. Build with Finch and push the image

The first image uses the repository's CUDA 12 build. The thin HealthOmics layer
adds `procps`, because Nextflow task monitoring needs `ps` before Python starts.
The help command checks CLI startup; the managed GPU run below tests inference.

```bash
cd "$RFD4_ROOT"
finch build --platform linux/amd64 --output type=image \
    --build-arg RFD4_ACCELERATOR=cu12 \
    -t "rfd4-proteina:cuda12-$RFD4_TAG" .
finch build --platform linux/amd64 --output type=image \
    -f examples/healthomics/Containerfile \
    --build-arg "RFD4_IMAGE=rfd4-proteina:cuda12-$RFD4_TAG" \
    -t "$RFD4_ECR_URI:$RFD4_TAG" .
finch run --rm --platform linux/amd64 "$RFD4_ECR_URI:$RFD4_TAG" \
    rfd4-generate --help
aws ecr get-login-password --region "$AWS_REGION" | \
    finch login --username AWS --password-stdin "${RFD4_ECR_URI%%/*}"
finch push "$RFD4_ECR_URI:$RFD4_TAG"
RFD4_IMAGE_DIGEST="$(aws ecr describe-images \
    --region "$AWS_REGION" --repository-name "$RFD4_ECR_REPOSITORY" \
    --image-ids "imageTag=$RFD4_TAG" \
    --query 'imageDetails[0].imageDigest' --output text)"
test "${#RFD4_IMAGE_DIGEST}" -eq 71
test "${RFD4_IMAGE_DIGEST%%:*}" = sha256
export RFD4_IMAGE_URI="$RFD4_ECR_URI@$RFD4_IMAGE_DIGEST"
```

Use a new tag for each rebuild: the repository deliberately rejects overwriting
a tag. Runs use the immutable digest. Review your account's image vulnerability
scan findings before production use; this example does not change registry-wide
scanning configuration.

The workstation/build environment needs approved access to the public base image
and package/model registries. This is separate from the restricted execution
network used by the HealthOmics run.

## 5. Upload only the required input files

All URIs below use the stack's dedicated bucket in the workflow Region. Upload
only the required files, use a fresh model prefix, and verify each object before
starting GPU compute. An upload command's exit status alone is not the acceptance
check: this procedure checks the remote size and server-validated SHA-256.

The pinned bundle fits S3's single-PUT limit. For larger replacement files, use
a multipart upload procedure with corresponding integrity verification.
If workstation bandwidth is limited, run the staging step on an approved AWS
host with the required tools and role. Keep the same checksum checks; the
full-size PUT in the reference validation was exercised in `us-east-1`.

```bash
python3 - <<'PYTHON'
import base64, hashlib, json, os, subprocess
from pathlib import Path

work = Path(os.environ["RFD4_WORK_DIR"])
bucket, tag = os.environ["RFD4_BUCKET"], os.environ["RFD4_TAG"]
files = [
    (work / "model/config.json", f"models/rfd4/{tag}/config.json"),
    (work / "model/model.safetensors", f"models/rfd4/{tag}/model.safetensors"),
    (work / "model/LICENSE", f"models/rfd4/{tag}/LICENSE"),
    (Path(os.environ["RFD4_ROOT"]) / "data/benchmarks/monomer/monomer-short.cif.zst",
     "inputs/monomer-short.cif.zst"),
    (work / "settings.json", f"inputs/{tag}/settings.json"),
]

def aws_json(*arguments):
    result = subprocess.run(
        ["aws", "--region", os.environ["AWS_REGION"], *arguments],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(result.stderr)
    return json.loads(result.stdout)

receipts = []
for local, key in files:
    size = local.stat().st_size
    if not 0 < size <= 5 * 1024**3:
        raise SystemExit(f"Use a verified multipart upload for {local}")
    with local.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").digest()
    checksum = base64.b64encode(digest).decode()
    aws_json("s3api", "put-object", "--bucket", bucket, "--key", key,
             "--body", str(local), "--checksum-algorithm", "SHA256",
             "--checksum-sha256", checksum)
    head = aws_json("s3api", "head-object", "--bucket", bucket, "--key", key,
                    "--checksum-mode", "ENABLED")
    if head["ContentLength"] != size or head.get("ChecksumSHA256") != checksum:
        raise SystemExit(f"Remote object verification failed: {key}")
    receipts.append({"key": key, "bytes": size, "sha256": digest.hex(),
                     "version_id": head.get("VersionId")})
    print(f"Verified {local.name}: {size} bytes, SHA-256 matches")
(work / "upload-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
PYTHON
```

## 6. Register the private workflow

Package the workflow from its own directory. The subshell preserves your current
working directory. Weights, credentials, run outputs and the bootstrap template
are not included in the workflow archive.

```bash
(
    cd "$RFD4_ROOT/examples/healthomics"
    zip -r "$RFD4_WORK_DIR/workflow.zip" \
        main.nf nextflow.config conf bin parameter-template.json \
        -x '*/__pycache__/*'
)
RFD4_WORKFLOW_ID="$(aws omics create-workflow \
    --region "$AWS_REGION" --name "rfd4-generation-$RFD4_TAG" \
    --engine NEXTFLOW --accelerators GPU --storage-type DYNAMIC \
    --definition-zip "fileb://$RFD4_WORK_DIR/workflow.zip" --main main.nf \
    --parameter-template "file://$RFD4_ROOT/examples/healthomics/parameter-template.json" \
    --query id --output text)"
export RFD4_WORKFLOW_ID
aws omics get-workflow --region "$AWS_REGION" --id "$RFD4_WORKFLOW_ID" \
    --query '{id:id,status:status}' --output table
```

Repeat `get-workflow` until the status is `ACTIVE`. Do not start a run while the
workflow is still being created. If creation fails, inspect the full response
and definition before submitting again. A private workflow is not an AWS-managed
Ready2Run workflow.

## 7. Write parameters and start the smoke run

The following command writes an actual `run-parameters.json` file. It explicitly
selects L4 and four requested CPUs for the tested short input; the workflow's
compatibility default is L40S. A 32 GB host-memory request maps this L4 task to
`omics.g6.2xlarge` even with four requested CPUs.

```bash
python3 - <<'PYTHON'
import json, os
from pathlib import Path
bucket = os.environ["RFD4_BUCKET"]
tag = os.environ["RFD4_TAG"]
parameters = {
    "input_cif": f"s3://{bucket}/inputs/monomer-short.cif.zst",
    "model_bundle": f"s3://{bucket}/models/rfd4/{tag}/",
    "settings": f"s3://{bucket}/inputs/{tag}/settings.json",
    "container_image": os.environ["RFD4_IMAGE_URI"],
    "num_shards": "1",
    "generation_cpus": "4",
    "generation_accelerator": "nvidia-l4"
}
Path(os.environ["RFD4_WORK_DIR"], "run-parameters.json").write_text(
    json.dumps(parameters, indent=2) + "\n"
)
PYTHON
RFD4_RUN_ID="$(aws omics start-run \
    --region "$AWS_REGION" --name "rfd4-smoke-$RFD4_TAG" \
    --workflow-id "$RFD4_WORKFLOW_ID" \
    --role-arn "$RFD4_RUN_ROLE_ARN" --run-group-id "$RFD4_RUN_GROUP_ID" \
    --parameters "file://$RFD4_WORK_DIR/run-parameters.json" \
    --output-uri "s3://$RFD4_BUCKET/outputs/$RFD4_TAG/" \
    --storage-type DYNAMIC --networking-mode RESTRICTED \
    --engine-settings '{"engineVersion":"25.10.0","profile":"healthomics"}' \
    --log-level ALL --query id --output text)"
export RFD4_RUN_ID
```

## 8. Monitor the run and inspect its output

A run can be `RUNNING` while its GPU task is still `STARTING`. Check task status
to distinguish provisioning from model processing. Provisioning time varies;
record it separately from inference time.

```bash
aws omics get-run --region "$AWS_REGION" --id "$RFD4_RUN_ID" \
    --query '{id:id,status:status,reason:statusMessage,output:runOutputUri}' --output table
aws omics list-run-tasks --region "$AWS_REGION" --id "$RFD4_RUN_ID" \
    --query 'items[].{task:taskId,name:name,status:status,instance:instanceType}' --output table
```

Wait for `COMPLETED` before treating the run as successful. For `FAILED` or
`CANCELLED`, inspect `statusMessage`, `failureReason` and the CloudWatch task logs
in `/aws/omics/WorkflowLog`. Failed tasks do not necessarily publish their files.

```bash
test "$(aws omics get-run --region "$AWS_REGION" --id "$RFD4_RUN_ID" \
    --query status --output text)" = COMPLETED
RFD4_RESULT_URI="$(aws omics get-run --region "$AWS_REGION" --id "$RFD4_RUN_ID" \
    --query runOutputUri --output text)"
export RFD4_RESULTS="$RFD4_WORK_DIR/results/$RFD4_RUN_ID"
aws s3 sync "${RFD4_RESULT_URI%/}/" "$RFD4_RESULTS/" --region "$AWS_REGION"
python3 - <<'PYTHON'
import json, os
from pathlib import Path
root = Path(os.environ["RFD4_RESULTS"])
summary = json.loads((root / "pubdir/summary/summary.json").read_text())
assert summary["status"] == "COMPLETED"
assert summary["generated_designs"] == 1
assert summary["shards"][0]["native_seed"] == 42
print("Verified smoke inventory:", summary["generated_designs"], "generated structure")
print("Structures:", root / "pubdir/shards")
print("Execution report:", root / "output/rfd4-execution-report.html")
PYTHON
```

The assertions above are for the supplied one-design smoke settings. Inspect the
CIF and relevant metrics for your scientific task as well. File/inventory checks
do not establish biological designability. See the
[input inspection helpers](../../skills/rfd4-proteina-annotated-cif/SKILL.md) and
[metrics guide](../../docs/how-to/evaluate-with-metrics.md).

## Troubleshooting and resource lifecycle

| Symptom | Check |
| --- | --- |
| ECR push fails after a rebuild | Use a new tag and refresh the ECR login. Tags are immutable. |
| ECR API times out during login | Check DNS from the actual builder. An existing VPC endpoint may route the request to a security group or policy that does not allow that builder. |
| Finch reports a terminal error after an AWS CLI error | Resolve the preceding AWS authentication/network error; an empty password stream can produce a secondary Finch error. |
| Finch fails because `HOME` is missing in a headless session | Run it in a normal login environment for the intended OS user. Do not insert credentials into the environment setup. |
| Build runs out of disk space | Check the Finch data disk, including existing images and build cache. Use an adequately sized build environment and preserve existing images. |
| Run cannot read S3 or pull the image | Confirm the Region, run role, object prefixes and ECR repository policy. |
| `ps` is missing | Build the thin HealthOmics Containerfile and use its image digest. |
| CUDA preflight fails | Check the accelerator request and CUDA image; a local Mac help check is not GPU validation. |
| GPU task stays `STARTING` | Inspect provisioning status and group limits; this is not a model throughput measurement. |
| Run times out | The validation group's duration includes workflow/provisioning time. Adjust limits deliberately for a larger run. |
| Missing/corrupt output or seed mismatch | The workflow fails rather than publishing a successful inventory. Inspect the task log. |
| VPC endpoint access fails | Verify private DNS, endpoint security groups, IAM/endpoint policies and the client network; see [networking](networking.md). |

To deliberately stop an active test, use `aws omics cancel-run --region
"$AWS_REGION" --id "$RFD4_RUN_ID"`. Do not delete the run group or execution role
while a run is active. The group itself does not keep GPU instances running.

Model files, versions, results and container images remain stored after a run.
The bootstrap template retains the bucket, bucket policy and ECR repository when
the stack is deleted. Treat deletion of retained data/images as a separate,
intentional cleanup operation; follow your organization's retention policy.

## Customize inputs and sampling

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


## Resource selection

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

An additional control used the same input and seed 79 for this wrapper and two
independent native CLI executions on L4. The parsed atom identities and coordinates
were identical; effective configurations matched after normalizing output paths
and excluding run-name metadata. This is evidence for that fixed configuration,
not a guarantee of identical samples across hardware or software changes.

The generated CIFs retain warnings for unspecified formal charges and missing
`_struct_conn`; no oracle refolding was performed. This validates the short
monomer example, not larger inputs, larger batches or other GPU types.

Additional validation on 2026-10-25 deployed `bootstrap.yaml` into new
`us-east-1` resources and read back the actual S3/ECR security settings and run
group limits. A fresh CUDA 12 image was built with Finch 1.19.0 on Linux, then
used for a one-design L4 run submitted through PrivateLink. The resulting
100-residue structure passed the repository's full annotated-CIF checker with
the existing formal-charge and `_struct_conn` warnings retained.
The same new resources also completed three designs in VPC networking mode,
using two shards, batch size two and base seed 61. Native inventories confirmed
61/61/62, transferred output hashes matched, and all three structures passed the
full checker with 100 residues and finite coordinates.
An independent seed-79 control on this fresh image compared one wrapper run
with two direct CLI runs in VPC mode. All three had the same 856 atom identities,
matching normalized effective configurations and a maximum raw coordinate
difference of 0.0 Angstrom. This remains a result for the tested configuration,
not a cross-hardware determinism guarantee.
The resource defaults were also exercised separately: one L40S generation task
on `omics.g6e.2xlarge`, eight requested CPUs and 32 GB requested host memory
produced one design with native seed 42 and verified file hashes. That task
spent about 35 minutes preparing and about 95 seconds running; the helper
recorded about 58 seconds. These single-run timings include different stages
and do not establish a GPU throughput or cost optimum.
Its 100-residue output also passed the full annotated-CIF checker, retaining
the same formal-charge and `_struct_conn` warning categories.

This validation caught a missing weights object despite an earlier upload
command returning success. That failed GPU run was retained. The model was
restaged into a fresh prefix and verified by remote size and full SHA-256 before
the successful retry. The guide now makes those checks an explicit prerequisite.
The full-size checksum PUT was also exercised from a host in the workflow
Region; workstation upload bandwidth is a separate consideration.

The [networking guide](networking.md) records the actual private API, S3
permission, private-service, TCP and DNS tests. These are bounded execution and
access-control checks using public benchmark data and synthetic canaries, not
an assessment of biological efficacy or a production security certification.

Run the standalone tests from the repository root without installing the model:

```bash
python3 tests/test_healthomics_workflow.py -v
```

When Nextflow 25.10.0 is on `PATH`, the same test runs the actual two-shard
workflow in stub mode using clearly marked fixtures. No Docker, GPU, model
weights or AWS account is needed for this test. Stub reports have status `STUB`
and zero generated designs; they are never reported as successful model runs.

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
