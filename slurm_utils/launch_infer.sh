#!/bin/bash
# =============================================================================
# launch_infer.sh - Cluster launch for inference and offline validation
# =============================================================================
# Submits generate.py as a SLURM job (optionally a job array for parallel
# execution) and automatically queues a dependent concatenation job to merge
# per-job CSV metrics into a single file once generation finishes.
#
# Usage:
#   ./launch_infer.sh <cluster> [options] [hydra_overrides...]
#
# Positional:
#   <cluster>               Cluster name (e.g. digs, ord, gb200)
#
# Options:
#   --njobs N               Parallel generation jobs (default: reads gen_njobs
#                           from Hydra config, fallback 1)
#   --gpus-per-job N        GPUs per SLURM array task (default: 1)
#   --config-name NAME      Hydra config name (default: inference). May be a
#                           path relative to configs/, e.g.
#                           sweep_configs/my_sweep/inf_000
#   --no-concatenate        Skip the automatic post-generation concatenation job
#   --foldseek              Enable Foldseek clustering during concatenation
#   --no-copy               Use the current repo in-place (skip rsync)
#   --overwrite-rundir      Allow overwriting an existing code directory
#   --follow, -f            Follow logs after submission (Ctrl+C to exit)
#
# Hydra overrides:
#   Any remaining arguments are passed directly to generate.py. Examples:
#     run_name_=my_run
#     out_dir=/path/to/outputs       Controls where outputs are saved
#     ckpt_path=/path/to/ckpts ckpt_name=model.ckpt
#     num_replicates=10
#     inference_experiment=inference_on_val_set
#     +datasets/val@generation=mcsa_subset
#
# Concatenation:
#   By default a dependent SLURM job is submitted that runs
#   concatenate_external_eval_results.py after all generation tasks finish.
#   Results land in <run_dir>/inference_metrics/.
#   Use --no-concatenate to skip, --foldseek to add structural clustering.
#
# Offline validation:
#   Pass inference_experiment=inference_on_val_set plus the dataset override
#   +datasets/val@generation=<name>. See the README for full examples.
#
# Examples:
#   # Parallel inference with 8 jobs
#   ./launch_infer.sh digs --njobs 8 \
#       ckpt_path=/path/to/ckpts ckpt_name=model.ckpt \
#       inputs=benchmarks/monomer.json run_name_=my_run
#
#   # Offline validation with Foldseek clustering
#   ./launch_infer.sh digs --njobs 8 --foldseek \
#       ckpt_path=/path/to/ckpts ckpt_name=model.ckpt \
#       inference_experiment=inference_on_val_set \
#       +datasets/val@generation=mcsa_subset run_name_=mcsa_val
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ============================================================
# Argument parsing
# ============================================================
CLUSTER=""
NJOBS=""               # empty = read from config's gen_njobs
GPUS_PER_JOB=1
CONFIG_NAME="inference"
DO_CONCATENATE="true"  # concatenate by default
CONCATENATE_FOLDSEEK="false"
COPY_CODE="true"
FOLLOW_LOGS="false"
OVERWRITE_RUNDIR="false"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --njobs)
            NJOBS="$2"
            shift 2
            ;;
        --gpus-per-job=*)
            GPUS_PER_JOB="${1#*=}"
            shift
            ;;
        --gpus-per-job)
            GPUS_PER_JOB="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --no-concatenate)
            DO_CONCATENATE="false"
            shift
            ;;
        --foldseek)
            CONCATENATE_FOLDSEEK="true"
            shift
            ;;
        --no-copy)
            COPY_CODE="false"
            shift
            ;;
        --overwrite-rundir)
            OVERWRITE_RUNDIR="true"
            shift
            ;;
        --follow|-f)
            FOLLOW_LOGS="true"
            shift
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

# ============================================================
# Validate arguments
# ============================================================
if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--njobs N] [--gpus-per-job N] [options] [hydra_overrides...]"
    echo "Available clusters: digs, ord, gb200"
    echo "See script header for full documentation."
    exit 1
fi

# ============================================================
# Load cluster configuration
# ============================================================
load_user_info "$CLUSTER" || exit 1
print_cluster_config

# Find repository root
REPO_ROOT=$(find_repo_root "$SCRIPT_DIR")
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find .project-root"
    exit 1
fi
cd "$REPO_ROOT"
echo "Repository root: $REPO_ROOT"

# Setup PYTHONPATH
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# ============================================================
# Resolve Hydra config
# ============================================================
# When CONFIG_NAME contains path separators, split into --config-path + --config-name
# so Hydra sees only the bare filename and places the config at the root level.
CONFIGS_ROOT="$REPO_ROOT/projects/proteinfoundation/configs"
if [[ "$CONFIG_NAME" == */* ]]; then
    CONFIG_SUBDIR="${CONFIG_NAME%/*}"
    CONFIG_BASE="${CONFIG_NAME##*/}"
    # If CONFIG_SUBDIR is absolute, use it directly,
    # otherwise treat it as relative to the repo's configs root.
    if [[ "$CONFIG_SUBDIR" == /* ]]; then
        CONFIG_HYDRA_ARGS="--config-path $CONFIG_SUBDIR --config-name $CONFIG_BASE"
    else
        CONFIG_HYDRA_ARGS="--config-path $CONFIGS_ROOT/$CONFIG_SUBDIR --config-name $CONFIG_BASE"
    fi
else
    CONFIG_BASE="$CONFIG_NAME"
    CONFIG_HYDRA_ARGS="--config-name $CONFIG_NAME"
fi

print_header "Resolving Hydra Config"
echo "  Config name: $CONFIG_NAME"
echo "  Extra args:  $EXTRA_ARGS"

eval $(resolve_hydra_config "$REPO_ROOT" "$CONFIG_NAME" $EXTRA_ARGS)

# Validate that run_name_ was resolved
if [ -z "$RUN_NAME" ]; then
    echo ""
    echo "Error: run_name_ was not found in the resolved config."
    echo "  Pass run_name_=<name> as an argument, or set run_name_: in your config."
    exit 1
fi

# ============================================================
# Determine number of parallel jobs
# ============================================================
# Priority: --njobs CLI flag > gen_njobs from resolved config > 1
if [ -z "$NJOBS" ]; then
    NJOBS="${GEN_NJOBS:-1}"
fi

# Calculate SLURM array task count
NUM_ARRAY_TASKS=$(( (NJOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

# ============================================================
# Set up run directory
# ============================================================
if [ -n "$OUT_DIR" ]; then
    RUN_DIR="$OUT_DIR"
else
    RUN_DIR="$ROOT_REMOTE/inference/${RUN_NAME}"
fi
RUN_DIR="${RUN_DIR%/}"

# generate.py writes outputs directly into RUN_DIR (as out_dir);
# concatenation looks for inference_metrics/ within that same directory.
GENERATION_OUTPUT_DIR="$RUN_DIR"

mkdir -p "$RUN_DIR/slurm_run_outputs"

# Clear old SLURM logs so that a re-run starts fresh
rm -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err"

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/${RUN_NAME}"

# Create symlink in local repo for easy browsing without SSH
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/${RUN_NAME}"
echo "  Local symlink: $REPO_ROOT/runs/${RUN_NAME}"

# ============================================================
# Copy code to run directory (or use repo in-place)
# ============================================================
if [ "$COPY_CODE" == "true" ]; then
    CODE_DIR="$RUN_DIR/code"
    if [ -d "$CODE_DIR/src" ] || [ -d "$CODE_DIR/projects/proteinfoundation/proteinfoundation" ]; then
        if [ "$OVERWRITE_RUNDIR" != "true" ]; then
            echo "Error: Code directory already exists at $CODE_DIR"
            echo "  To overwrite, use --overwrite-rundir"
            echo "  To use existing code, use --no-copy"
            exit 1
        fi
        echo "Warning: Overwriting existing code directory at $CODE_DIR"
    fi

    print_header "Copying Code"
    echo "  $REPO_ROOT -> $CODE_DIR"

    rsync -a "$REPO_ROOT/src/" "$CODE_DIR/src/"
    mkdir -p "$CODE_DIR/models/rf3" && rsync -a "$REPO_ROOT/models/rf3/src/" "$CODE_DIR/models/rf3/src/"
    mkdir -p "$CODE_DIR/lib/atomworks" && rsync -a "$REPO_ROOT/lib/atomworks/src/" "$CODE_DIR/lib/atomworks/src/"

    echo "  Copying projects/proteinfoundation..."
    mkdir -p "$CODE_DIR/projects/proteinfoundation"
    rsync -a --exclude='store' --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' \
        "$REPO_ROOT/projects/proteinfoundation/proteinfoundation/" "$CODE_DIR/projects/proteinfoundation/proteinfoundation/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/configs/" "$CODE_DIR/projects/proteinfoundation/configs/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/benchmarks/" "$CODE_DIR/projects/proteinfoundation/benchmarks/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/data/" "$CODE_DIR/projects/proteinfoundation/data/"

    cp "$REPO_ROOT/.project-root" "$CODE_DIR/"
    mkdir -p "$CODE_DIR/.ipd/shebang" && rsync -a "$REPO_ROOT/.ipd/shebang/" "$CODE_DIR/.ipd/shebang/"
    rsync -a "$REPO_ROOT/.ipd/".env.* "$CODE_DIR/.ipd/"
    mkdir -p "$CODE_DIR/.ipd/apptainer"

    # Apptainer: recreate container symlink with /mnt/net prefix for compute nodes
    if [ "$CONTAINER_TYPE" == "apptainer" ] && [ -L "$REPO_ROOT/.ipd/apptainer/latent-dev.sif" ]; then
        CONTAINER_TARGET="$(readlink -f "$REPO_ROOT/.ipd/apptainer/latent-dev.sif" 2>/dev/null || true)"
        if [ -n "$CONTAINER_TARGET" ]; then
            [[ "$CONTAINER_TARGET" == /net/* ]] && CONTAINER_TARGET="/mnt${CONTAINER_TARGET}"
            ln -sf "$CONTAINER_TARGET" "$CODE_DIR/.ipd/apptainer/latent-dev.sif"
        fi
    fi
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

# ============================================================
# Create .env file
# ============================================================
write_env_file "$RUN_DIR/.env"

# ============================================================
# Print configuration summary
# ============================================================
print_header "Job Configuration"
echo "  Run name:         $RUN_NAME"
echo "  Run directory:    $RUN_DIR"
echo "  Output directory: $GENERATION_OUTPUT_DIR"
echo "  Parallel jobs:    $NJOBS"
echo "  GPUs per job:     $GPUS_PER_JOB"
echo "  SLURM array tasks: $NUM_ARRAY_TASKS"
echo "  CPUs per task:    $CPUS_PER_TASK"
echo "  Concatenate:      $DO_CONCATENATE"
[ "$CONCATENATE_FOLDSEEK" == "true" ] && echo "  Foldseek:         true"
echo ""

# ============================================================
# Build main generation SBATCH script
# ============================================================
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES=1
GPUS_PER_NODE="$GPUS_PER_JOB"
NTASKS_PER_NODE=1
build_sbatch_header > "$SBATCH_FILE"

# Array directive if multiple tasks are needed
if [ "$NUM_ARRAY_TASKS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NUM_ARRAY_TASKS - 1))" >> "$SBATCH_FILE"
fi

# Output/error paths and common environment
cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err

# Environment setup (all clusters)
export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/projects/proteinfoundation:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Job array configuration
ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
NJOBS=$NJOBS

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB, total jobs: \$NJOBS"
EOF

# Apptainer clusters: pass cluster identity to exec_core.sh (via shebang).
# Static env vars (NCCL, PyTorch) come from .ipd/.env.${CLUSTER} via exec_core.sh.
if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    cat >> "$SBATCH_FILE" << EOF

# Cluster identity — read by exec_core.sh (shebang) to source .ipd/.env.\${CLUSTER}
export CLUSTER=$CLUSTER
export CONTAINER_TYPE=$CONTAINER_TYPE
EOF
fi

# Optional mounts
if [ -n "$DATASET_SQUASH_MOUNTS" ]; then
    echo "export DATASET_SQUASH_MOUNTS=\"$DATASET_SQUASH_MOUNTS\"" >> "$SBATCH_FILE"
fi
if [ -n "$USE_CUEQUIVARIANCE" ]; then
    echo "export SHOULD_USE_CUEQUIVARIANCE=$USE_CUEQUIVARIANCE" >> "$SBATCH_FILE"
fi

# DDP coordination (unique port per job, master = first node)
cat >> "$SBATCH_FILE" << 'EOF'

export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Verify scontrol works (needed for auto-requeue)
scontrol show job $SLURM_JOB_ID > /dev/null 2>&1 && echo '[slurm] scontrol works - auto-requeue enabled' || echo '[slurm] WARNING: scontrol not working - auto-requeue disabled'
EOF

# Platform-specific generation command
SCRIPT_GENERATE="$CODE_DIR/projects/proteinfoundation/proteinfoundation/generate.py"

if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    # Apptainer: shebang in generate.py handles container execution via exec_core.sh
    cat >> "$SBATCH_FILE" << 'EOF'

# Run generation - each GPU gets its own job_id slice of the dataset
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running generation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none \\
            --kill-on-bad-exit $SCRIPT_GENERATE \\
            $CONFIG_HYDRA_ARGS \\
            $EXTRA_ARGS \\
            out_dir=$RUN_DIR \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
else
    # OCI/GB200: Docker/Pyxis container via srun --container-image
    CONTAINER_CMD=$(get_srun_cmd)
    CONTAINER_MOUNTS_STR=$(build_container_mounts_string)
    CONTAINER_ENV_CMDS=$(build_container_env_cmds)
    cat >> "$SBATCH_FILE" << 'EOF'

# Run generation - each GPU gets its own job_id slice of the dataset
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running generation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none --kill-on-bad-exit $CONTAINER_CMD \\
            --container-mounts="$CONTAINER_MOUNTS_STR" \\
            --container-workdir=$RUN_DIR \\
            bash -c "$CONTAINER_ENV_CMDS && $PYTHON $SCRIPT_GENERATE \\
            $CONFIG_HYDRA_ARGS \\
            $EXTRA_ARGS \\
            out_dir=$RUN_DIR \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            job_id=\\\$JOB_ID \\
            gen_njobs=\\\$NJOBS" &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
fi

# ============================================================
# Build concatenation SBATCH script (if enabled)
# ============================================================
# The concat job runs concatenate_external_eval_results.py after all generation
# jobs finish. It is submitted with --dependency=afterany:<gen_job_id> so the
# launch script never has to wait.
if [ "$DO_CONCATENATE" == "true" ]; then
    CONCAT_SBATCH_FILE="$RUN_DIR/slurm_concat_script.sh"
    CONCAT_SCRIPT="$CODE_DIR/projects/proteinfoundation/proteinfoundation/inference_utils/concatenate_external_eval_results.py"

    # Metrics CSVs are written by generate.py under <out_dir>/inference_metrics/
    CONCAT_DIR="$GENERATION_OUTPUT_DIR/inference_metrics"

    FOLDSEEK_FLAG=""
    if [ "$CONCATENATE_FOLDSEEK" == "true" ]; then
        FOLDSEEK_FLAG="--cluster-with-foldseek"
    fi

    # Build concat sbatch header (CPU-only, minimal resources)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name concat-${RUN_NAME}"
        if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
            echo "#SBATCH --account $ACCOUNT"
        fi
        echo "#SBATCH --partition ${PARTITION:-batch}"
        echo "#SBATCH --time 01:00:00"
        echo "#SBATCH --nodes 1"
        echo "#SBATCH --ntasks-per-node 1"
        echo "#SBATCH --mem 16g"
        echo "#SBATCH --output ./slurm_run_outputs/slurm_concat.out"
        echo "#SBATCH --error ./slurm_run_outputs/slurm_concat.err"
        echo "#SBATCH --open-mode append"
        echo ""
        echo "export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/projects/proteinfoundation:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH"
        echo "export PYTHONFAULTHANDLER=1"
        echo ""
        echo "echo \"Running result concatenation for: $RUN_NAME\""
        echo "echo \"Concat dir: $CONCAT_DIR\""
    } > "$CONCAT_SBATCH_FILE"

    if [ "$CONTAINER_TYPE" == "apptainer" ]; then
        # Apptainer: shebang in concatenate_external_eval_results.py handles container
        cat >> "$CONCAT_SBATCH_FILE" << EOF
# Cluster identity — read by exec_core.sh (shebang) to source .ipd/.env.\${CLUSTER}
export CLUSTER=$CLUSTER
export CONTAINER_TYPE=$CONTAINER_TYPE

"$CONCAT_SCRIPT" --output-dir "$CONCAT_DIR" $FOLDSEEK_FLAG
EOF
    else
        CONTAINER_CMD_CONCAT=$(get_srun_cmd)
        CONTAINER_MOUNTS_STR_CONCAT=$(build_container_mounts_string)
        CONTAINER_ENV_CMDS_CONCAT=$(build_container_env_cmds)
        cat >> "$CONCAT_SBATCH_FILE" << EOF
srun --ntasks=1 $CONTAINER_CMD_CONCAT \\
    --container-mounts="$CONTAINER_MOUNTS_STR_CONCAT" \\
    --container-workdir=$RUN_DIR \\
    bash -c "$CONTAINER_ENV_CMDS_CONCAT && $PYTHON \"$CONCAT_SCRIPT\" --output-dir \"$CONCAT_DIR\" $FOLDSEEK_FLAG"
EOF
    fi

    cat >> "$CONCAT_SBATCH_FILE" << 'EOF'
echo "Concatenation complete!"
EOF
fi

# ============================================================
# Submit jobs
# ============================================================
print_header "Submitting Job"
echo "  SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

# Warn if a job with this name is already running (avoids silent duplicates)
EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' is already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID"
    echo "  Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
    read -p "Continue with submission? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

if [ -n "$WANDBKEY" ]; then
    export WANDB_API_KEY="$WANDBKEY"
fi

# Submit main generation job
SBATCH_OUTPUT=$(sbatch slurm_batch_script.sh)
MAIN_JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')
echo "$SBATCH_OUTPUT"

# Submit concatenation job with dependency (non-blocking)
CONCAT_JOB_ID=""
if [ "$DO_CONCATENATE" == "true" ] && [ -n "$MAIN_JOB_ID" ]; then
    echo ""
    echo "Submitting concatenation job (waits for job $MAIN_JOB_ID to finish)..."
    CONCAT_SBATCH_OUTPUT=$(sbatch --dependency=afterany:$MAIN_JOB_ID slurm_concat_script.sh)
    CONCAT_JOB_ID=$(echo "$CONCAT_SBATCH_OUTPUT" | grep -oP '\d+')
    echo "$CONCAT_SBATCH_OUTPUT"
fi

# ============================================================
# Print monitoring information
# ============================================================
echo ""
echo "=============================================="
echo "Jobs submitted"
echo "=============================================="
echo "  Monitor:     squeue -u $USER --name=$RUN_NAME"
echo "  Gen logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
if [ -n "$CONCAT_JOB_ID" ]; then
    echo "  Concat job:  squeue -j $CONCAT_JOB_ID"
    echo "  Concat logs: tail -f $RUN_DIR/slurm_run_outputs/slurm_concat.out"
    echo "  Results:     $GENERATION_OUTPUT_DIR/inference_metrics/"
fi
echo ""

# Follow logs if requested (tail until Ctrl+C)
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "Following logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
fi
