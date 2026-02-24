#!/bin/bash
# =============================================================================
# launch_offline_val.sh - Direct cluster launch for offline validation
# =============================================================================
# Run this script directly on the cluster for offline validation
#
# Usage:
#   ./launch_offline_val.sh <cluster> [options] [extra_args...]
#
# Options:
#   --no-copy          Skip copying code to run directory (use current repo)
#   --overwrite-rundir Allow overwriting existing code directory in run directory
#   --follow, -f       Follow log output after job submission (Ctrl+C to exit)
#   --no-cleanup       Don't wait for jobs to complete (skip concatenation/foldseek)
#   --ckpt-path <path> Path to the checkpoint directory (required)
#   --ckpt-name <name> Name of the checkpoint file (required)
#   --njobs <n>        Number of parallel jobs (default: 1)
#   --gpus-per-job <n> GPUs per job (default: cluster default)
#
# Required Hydra overrides (via extra_args):
#   run_name_=<name>                        Unique name for this validation run
#   +datasets/val@generation=<val_set>      Name of the validation set to use
#
# Prohibited overrides:
#   out_dir=...                            Output directory is managed by this script
#   hardware.ngpus_per_node_=...           Control using --gpus-per-job option
#   hardware.nnodes_=...                   To use multiple nodes, adjust --njobs
#   ckpt_path=...                          Use --ckpt-path instead
#   ckpt_name=...                          Use --ckpt-name instead
#
# Examples:
#   ./launch_offline_val.sh digs \
#       --ckpt-path /path/to/checkpoints/ \
#       --ckpt-name "model_epoch=00039_step=080000.ckpt" \
#       +datasets/val@generation=mcsa_full \
#       run_name_=my_validation_run
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
NJOBS=1
GPUS_PER_JOB=""  # Default will be set after loading cluster config
COPY_CODE="true"
FOLLOW_LOGS="false"
OVERWRITE_RUNDIR="false"
WAIT_FOR_COMPLETION="true"  # Enables concatenation and foldseek
CKPT_PATH=""
CKPT_NAME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-copy)
            COPY_CODE="false"
            shift
            ;;
        --overwrite-rundir)
            OVERWRITE_RUNDIR="true"
            shift
            ;;
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
        --follow|-f)
            FOLLOW_LOGS="true"
            shift
            ;;
        --no-cleanup)
            WAIT_FOR_COMPLETION="false"
            shift
            ;;
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --ckpt-name)
            CKPT_NAME="$2"
            shift 2
            ;;
        -*)
            EXTRA_ARGS="$EXTRA_ARGS $1"
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

if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> --ckpt-path <path> --ckpt-name <name> [options] [extra_args...]"
    echo "Available clusters: digs, ord, gb200"
    exit 1
fi

# Validate required arguments
if [ -z "$CKPT_PATH" ] || [ -z "$CKPT_NAME" ]; then
    echo "Error: --ckpt-path and --ckpt-name are required"
    echo "Usage: $0 <cluster> --ckpt-path <path> --ckpt-name <name> [options] [extra_args...]"
    exit 1
fi

# Load cluster configuration
load_user_info "$CLUSTER" || exit 1
print_cluster_config

# Set default GPUs per job from cluster config if not specified
if [ -z "$GPUS_PER_JOB" ]; then
    GPUS_PER_JOB="${GPUS_PER_NODE_CLUSTER:-8}"
fi

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

# Resolve configuration using Hydra (handles composition, interpolations, overrides)
# This properly resolves defaults, experiment configs, and command-line overrides
CONFIG_NAME="inference"
print_header "Resolving Hydra Config"
echo "  Config name: $CONFIG_NAME"
echo "  Extra args: $EXTRA_ARGS"

eval $(resolve_hydra_config "$REPO_ROOT" "$CONFIG_NAME" $EXTRA_ARGS)

# Validate required values
if [ -z "$RUN_NAME" ]; then
    echo "Error: run_name_ not found in resolved config"
    exit 1
fi

if [ -z "$NGPUS_PER_NODE" ] || [ "$NGPUS_PER_NODE" = "0" ]; then
    echo "Error: ngpus_per_node_ must be specified in config"
    exit 1
fi

# Rename for consistency with rest of script
GPUS_PER_NODE="$NGPUS_PER_NODE"

# Set output directory
RUN_DIR="$ROOT_REMOTE/validation/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_run_outputs"

# Clear old slurm logs to start fresh
rm -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err"

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/$RUN_NAME"

# Create symlink in repo for easy local browsing
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/$RUN_NAME"
echo "  Local symlink: $REPO_ROOT/runs/$RUN_NAME"

# Copy code to run directory (or use current repo)
if [ "$COPY_CODE" == "true" ]; then
    # Check if code directory already exists
    if [ -d "$RUN_DIR/src" ] || [ -d "$RUN_DIR/projects/proteinfoundation/proteinfoundation" ]; then
        if [ "$OVERWRITE_RUNDIR" != "true" ]; then
            echo "Error: Code directory already exists at $RUN_DIR"
            echo "  The run directory contains existing code from a previous run."
            echo "  To create a new run directory, choose a different run_name_"
            echo "  To overwrite, use the --overwrite-rundir flag"
            echo "  To skip copying and use existing code, use the --no-copy flag"
            exit 1
        fi
        echo "Warning: Overwriting existing code directory at $RUN_DIR"
    fi

    print_header "Copying Code"
    echo "  $REPO_ROOT -> $RUN_DIR"

    # Copy essential directories (matching PYTHONPATH)
    rsync -a "$REPO_ROOT/src/" "$RUN_DIR/src/"
    mkdir -p "$RUN_DIR/models/rf3" && rsync -a "$REPO_ROOT/models/rf3/src/" "$RUN_DIR/models/rf3/src/"
    mkdir -p "$RUN_DIR/lib/atomworks" && rsync -a "$REPO_ROOT/lib/atomworks/src/" "$RUN_DIR/lib/atomworks/src/"

    echo "  Copying projects/proteinfoundation..."
    mkdir -p "$RUN_DIR/projects/proteinfoundation"
    rsync -a --exclude='store' --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' \
        "$REPO_ROOT/projects/proteinfoundation/proteinfoundation/" "$RUN_DIR/projects/proteinfoundation/proteinfoundation/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/configs/" "$RUN_DIR/projects/proteinfoundation/configs/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/benchmarks/" "$RUN_DIR/projects/proteinfoundation/benchmarks/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/data/" "$RUN_DIR/projects/proteinfoundation/data/"

    cp "$REPO_ROOT/.project-root" "$RUN_DIR/"
    mkdir -p "$RUN_DIR/.ipd/shebang" && rsync -a "$REPO_ROOT/.ipd/shebang/" "$RUN_DIR/.ipd/shebang/"
    mkdir -p "$RUN_DIR/.ipd/apptainer"

    # DIGS: recreate container symlink with /mnt/net prefix for compute nodes
    if [ "$CLUSTER_TYPE" = "digs" ] && [ -L "$REPO_ROOT/.ipd/apptainer/latent-dev.sif" ]; then
        CONTAINER_TARGET="$(readlink -f "$REPO_ROOT/.ipd/apptainer/latent-dev.sif" 2>/dev/null || true)"
        if [ -n "$CONTAINER_TARGET" ]; then
            [[ "$CONTAINER_TARGET" == /net/* ]] && CONTAINER_TARGET="/mnt${CONTAINER_TARGET}"
        ln -sf "$CONTAINER_TARGET" "$RUN_DIR/.ipd/apptainer/latent-dev.sif"
        fi
    fi
    CODE_DIR="$RUN_DIR"
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

# Create .env file
cat > "$RUN_DIR/.env" << EOF
# Proteina Environment Variables
DATA_PATH=$DATA_PATH
CACHE_DIR=$CACHE_DIR
PDB_MIRROR_PATH=$PDB_MIRROR_PATH
CCD_MIRROR_PATH=$CCD_MIRROR_PATH
FOLDSEEK_EXEC=$FOLDSEEK_REMOTE
HBPLUS_PATH=$HBPLUS_PATH
MOLPROBITY_PATH=$MOLPROBITY_PATH
MACE_MODEL_PATH=$MACE_MODEL_PATH
LOGURU_LEVEL='INFO'
EOF


# Calculate SLURM array size based on gpus-per-job
NUM_ARRAY_TASKS=$(( (NJOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

print_header "Job Configuration"
echo "  Run name: $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  Extra args: $EXTRA_ARGS"
echo "  Total jobs: $NJOBS"
echo "  SLURM array tasks: $NUM_ARRAY_TASKS"
echo "  GPUs per SLURM job: $GPUS_PER_JOB"
echo "  Nodes: $NNODES"
echo "  CPUs per task: $CPUS_PER_TASK"
echo ""

# Build SBATCH script
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES="$NNODES"
GPUS_PER_NODE="$GPUS_PER_JOB"
NTASKS_PER_NODE=1  # Single task that spawns multiple Lightning jobs internally
build_sbatch_header > "$SBATCH_FILE"

# Add array directive if using multiple tasks
if [ "$NUM_ARRAY_TASKS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NUM_ARRAY_TASKS - 1))" >> "$SBATCH_FILE"
fi

# Add output/error paths
cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err

# Environment setup
export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/projects/proteinfoundation:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Job array and multi-GPU configuration
ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
NJOBS=$NJOBS

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB"
EOF

# Add DATASET_SQUASH_MOUNTS if specified (for Apptainer squashfs bind mounts)
if [ -n "$DATASET_SQUASH_MOUNTS" ]; then
    echo "export DATASET_SQUASH_MOUNTS=\"$DATASET_SQUASH_MOUNTS\"" >> "$SBATCH_FILE"
fi

# Add cuEquivariance if specified
if [ -n "$USE_CUEQUIVARIANCE" ]; then
    echo "export SHOULD_USE_CUEQUIVARIANCE=$USE_CUEQUIVARIANCE" >> "$SBATCH_FILE"
fi

# Add distributed generation coordination (required for multi-GPU DDP)
cat >> "$SBATCH_FILE" << 'EOF'

# Distributed generation coordination (unique port per job, master = first node)
export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
EOF

# Add container execution command
SCRIPT_GENERATE="$CODE_DIR/projects/proteinfoundation/proteinfoundation/generate.py"

if [ "$CLUSTER_TYPE" == "digs" ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

# Run offline validation - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running validation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 \\
            --kill-on-bad-exit $PYTHON $SCRIPT_GENERATE \\
            inference_experiment=inference_on_val_set \\
            out_dir=$RUN_DIR/validation_outputs \\
            ckpt_path='$CKPT_PATH' \\
            'ckpt_name="$CKPT_NAME"' \\
            $EXTRA_ARGS \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            offline_validation=true \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
else
    CONTAINER_CMD=$(get_srun_cmd)
    # OCI/GB200: srun with container flags

    cat >> "$SBATCH_FILE" << 'EOF'

# Run offline validation - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running validation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 $CONTAINER_CMD $PYTHON $SCRIPT_GENERATE \\
            inference_experiment=inference_on_val_set \\
            out_dir=$RUN_DIR/validation_outputs \\
            ckpt_path='$CKPT_PATH' \\
            'ckpt_name="$CKPT_NAME"' \\
            $EXTRA_ARGS \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            offline_validation=true \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
fi

print_header "Submitting Job"
echo "SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

# Check for already running job
EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID | Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
    read -p "Continue with submission? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

if [ -n "$WANDBKEY" ]; then
    export WANDB_API_KEY="$WANDBKEY"
fi

# Submit job
SBATCH_OUTPUT=$(sbatch slurm_batch_script.sh)
JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')
echo "$SBATCH_OUTPUT"

echo ""
echo "Monitor: squeue -u $USER --name=$RUN_NAME"
echo "Logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"

# Wait for jobs to complete and concatenate results
if [ "$WAIT_FOR_COMPLETION" == "true" ]; then
    print_header "Waiting for Jobs to Complete"
    echo "Will concatenate results once all jobs finish..."
    echo ""

    WAIT_COUNT=0
    while true; do
        # Check if any jobs are still running or pending
        INCOMPLETE_JOBS=$(squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | wc -l)

        if [ "$INCOMPLETE_JOBS" -eq 0 ]; then
            echo "All jobs finished running!"
            break
        fi

        # Show status every 10 minutes
        if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
            STATES=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%T" 2>/dev/null | sort | uniq -c)
            echo "$(date '+%H:%M:%S') - Jobs still active: $INCOMPLETE_JOBS"
            echo "$STATES" | sed 's/^/  /'
        fi

        sleep 60
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done

    echo ""
    print_header "Concatenating Results"

    # Run the concatenation script (uses same script as external eval)
    CONCAT_SCRIPT="$CODE_DIR/projects/proteinfoundation/proteinfoundation/inference_utils/concatenate_external_eval_results.py"
    VAL_METRICS_DIR="$RUN_DIR/validation_outputs/val_metrics"

    if [ "$CLUSTER_TYPE" == "digs" ]; then
        $PYTHON "$CONCAT_SCRIPT" --output-dir "$VAL_METRICS_DIR" --cluster-with-foldseek
    else
        # Build srun command with account/partition for standalone execution
        CONTAINER_CMD=$(get_srun_cmd)
        SRUN_ARGS="--kill-on-bad-exit"
        if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
            SRUN_ARGS="$SRUN_ARGS --account=$ACCOUNT"
        fi
        if [ -n "$PARTITION" ]; then
            SRUN_ARGS="$SRUN_ARGS --partition=$PARTITION"
        fi
        srun $SRUN_ARGS $CONTAINER_CMD --container-workdir=$RUN_DIR \
            $PYTHON "$CONCAT_SCRIPT" --output-dir "$VAL_METRICS_DIR" --cluster-with-foldseek
    fi

    echo ""
    echo "Results concatenated successfully!"
    echo "  Per-example metrics: $VAL_METRICS_DIR/per_example_metrics.csv"
    echo "  Aggregated metrics:  $VAL_METRICS_DIR/aggregated_metrics.csv"
    echo "  Individual job results moved to: $VAL_METRICS_DIR/individual_job_results/"
fi

# Follow logs if requested
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "\nFollowing logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
            sleep 2
        done
        tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
    fi
